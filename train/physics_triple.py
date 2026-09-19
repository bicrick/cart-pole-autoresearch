"""Batched cart + triple-pendulum dynamics. Theta = 0 is upright."""

from __future__ import annotations

import json
from pathlib import Path

import torch

CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "shared" / "constants-triple.json"

# Soft caps so impulse / stiff grabs cannot explode semi-implicit Euler into NaNs.
_MAX_CART_VEL = 30.0
_MAX_ANG_VEL = 50.0
_MAX_ACC = 1e4


def load_constants() -> dict:
    return json.loads(CONSTANTS_PATH.read_text())


def _as_tensor(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def solve_mass_4(m, rhs):
    """Solve batched symmetric 4x4 M a = rhs.

    m: [..., 4, 4], rhs: [..., 4] -> [..., 4]
    """
    # Tiny diagonal jitter for ill-conditioned hanging/fold poses.
    eye = torch.eye(4, device=m.device, dtype=m.dtype)
    m_reg = m + 1e-8 * eye
    return torch.linalg.solve(m_reg, rhs.unsqueeze(-1)).squeeze(-1)


def accelerations(state, force, extra_q=None, constants=None):
    """
    state: [..., 8] = x, xd, th1, th1d, th2, th2d, th3, th3d
    force: [...] cart force
    extra_q: optional (Qx, Qth1, Qth2, Qth3)
    returns: [..., 4] = xdd, t1dd, t2dd, t3dd
    """
    if constants is None:
        constants = load_constants()
    device = state.device
    dtype = state.dtype
    M = _as_tensor(constants["cartMass"], device, dtype)
    m1 = _as_tensor(constants["poleMass1"], device, dtype)
    m2 = _as_tensor(constants["poleMass2"], device, dtype)
    m3 = _as_tensor(constants["poleMass3"], device, dtype)
    l1 = _as_tensor(constants["poleLength1"], device, dtype)
    l2 = _as_tensor(constants["poleLength2"], device, dtype)
    l3 = _as_tensor(constants["poleLength3"], device, dtype)
    g = _as_tensor(constants["gravity"], device, dtype)
    b = _as_tensor(constants["cartFriction"], device, dtype)
    c1 = _as_tensor(constants["jointDamping1"], device, dtype)
    c2 = _as_tensor(constants["jointDamping2"], device, dtype)
    c3 = _as_tensor(constants["jointDamping3"], device, dtype)

    x, xd, th1, th1d, th2, th2d, th3, th3d = state.unbind(dim=-1)
    s1 = torch.sin(th1)
    cth1 = torch.cos(th1)
    s2 = torch.sin(th2)
    cth2 = torch.cos(th2)
    s3 = torch.sin(th3)
    cth3 = torch.cos(th3)
    s12 = torch.sin(th1 - th2)
    c12 = torch.cos(th1 - th2)
    s13 = torch.sin(th1 - th3)
    c13 = torch.cos(th1 - th3)
    s23 = torch.sin(th2 - th3)
    c23 = torch.cos(th2 - th3)

    qx = q1 = q2 = q3 = 0.0
    if extra_q is not None:
        qx, q1, q2, q3 = extra_q

    # Mass matrix (Lagrange, theta=0 upright, point masses at tips).
    m11 = M + m1 + m2 + m3
    m12 = (m1 + m2 + m3) * l1 * cth1
    m13 = (m2 + m3) * l2 * cth2
    m14 = m3 * l3 * cth3
    m22 = (m1 + m2 + m3) * l1 * l1
    m23 = (m2 + m3) * l1 * l2 * c12
    m24 = m3 * l1 * l3 * c13
    m33 = (m2 + m3) * l2 * l2
    m34 = m3 * l2 * l3 * c23
    m44 = m3 * l3 * l3

    # Build [..., 4, 4]
    lead = state.shape[:-1]
    mass = torch.zeros(*lead, 4, 4, device=device, dtype=dtype)
    mass[..., 0, 0] = m11
    mass[..., 0, 1] = mass[..., 1, 0] = m12
    mass[..., 0, 2] = mass[..., 2, 0] = m13
    mass[..., 0, 3] = mass[..., 3, 0] = m14
    mass[..., 1, 1] = m22
    mass[..., 1, 2] = mass[..., 2, 1] = m23
    mass[..., 1, 3] = mass[..., 3, 1] = m24
    mass[..., 2, 2] = m33
    mass[..., 2, 3] = mass[..., 3, 2] = m34
    mass[..., 3, 3] = m44

    rhs_x = (
        force
        - b * xd
        + (m1 + m2 + m3) * l1 * s1 * th1d * th1d
        + (m2 + m3) * l2 * s2 * th2d * th2d
        + m3 * l3 * s3 * th3d * th3d
        + qx
    )
    rhs_1 = (
        q1
        - c1 * th1d
        - (m2 + m3) * l1 * l2 * s12 * th2d * th2d
        - m3 * l1 * l3 * s13 * th3d * th3d
        + (m1 + m2 + m3) * g * l1 * s1
    )
    rhs_2 = (
        q2
        - c2 * th2d
        + (m2 + m3) * l1 * l2 * s12 * th1d * th1d
        - m3 * l2 * l3 * s23 * th3d * th3d
        + (m2 + m3) * g * l2 * s2
    )
    rhs_3 = (
        q3
        - c3 * th3d
        + m3 * l1 * l3 * s13 * th1d * th1d
        + m3 * l2 * l3 * s23 * th2d * th2d
        + m3 * g * l3 * s3
    )
    rhs = torch.stack((rhs_x, rhs_1, rhs_2, rhs_3), dim=-1)
    return solve_mass_4(mass, rhs)


def step(state, force, extra_q=None, constants=None):
    """Semi-implicit Euler. Returns next state with the same leading dims."""
    if constants is None:
        constants = load_constants()
    dt = state.new_tensor(constants["dt"])
    fmax = constants["forceLimit"]
    state = state.clone()
    state[..., 1] = state[..., 1].clamp(-_MAX_CART_VEL, _MAX_CART_VEL)
    state[..., 3] = state[..., 3].clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    state[..., 5] = state[..., 5].clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    state[..., 7] = state[..., 7].clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    force = torch.nan_to_num(force, nan=0.0, posinf=fmax, neginf=-fmax).clamp(-fmax, fmax)
    acc = accelerations(state, force, extra_q=extra_q, constants=constants)
    acc = torch.nan_to_num(acc, nan=0.0, posinf=_MAX_ACC, neginf=-_MAX_ACC).clamp(
        -_MAX_ACC, _MAX_ACC
    )
    xd = (state[..., 1] + acc[..., 0] * dt).clamp(-_MAX_CART_VEL, _MAX_CART_VEL)
    th1d = (state[..., 3] + acc[..., 1] * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th2d = (state[..., 5] + acc[..., 2] * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th3d = (state[..., 7] + acc[..., 3] * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    x = state[..., 0] + xd * dt
    th1 = state[..., 2] + th1d * dt
    th2 = state[..., 4] + th2d * dt
    th3 = state[..., 6] + th3d * dt
    # No track walls. Cart may leave |x| > trackLimit; training ends the
    # episode on oob (void → respawn).
    next_state = torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)
    return torch.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)


def observe(state):
    """11-D state features: x, xd, sin/cos ×3, ω ×3."""
    x, xd, th1, th1d, th2, th2d, th3, th3d = state.unbind(dim=-1)
    return torch.stack(
        (
            x,
            xd,
            torch.sin(th1),
            torch.cos(th1),
            torch.sin(th2),
            torch.cos(th2),
            torch.sin(th3),
            torch.cos(th3),
            th1d,
            th2d,
            th3d,
        ),
        dim=-1,
    )


def normalize_obs(obs, constants=None):
    if constants is None:
        constants = load_constants()
    obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    low = obs.new_tensor(constants["obsLow"])
    high = obs.new_tensor(constants["obsHigh"])
    mid = 0.5 * (low + high)
    scale = 0.5 * (high - low).clamp(min=1e-6)
    return ((obs - mid) / scale).clamp(-3.0, 3.0)


def tip_positions(state, constants=None):
    if constants is None:
        constants = load_constants()
    x = state[..., 0]
    th1 = state[..., 2]
    th2 = state[..., 4]
    th3 = state[..., 6]
    l1 = state.new_tensor(constants["poleLength1"])
    l2 = state.new_tensor(constants["poleLength2"])
    l3 = state.new_tensor(constants["poleLength3"])
    p1x = x + l1 * torch.sin(th1)
    p1y = l1 * torch.cos(th1)
    p2x = p1x + l2 * torch.sin(th2)
    p2y = p1y + l2 * torch.cos(th2)
    p3x = p2x + l3 * torch.sin(th3)
    p3y = p2y + l3 * torch.cos(th3)
    return p1x, p1y, p2x, p2y, p3x, p3y
