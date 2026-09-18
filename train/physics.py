"""Batched cart + double-pendulum dynamics. Theta = 0 is upright."""

from __future__ import annotations

import json
from pathlib import Path

import torch

CONSTANTS_PATH = Path(__file__).resolve().parents[1] / "shared" / "constants.json"


def load_constants() -> dict:
    return json.loads(CONSTANTS_PATH.read_text())


def _as_tensor(value, device, dtype):
    return torch.as_tensor(value, device=device, dtype=dtype)


def solve_mass(a, b, c, d, e, f, rx, r1, r2):
    """Solve symmetric 3x3 [[a,b,c],[b,d,e],[c,e,f]] x = r."""
    det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - d * c)
    det = det.clamp(min=1e-8)
    a11 = d * f - e * e
    a12 = c * e - b * f
    a13 = b * e - d * c
    a22 = a * f - c * c
    a23 = c * b - a * e
    a33 = a * d - b * b
    xdd = (a11 * rx + a12 * r1 + a13 * r2) / det
    t1dd = (a12 * rx + a22 * r1 + a23 * r2) / det
    t2dd = (a13 * rx + a23 * r1 + a33 * r2) / det
    return xdd, t1dd, t2dd


def accelerations(state, force, extra_q=None, constants=None):
    """
    state: [..., 6] = x, xd, th1, th1d, th2, th2d
    force: [...] cart force, already clipped by caller if desired
    extra_q: optional (Qx, Qth1, Qth2) generalized forces from grabs/impulses
    """
    if constants is None:
        constants = load_constants()
    device = state.device
    dtype = state.dtype
    M = _as_tensor(constants["cartMass"], device, dtype)
    m1 = _as_tensor(constants["poleMass1"], device, dtype)
    m2 = _as_tensor(constants["poleMass2"], device, dtype)
    l1 = _as_tensor(constants["poleLength1"], device, dtype)
    l2 = _as_tensor(constants["poleLength2"], device, dtype)
    g = _as_tensor(constants["gravity"], device, dtype)
    b = _as_tensor(constants["cartFriction"], device, dtype)
    c1 = _as_tensor(constants["jointDamping1"], device, dtype)
    c2 = _as_tensor(constants["jointDamping2"], device, dtype)

    x, xd, th1, th1d, th2, th2d = state.unbind(dim=-1)
    s1 = torch.sin(th1)
    cth1 = torch.cos(th1)
    s2 = torch.sin(th2)
    cth2 = torch.cos(th2)
    s12 = torch.sin(th1 - th2)
    c12 = torch.cos(th1 - th2)

    qx = q1 = q2 = 0.0
    if extra_q is not None:
        qx, q1, q2 = extra_q

    m11 = M + m1 + m2
    m12 = (m1 + m2) * l1 * cth1
    m13 = m2 * l2 * cth2
    m22 = (m1 + m2) * l1 * l1
    m23 = m2 * l1 * l2 * c12
    m33 = m2 * l2 * l2

    rhs_x = force - b * xd + (m1 + m2) * l1 * s1 * th1d * th1d + m2 * l2 * s2 * th2d * th2d + qx
    rhs_1 = q1 - c1 * th1d - m2 * l1 * l2 * s12 * th2d * th2d + (m1 + m2) * g * l1 * s1
    rhs_2 = q2 - c2 * th2d + m2 * l1 * l2 * s12 * th1d * th1d + m2 * g * l2 * s2

    xdd, t1dd, t2dd = solve_mass(m11, m12, m13, m22, m23, m33, rhs_x, rhs_1, rhs_2)
    return torch.stack((xdd, t1dd, t2dd), dim=-1)


# Soft caps so impulse / stiff grabs cannot explode semi-implicit Euler into NaNs.
_MAX_CART_VEL = 30.0
_MAX_ANG_VEL = 50.0
_MAX_ACC = 1e4


def step(state, force, extra_q=None, constants=None):
    """Semi-implicit Euler. Returns next state with the same leading dims."""
    if constants is None:
        constants = load_constants()
    dt = state.new_tensor(constants["dt"])
    fmax = constants["forceLimit"]
    # Cap inbound velocities (impulses / prior blow-ups) before mass solve.
    state = state.clone()
    state[..., 1] = state[..., 1].clamp(-_MAX_CART_VEL, _MAX_CART_VEL)
    state[..., 3] = state[..., 3].clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    state[..., 5] = state[..., 5].clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    force = torch.nan_to_num(force, nan=0.0, posinf=fmax, neginf=-fmax).clamp(-fmax, fmax)
    acc = accelerations(state, force, extra_q=extra_q, constants=constants)
    acc = torch.nan_to_num(acc, nan=0.0, posinf=_MAX_ACC, neginf=-_MAX_ACC).clamp(
        -_MAX_ACC, _MAX_ACC
    )
    xd = (state[..., 1] + acc[..., 0] * dt).clamp(-_MAX_CART_VEL, _MAX_CART_VEL)
    th1d = (state[..., 3] + acc[..., 1] * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th2d = (state[..., 5] + acc[..., 2] * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    x = state[..., 0] + xd * dt
    th1 = state[..., 2] + th1d * dt
    th2 = state[..., 4] + th2d * dt
    # No track walls. Cart may leave |x| > trackLimit; training ends the
    # episode on oob (void → respawn). Web demo mirrors that.
    next_state = torch.stack((x, xd, th1, th1d, th2, th2d), dim=-1)
    return torch.nan_to_num(next_state, nan=0.0, posinf=0.0, neginf=0.0)


def observe(state):
    x, xd, th1, th1d, th2, th2d = state.unbind(dim=-1)
    return torch.stack(
        (x, xd, torch.sin(th1), torch.cos(th1), torch.sin(th2), torch.cos(th2), th1d, th2d),
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


def reward(state, force, constants=None):
    if constants is None:
        constants = load_constants()
    x, xd, th1, th1d, th2, th2d = state.unbind(dim=-1)
    upright = torch.cos(th1) + torch.cos(th2)
    center = 0.02 * (x * x + 0.1 * xd * xd)
    spin = 0.002 * (th1d * th1d + th2d * th2d)
    effort = 0.001 * (force / constants["forceLimit"]) ** 2
    return upright - center - spin - effort


def tip_positions(state, constants=None):
    if constants is None:
        constants = load_constants()
    x = state[..., 0]
    th1 = state[..., 2]
    th2 = state[..., 4]
    l1 = state.new_tensor(constants["poleLength1"])
    l2 = state.new_tensor(constants["poleLength2"])
    p1x = x + l1 * torch.sin(th1)
    p1y = l1 * torch.cos(th1)
    p2x = p1x + l2 * torch.sin(th2)
    p2y = p1y + l2 * torch.cos(th2)
    return p1x, p1y, p2x, p2y


def grab_forces(state, target_xy, body, stiffness, damping, constants=None):
    """PD force at a grabbed body. body: 0 cart, 1 lower tip, 2 upper tip."""
    if constants is None:
        constants = load_constants()
    x, xd, th1, th1d, th2, th2d = state.unbind(dim=-1)
    p1x, p1y, p2x, p2y = tip_positions(state, constants)
    l1 = state.new_tensor(constants["poleLength1"])
    l2 = state.new_tensor(constants["poleLength2"])
    zeros = torch.zeros_like(x)
    px = torch.where(body == 2, p2x, torch.where(body == 1, p1x, x))
    py = torch.where(body == 2, p2y, torch.where(body == 1, p1y, zeros))
    vx = torch.where(
        body == 2,
        xd + l1 * torch.cos(th1) * th1d + l2 * torch.cos(th2) * th2d,
        torch.where(body == 1, xd + l1 * torch.cos(th1) * th1d, xd),
    )
    vy = torch.where(
        body == 2,
        -l1 * torch.sin(th1) * th1d - l2 * torch.sin(th2) * th2d,
        torch.where(body == 1, -l1 * torch.sin(th1) * th1d, zeros),
    )
    fx = stiffness * (target_xy[..., 0] - px) - damping * vx
    fy = stiffness * (target_xy[..., 1] - py) - damping * vy
    qx = fx
    q1 = torch.where(
        body >= 1,
        l1 * torch.cos(th1) * fx - l1 * torch.sin(th1) * fy,
        zeros,
    )
    q2 = torch.where(
        body == 2,
        l2 * torch.cos(th2) * fx - l2 * torch.sin(th2) * fy,
        zeros,
    )
    return qx, q1, q2
