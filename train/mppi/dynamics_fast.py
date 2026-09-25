"""Batched cart-triple step with a closed-form 4x4 SPD solve.

Same equations, clips, and wall logic as ``physics_triple.step``. The mass
matrix is solved by an unrolled LDL^T so a batch of thousands of rollouts is a
handful of elementwise kernels instead of ``torch.linalg.solve``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from physics_triple import _MAX_ACC, _MAX_ANG_VEL, _MAX_CART_VEL, load_constants


@dataclass(frozen=True)
class Plant:
    M: float
    m1: float
    m2: float
    m3: float
    l1: float
    l2: float
    l3: float
    g: float
    b: float
    c1: float
    c2: float
    c3: float
    dt: float
    force_limit: float
    track_limit: float
    walls: bool

    @classmethod
    def from_constants(cls, constants: dict | None = None, *, walls: bool | None = None) -> "Plant":
        c = constants or load_constants()
        return cls(
            M=float(c["cartMass"]),
            m1=float(c["poleMass1"]),
            m2=float(c["poleMass2"]),
            m3=float(c["poleMass3"]),
            l1=float(c["poleLength1"]),
            l2=float(c["poleLength2"]),
            l3=float(c["poleLength3"]),
            g=float(c["gravity"]),
            b=float(c["cartFriction"]),
            c1=float(c["jointDamping1"]),
            c2=float(c["jointDamping2"]),
            c3=float(c["jointDamping3"]),
            dt=float(c["dt"]),
            force_limit=float(c["forceLimit"]),
            track_limit=float(c.get("trackLimit", 2.4)),
            walls=bool(c.get("trackWalls", False)) if walls is None else bool(walls),
        )


def _solve_spd4(a00, a01, a02, a03, a11, a12, a13, a22, a23, a33, r0, r1, r2, r3):
    """Solve symmetric positive-definite 4x4 A x = r elementwise (LDL^T)."""
    eps = 1e-8
    d0 = a00 + eps
    l10 = a01 / d0
    l20 = a02 / d0
    l30 = a03 / d0
    d1 = a11 + eps - l10 * l10 * d0
    l21 = (a12 - l20 * l10 * d0) / d1
    l31 = (a13 - l30 * l10 * d0) / d1
    d2 = a22 + eps - l20 * l20 * d0 - l21 * l21 * d1
    l32 = (a23 - l30 * l20 * d0 - l31 * l21 * d1) / d2
    d3 = a33 + eps - l30 * l30 * d0 - l31 * l31 * d1 - l32 * l32 * d2

    y0 = r0
    y1 = r1 - l10 * y0
    y2 = r2 - l20 * y0 - l21 * y1
    y3 = r3 - l30 * y0 - l31 * y1 - l32 * y2

    x3 = y3 / d3
    x2 = y2 / d2 - l32 * x3
    x1 = y1 / d1 - l21 * x2 - l31 * x3
    x0 = y0 / d0 - l10 * x1 - l20 * x2 - l30 * x3
    return x0, x1, x2, x3


def accelerations(p: Plant, state: torch.Tensor, force: torch.Tensor, extra=None):
    x, xd, th1, th1d, th2, th2d, th3, th3d = state.unbind(-1)
    s1, c1_ = torch.sin(th1), torch.cos(th1)
    s2, c2_ = torch.sin(th2), torch.cos(th2)
    s3, c3_ = torch.sin(th3), torch.cos(th3)
    d12 = th1 - th2
    d13 = th1 - th3
    d23 = th2 - th3
    s12, c12 = torch.sin(d12), torch.cos(d12)
    s13, c13 = torch.sin(d13), torch.cos(d13)
    s23, c23 = torch.sin(d23), torch.cos(d23)

    m123 = p.m1 + p.m2 + p.m3
    m23 = p.m2 + p.m3
    a00 = torch.full_like(x, p.M + m123)
    a01 = m123 * p.l1 * c1_
    a02 = m23 * p.l2 * c2_
    a03 = p.m3 * p.l3 * c3_
    a11 = torch.full_like(x, m123 * p.l1 * p.l1)
    a12 = m23 * p.l1 * p.l2 * c12
    a13 = p.m3 * p.l1 * p.l3 * c13
    a22 = torch.full_like(x, m23 * p.l2 * p.l2)
    a23 = p.m3 * p.l2 * p.l3 * c23
    a33 = torch.full_like(x, p.m3 * p.l3 * p.l3)

    w1 = th1d * th1d
    w2 = th2d * th2d
    w3 = th3d * th3d
    r0 = force - p.b * xd + m123 * p.l1 * s1 * w1 + m23 * p.l2 * s2 * w2 + p.m3 * p.l3 * s3 * w3
    r1 = -p.c1 * th1d - m23 * p.l1 * p.l2 * s12 * w2 - p.m3 * p.l1 * p.l3 * s13 * w3 + m123 * p.g * p.l1 * s1
    r2 = -p.c2 * th2d + m23 * p.l1 * p.l2 * s12 * w1 - p.m3 * p.l2 * p.l3 * s23 * w3 + m23 * p.g * p.l2 * s2
    r3 = -p.c3 * th3d + p.m3 * p.l1 * p.l3 * s13 * w1 + p.m3 * p.l2 * p.l3 * s23 * w2 + p.m3 * p.g * p.l3 * s3
    if extra is not None:
        qx, q1, q2, q3 = extra
        r0, r1, r2, r3 = r0 + qx, r1 + q1, r2 + q2, r3 + q3
    return _solve_spd4(a00, a01, a02, a03, a11, a12, a13, a22, a23, a33, r0, r1, r2, r3)


def pole_energy(p: Plant, state: torch.Tensor) -> torch.Tensor:
    """Pole kinetic + potential energy in the cart frame (cart velocity excluded)."""
    _, _, th1, th1d, th2, th2d, th3, th3d = state.unbind(-1)
    v1x = p.l1 * torch.cos(th1) * th1d
    v1y = -p.l1 * torch.sin(th1) * th1d
    v2x = v1x + p.l2 * torch.cos(th2) * th2d
    v2y = v1y - p.l2 * torch.sin(th2) * th2d
    v3x = v2x + p.l3 * torch.cos(th3) * th3d
    v3y = v2y - p.l3 * torch.sin(th3) * th3d
    ke = 0.5 * (p.m1 * (v1x * v1x + v1y * v1y) + p.m2 * (v2x * v2x + v2y * v2y) + p.m3 * (v3x * v3x + v3y * v3y))
    return ke + potential_energy(p, th1, th2, th3)


def potential_energy(p: Plant, th1, th2, th3):
    y1 = p.l1 * torch.cos(th1)
    y2 = y1 + p.l2 * torch.cos(th2)
    y3 = y2 + p.l3 * torch.cos(th3)
    return p.g * (p.m1 * y1 + p.m2 * y2 + p.m3 * y3)


def step(p: Plant, state: torch.Tensor, force: torch.Tensor, extra=None) -> torch.Tensor:
    """Semi-implicit Euler, identical contract to ``physics_triple.step``.

    ``extra`` is an optional generalized force (Qx, Q1, Q2, Q3), e.g. a mouse drag.
    """
    fmax = p.force_limit
    x, xd, th1, th1d, th2, th2d, th3, th3d = state.unbind(-1)
    xd = xd.clamp(-_MAX_CART_VEL, _MAX_CART_VEL)
    th1d = th1d.clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th2d = th2d.clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th3d = th3d.clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    force = torch.nan_to_num(force, nan=0.0, posinf=fmax, neginf=-fmax).clamp(-fmax, fmax)
    clean = torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)
    acc = accelerations(p, clean, force, extra)
    ax, a1, a2, a3 = (
        torch.nan_to_num(a, nan=0.0, posinf=_MAX_ACC, neginf=-_MAX_ACC).clamp(-_MAX_ACC, _MAX_ACC) for a in acc
    )
    dt = p.dt
    xd = (xd + ax * dt).clamp(-_MAX_CART_VEL, _MAX_CART_VEL)
    th1d = (th1d + a1 * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th2d = (th2d + a2 * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    th3d = (th3d + a3 * dt).clamp(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    x = x + xd * dt
    th1 = th1 + th1d * dt
    th2 = th2 + th2d * dt
    th3 = th3 + th3d * dt
    if p.walls:
        hit = (x > p.track_limit) | (x < -p.track_limit)
        x = x.clamp(-p.track_limit, p.track_limit)
        xd = torch.where(hit, torch.zeros_like(xd), xd)
    nxt = torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)
    return torch.nan_to_num(nxt, nan=0.0, posinf=0.0, neginf=0.0)
