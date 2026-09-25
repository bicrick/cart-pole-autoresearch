"""Fused multicore MPPI rollout (numba), same math as the torch path.

``rollout_costs`` evaluates N residual plans from N start states: each sample
runs its whole horizon (anchor + residual -> plant step -> running cost) in a
tight compiled loop, samples spread across all cores with ``prange``. This
replaces ~180 small torch kernels per replan and is what lets the teacher run
in real time. Parity with ``MPPI.rollout_costs`` (torch): ``test_fast_rollout``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from physics_triple import _MAX_ACC, _MAX_ANG_VEL, _MAX_CART_VEL

# Packed parameter layout (see pack_params).
_P_FIELDS = (
    "M", "m1", "m2", "m3", "l1", "l2", "l3", "g", "b", "c1", "c2", "c3", "dt", "fmax",
    "t1", "t2", "t3", "e_target",
    "w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft", "w_barrier",
    "x_dead", "w_dead", "w_energy", "w_u", "w_du", "w_terminal_lqr", "w_terminal_energy",
    "gate_in", "gate_out", "anchor_on", "knot",
)
IDX = {k: i for i, k in enumerate(_P_FIELDS)}


def pack_params(plant, cost, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(scalars, K_lqr[8], P[8,8]) as float64 arrays."""
    c = cost.cfg
    t = cost.target_angles.double().tolist()
    vals = {
        "M": plant.M, "m1": plant.m1, "m2": plant.m2, "m3": plant.m3,
        "l1": plant.l1, "l2": plant.l2, "l3": plant.l3, "g": plant.g, "b": plant.b,
        "c1": plant.c1, "c2": plant.c2, "c3": plant.c3, "dt": plant.dt, "fmax": plant.force_limit,
        "t1": t[0], "t2": t[1], "t3": t[2], "e_target": cost.e_target,
        "w_angle": c.w_angle, "w_vel": c.w_vel, "w_vel_near": c.w_vel_near, "near_scale": c.near_scale,
        "w_x": c.w_x, "w_xd": c.w_xd, "x_soft": c.x_soft, "w_barrier": c.w_barrier,
        "x_dead": c.x_dead, "w_dead": c.w_dead, "w_energy": c.w_energy, "w_u": c.w_u, "w_du": c.w_du,
        "w_terminal_lqr": c.w_terminal_lqr, "w_terminal_energy": c.w_terminal_energy,
        "gate_in": cfg.gate_in, "gate_out": cfg.gate_out, "anchor_on": 1.0 if cfg.anchor else 0.0,
        "knot": float(cfg.knot),
    }
    if plant.walls:
        raise ValueError("fast rollout models the wall-free plant only")
    scal = np.array([float(vals[k]) for k in _P_FIELDS], dtype=np.float64)
    K = cost.K_lqr.double().numpy().astype(np.float64)
    P = cost.P.double().numpy().astype(np.float64)
    return scal, K, P


@njit(inline="always", fastmath=False)
def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@njit(inline="always", fastmath=False)
def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@njit(inline="always", fastmath=False)
def _fin(v, bad):
    return v if math.isfinite(v) else bad


@njit(inline="always", fastmath=False)
def _pole_energy(p, s1, c1, w1, s2, c2, w2, s3, c3, w3):
    """Pole KE + PE in the cart frame from cached sin/cos."""
    l1, l2, l3 = p[4], p[5], p[6]
    v1x = l1 * c1 * w1
    v1y = -l1 * s1 * w1
    v2x = v1x + l2 * c2 * w2
    v2y = v1y - l2 * s2 * w2
    v3x = v2x + l3 * c3 * w3
    v3y = v2y - l3 * s3 * w3
    ke = 0.5 * (p[1] * (v1x * v1x + v1y * v1y) + p[2] * (v2x * v2x + v2y * v2y) + p[3] * (v3x * v3x + v3y * v3y))
    y1 = l1 * c1
    y2 = y1 + l2 * c2
    y3 = y2 + l3 * c3
    return ke + p[7] * (p[1] * y1 + p[2] * y2 + p[3] * y3)


@njit(fastmath=False)
def _rollout_one(p, K, P, s0, U, n_knots):
    """One sample's horizon. sin/cos of each angle are computed once per step and
    shared by the anchor, dynamics (difference angles via addition identities),
    running cost, and energy."""
    M, m1, m2, m3 = p[0], p[1], p[2], p[3]
    l1, l2, l3, g = p[4], p[5], p[6], p[7]
    b, cd1, cd2, cd3 = p[8], p[9], p[10], p[11]
    dt, fmax = p[12], p[13]
    t1, t2, t3, e_t = p[14], p[15], p[16], p[17]
    w_angle, w_vel, w_vel_near, near_scale = p[18], p[19], p[20], p[21]
    w_x, w_xd, x_soft, w_barrier = p[22], p[23], p[24], p[25]
    x_dead, w_dead, w_energy, w_u, w_du = p[26], p[27], p[28], p[29], p[30]
    w_tl, w_te = p[31], p[32]
    g_in, g_out, anchor_on = p[33], p[34], p[35]
    knot = int(p[36])
    ct1, st1 = math.cos(t1), math.sin(t1)
    ct2, st2 = math.cos(t2), math.sin(t2)
    ct3, st3 = math.cos(t3), math.sin(t3)
    m123 = m1 + m2 + m3
    m23 = m2 + m3
    eps = 1e-8
    a00 = M + m123 + eps
    a11 = m123 * l1 * l1 + eps
    a22 = m23 * l2 * l2 + eps
    a33 = m3 * l3 * l3 + eps

    x, xd, th1, w1, th2, w2, th3, w3 = s0[0], s0[1], s0[2], s0[3], s0[4], s0[5], s0[6], s0[7]
    s1, c1 = math.sin(th1), math.cos(th1)
    s2, c2 = math.sin(th2), math.cos(th2)
    s3, c3 = math.sin(th3), math.cos(th3)
    acc = 0.0
    for j in range(n_knots):
        r = U[j]
        for _ in range(knot):
            # --- gated target-LQR anchor + residual
            fb_u = 0.0
            if anchor_on > 0.0:
                e1 = _wrap(th1 - t1)
                e2 = _wrap(th2 - t2)
                e3 = _wrap(th3 - t3)
                d = (1.0 - (c1 * ct1 + s1 * st1)) + (1.0 - (c2 * ct2 + s2 * st2)) + (1.0 - (c3 * ct3 + s3 * st3))
                gate = _clip((g_out - d) / (g_out - g_in), 0.0, 1.0)
                if gate > 0.0:
                    fb = K[0] * x + K[1] * xd + K[2] * e1 + K[3] * w1 + K[4] * e2 + K[5] * w2 + K[6] * e3 + K[7] * w3
                    fb_u = -gate * fb
            u = _clip(fb_u + r, -fmax, fmax)
            if not math.isfinite(u):
                u = 0.0
            # --- plant step (dynamics_fast.step)
            xd = _clip(xd, -_MAX_CART_VEL, _MAX_CART_VEL)
            w1 = _clip(w1, -_MAX_ANG_VEL, _MAX_ANG_VEL)
            w2 = _clip(w2, -_MAX_ANG_VEL, _MAX_ANG_VEL)
            w3 = _clip(w3, -_MAX_ANG_VEL, _MAX_ANG_VEL)
            s12 = s1 * c2 - c1 * s2
            c12 = c1 * c2 + s1 * s2
            s13 = s1 * c3 - c1 * s3
            c13 = c1 * c3 + s1 * s3
            s23 = s2 * c3 - c2 * s3
            c23 = c2 * c3 + s2 * s3
            a01 = m123 * l1 * c1
            a02 = m23 * l2 * c2
            a03 = m3 * l3 * c3
            a12 = m23 * l1 * l2 * c12
            a13 = m3 * l1 * l3 * c13
            a23 = m3 * l2 * l3 * c23
            q1s, q2s, q3s = w1 * w1, w2 * w2, w3 * w3
            r0 = u - b * xd + m123 * l1 * s1 * q1s + m23 * l2 * s2 * q2s + m3 * l3 * s3 * q3s
            r1 = -cd1 * w1 - m23 * l1 * l2 * s12 * q2s - m3 * l1 * l3 * s13 * q3s + m123 * g * l1 * s1
            r2 = -cd2 * w2 + m23 * l1 * l2 * s12 * q1s - m3 * l2 * l3 * s23 * q3s + m23 * g * l2 * s2
            r3 = -cd3 * w3 + m3 * l1 * l3 * s13 * q1s + m3 * l2 * l3 * s23 * q2s + m3 * g * l3 * s3
            d0 = a00
            l10 = a01 / d0
            l20 = a02 / d0
            l30 = a03 / d0
            d1 = a11 - l10 * l10 * d0
            l21 = (a12 - l20 * l10 * d0) / d1
            l31 = (a13 - l30 * l10 * d0) / d1
            d2 = a22 - l20 * l20 * d0 - l21 * l21 * d1
            l32 = (a23 - l30 * l20 * d0 - l31 * l21 * d1) / d2
            d3 = a33 - l30 * l30 * d0 - l31 * l31 * d1 - l32 * l32 * d2
            y0 = r0
            y1 = r1 - l10 * y0
            y2 = r2 - l20 * y0 - l21 * y1
            y3 = r3 - l30 * y0 - l31 * y1 - l32 * y2
            xa3 = y3 / d3
            xa2 = y2 / d2 - l32 * xa3
            xa1 = y1 / d1 - l21 * xa2 - l31 * xa3
            xa0 = y0 / d0 - l10 * xa1 - l20 * xa2 - l30 * xa3
            xa0 = _clip(_fin(xa0, 0.0), -_MAX_ACC, _MAX_ACC)
            xa1 = _clip(_fin(xa1, 0.0), -_MAX_ACC, _MAX_ACC)
            xa2 = _clip(_fin(xa2, 0.0), -_MAX_ACC, _MAX_ACC)
            xa3 = _clip(_fin(xa3, 0.0), -_MAX_ACC, _MAX_ACC)
            xd = _clip(xd + xa0 * dt, -_MAX_CART_VEL, _MAX_CART_VEL)
            w1 = _clip(w1 + xa1 * dt, -_MAX_ANG_VEL, _MAX_ANG_VEL)
            w2 = _clip(w2 + xa2 * dt, -_MAX_ANG_VEL, _MAX_ANG_VEL)
            w3 = _clip(w3 + xa3 * dt, -_MAX_ANG_VEL, _MAX_ANG_VEL)
            x = x + xd * dt
            th1 = th1 + w1 * dt
            th2 = th2 + w2 * dt
            th3 = th3 + w3 * dt
            s1, c1 = math.sin(th1), math.cos(th1)
            s2, c2 = math.sin(th2), math.cos(th2)
            s3, c3 = math.sin(th3), math.cos(th3)
            # --- running cost on the new state (costs.TargetCost.running)
            ang = (1.0 - (c1 * ct1 + s1 * st1)) + (1.0 - (c2 * ct2 + s2 * st2)) + (1.0 - (c3 * ct3 + s3 * st3))
            near = math.exp(-ang / near_scale)
            spin = w1 * w1 + w2 * w2 + w3 * w3
            ax = abs(x)
            over = ax - x_soft if ax > x_soft else 0.0
            dead = 1.0 if ax > x_dead else 0.0
            en = _pole_energy(p, s1, c1, w1, s2, c2, w2, s3, c3, w3) - e_t
            step_cost = (
                w_angle * ang
                + (w_vel + w_vel_near * near) * spin
                + w_x * x * x
                + w_xd * xd * xd
                + w_barrier * over * over
                + w_dead * dead
                + w_energy * (1.0 - near) * en * en
                + w_u * u * u
            )
            acc += step_cost * dt
    # --- terminal cost
    e = np.empty(8)
    e[0], e[1], e[2], e[3] = x, xd, _wrap(th1 - t1), w1
    e[4], e[5], e[6], e[7] = _wrap(th2 - t2), w2, _wrap(th3 - t3), w3
    quad = 0.0
    for i in range(8):
        row = 0.0
        for k in range(8):
            row += P[i, k] * e[k]
        quad += e[i] * row
    ang = (1.0 - (c1 * ct1 + s1 * st1)) + (1.0 - (c2 * ct2 + s2 * st2)) + (1.0 - (c3 * ct3 + s3 * st3))
    near = math.exp(-ang / near_scale)
    en = _pole_energy(p, s1, c1, w1, s2, c2, w2, s3, c3, w3) - e_t
    acc += near * w_tl * quad + (1.0 - near) * w_te * en * en
    # --- residual smoothness
    kdt = knot * dt
    for j in range(n_knots - 1):
        du = (U[j + 1] - U[j]) / kdt
        acc += w_du * du * du * kdt
    return acc


@njit(parallel=True, fastmath=False, cache=True, nogil=True)
def rollout_costs(p, K, P, starts, plans):
    """starts [N,8], plans [N,T] residual knots -> cost [N]."""
    n, t = plans.shape
    out = np.empty(n)
    for i in prange(n):
        out[i] = _rollout_one(p, K, P, starts[i], plans[i], t)
    return out
