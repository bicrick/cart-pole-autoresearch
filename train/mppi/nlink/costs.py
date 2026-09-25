"""Per-goal MPPI cost and anchor parameters for the n-link kernels.

The triple's ``mppi.costs.CostConfig`` weights and structure, summed over n
links (angle error, spin, cart, soft/dead track barrier, energy gap, LQR
terminal inside the near region, energy terminal outside it). ``pack`` lays
them out for ``kernel_cpu`` / ``kernel_cuda``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mppi.costs import CostConfig
from mppi.nlink.goals import goal_angles
from mppi.nlink.lqr import design
from mppi.nlink.plant import NLinkPlant

# Scalar layout shared by both kernels.
FIELDS = (
    "M", "g", "b", "dt", "fmax", "e_target",
    "w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft", "w_barrier",
    "x_dead", "w_dead", "w_energy", "w_u", "w_du", "w_terminal_lqr", "w_terminal_energy",
    "gate_in", "gate_out", "anchor_on", "knot", "early_exit",
)
IDX = {k: i for i, k in enumerate(FIELDS)}
# Per-link rows of the `links` array.
ROW_M, ROW_L, ROW_C, ROW_S, ROW_T = range(5)


def potential(p: NLinkPlant, angles: tuple[float, ...]) -> float:
    y, pe = 0.0, 0.0
    for m, l, t in zip(p.m, p.l, angles):
        y += l * math.cos(t)
        pe += m * p.g * y
    return pe


@dataclass
class Packed:
    goal: str
    scal: np.ndarray  # [len(FIELDS)]
    links: np.ndarray  # [5, n]: m, l, c, S, target
    K: np.ndarray  # [2 + 2n]
    P: np.ndarray  # [2 + 2n, 2 + 2n]
    unstable_rate: float


def pack(p: NLinkPlant, goal: str, *, knot: int, gate_in: float = 0.02, gate_out: float = 0.10,
         early_exit: bool = True, cost: CostConfig | None = None) -> Packed:
    c = cost or CostConfig()
    ang = goal_angles(goal)
    lq = design(p, goal, c.lqr_q_theta, c.lqr_r)
    vals = {
        "M": p.M, "g": p.g, "b": p.b, "dt": p.dt, "fmax": p.force_limit, "e_target": potential(p, ang),
        **{k: getattr(c, k) for k in ("w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft",
                                      "w_barrier", "x_dead", "w_dead", "w_energy", "w_u", "w_du",
                                      "w_terminal_lqr", "w_terminal_energy")},
        "gate_in": gate_in, "gate_out": gate_out, "anchor_on": 1.0, "knot": float(knot),
        "early_exit": 1.0 if early_exit else 0.0,
    }
    scal = np.array([float(vals[k]) for k in FIELDS])
    links = np.array([p.m, p.l, p.c, p.S, ang], dtype=np.float64)
    return Packed(goal, scal, links, lq.K.astype(np.float64), lq.P.astype(np.float64), lq.unstable_rate)
