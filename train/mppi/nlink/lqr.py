"""Per-goal discrete LQR for the n-link plant (the triple's recipe, any n).

Finite-difference A, B of ``plant.step`` at the equilibrium, DARE with
Q = diag(q_x, q_xd, [q_theta, q_omega] * n), R = r. Returns the gain K and the
cost-to-go P in per-second running-cost units (P * dt), like ``costs.TargetCost``.

    python -m mppi.nlink.lqr --n 4      # per-goal unstable rates
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace

import numpy as np
import torch

from lqr_uuu import solve_dare
from mppi.nlink.goals import goal_angles, goal_ids
from mppi.nlink.plant import NLinkPlant, step


@dataclass
class GoalLQR:
    goal: str
    K: np.ndarray  # [2 + 2n]
    P: np.ndarray  # [2 + 2n, 2 + 2n], per-second units
    unstable_rate: float  # fastest open-loop divergence rate at the equilibrium [1/s]


def equilibrium(p: NLinkPlant, goal: str) -> torch.Tensor:
    z = torch.zeros(p.dim, dtype=torch.float64)
    z[2::2] = torch.tensor(goal_angles(goal), dtype=torch.float64)
    return z


def linearize(p: NLinkPlant, goal: str, eps: float = 1e-5, eu: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    q = replace(p, force_limit=1e3)
    z0 = equilibrium(p, goal).unsqueeze(0)
    u0 = torch.zeros(1, dtype=torch.float64)
    A = np.zeros((p.dim, p.dim))
    for i in range(p.dim):
        zp, zm = z0.clone(), z0.clone()
        zp[0, i] += eps
        zm[0, i] -= eps
        A[:, i] = ((step(q, zp, u0) - step(q, zm, u0)) / (2 * eps)).squeeze(0).numpy()
    B = ((step(q, z0, u0 + eu) - step(q, z0, u0 - eu)) / (2 * eu)).numpy().reshape(p.dim, 1)
    return A, B


def design(p: NLinkPlant, goal: str, q_theta: float = 100.0, r: float = 0.01) -> GoalLQR:
    A, B = linearize(p, goal)
    Q = np.diag([1.0, 1.0] + [q_theta, 1.0] * p.n)
    R = np.array([[r]])
    P = solve_dare(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    rate = float(np.log(np.abs(np.linalg.eigvals(A))).max() / p.dt)
    return GoalLQR(goal=goal, K=K.reshape(-1), P=P * p.dt, unstable_rate=rate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    p = NLinkPlant.load(args.n)
    for goal in goal_ids(args.n):
        g = design(p, goal)
        dbl = f"doubles every {1000 * math.log(2) / g.unstable_rate:5.0f} ms" if g.unstable_rate > 1e-3 else "stable"
        print(f"{goal}: fastest open-loop rate {g.unstable_rate:6.2f} /s ({dbl}); |K| max {np.abs(g.K).max():8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
