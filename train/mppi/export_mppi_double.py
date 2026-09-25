#!/usr/bin/env python3
"""Export the MPPI controller for the cart + double pendulum (4 goals).

Same recipe as the triple (``export_mppi``): per goal a target-LQR anchor and
cost-to-go P (DARE on a finite-difference linearization of ``physics.step``),
the triple's running/terminal cost weights, and one MPPI config. The browser
rolls samples out with ``web/src/mppi/rollout-double.js``, which follows the
page's plant (``web/src/physics.js``) exactly.

Writes ``web/public/mppi/double.json``.

    python -m mppi.export_mppi_double [--mppi '{"n_samples": 1024}']
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from lqr_uuu import solve_dare
from mppi.costs import CostConfig
from physics import load_constants, step

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "mppi" / "double.json"

GOALS = {"UU": (0.0, 0.0), "UD": (0.0, math.pi), "DU": (math.pi, 0.0), "DD": (math.pi, math.pi)}

FIELDS = (
    "M", "m1", "m2", "l1", "l2", "g", "b", "c1", "c2", "dt", "fmax",
    "t1", "t2", "e_target",
    "w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft", "w_barrier",
    "x_dead", "w_dead", "w_energy", "w_u", "w_du", "w_terminal_lqr", "w_terminal_energy",
    "gate_in", "gate_out", "anchor_on", "knot", "sub", "early_exit",
)

# Noise scales follow the 20 N force limit (the triple's 8 / 25 N are for 40 N).
MPPI = {
    "n_samples": 2048, "n_knots": 45, "knot": 4, "sigma": 4.0, "sigma_wide": 12.5, "wide_frac": 0.25,
    "noise_beta": 0.7, "ess": 32.0, "n_iters": 1, "gate_in": 0.02, "gate_out": 0.10,
    "rollout_sub": 1, "early_exit": True, "hold_samples": 128, "hold_d": 0.01, "hold_spin": 0.5,
}


def lqr(consts: dict, target: tuple[float, float], q_theta: float, r: float, dt: float):
    """Discrete LQR at the equilibrium -> (K [6], P [6,6] in per-second cost units)."""
    c = dict(consts)
    c["forceLimit"] = 1e3
    z0 = torch.tensor([[0.0, 0.0, target[0], 0.0, target[1], 0.0]], dtype=torch.float64)
    u0 = torch.zeros(1, dtype=torch.float64)
    eps, eu = 1e-5, 1e-3
    A = np.zeros((6, 6))
    for i in range(6):
        zp, zm = z0.clone(), z0.clone()
        zp[0, i] += eps
        zm[0, i] -= eps
        A[:, i] = ((step(zp, u0, constants=c) - step(zm, u0, constants=c)) / (2 * eps)).squeeze(0).numpy()
    B = ((step(z0, u0 + eu, constants=c) - step(z0, u0 - eu, constants=c)) / (2 * eu)).numpy().reshape(6, 1)
    Q = np.diag([1.0, 1.0, q_theta, 1.0, q_theta, 1.0])
    R = np.array([[r]])
    P = solve_dare(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K.reshape(-1), P * dt


def potential(consts: dict, t1: float, t2: float) -> float:
    y1 = consts["poleLength1"] * math.cos(t1)
    y2 = y1 + consts["poleLength2"] * math.cos(t2)
    return consts["gravity"] * (consts["poleMass1"] * y1 + consts["poleMass2"] * y2)


def export(overrides: dict | None = None) -> dict:
    consts = load_constants()
    cfg = {**MPPI, **(overrides or {})}
    cost = CostConfig()
    dt = consts["dt"]
    goals = {}
    for goal, (t1, t2) in GOALS.items():
        K, P = lqr(consts, (t1, t2), cost.lqr_q_theta, cost.lqr_r, dt)
        vals = {
            "M": consts["cartMass"], "m1": consts["poleMass1"], "m2": consts["poleMass2"],
            "l1": consts["poleLength1"], "l2": consts["poleLength2"], "g": consts["gravity"],
            "b": consts["cartFriction"], "c1": consts["jointDamping1"], "c2": consts["jointDamping2"],
            "dt": dt, "fmax": consts["forceLimit"], "t1": t1, "t2": t2, "e_target": potential(consts, t1, t2),
            **{k: getattr(cost, k) for k in (
                "w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft", "w_barrier", "x_dead",
                "w_dead", "w_energy", "w_u", "w_du", "w_terminal_lqr", "w_terminal_energy")},
            "gate_in": cfg["gate_in"], "gate_out": cfg["gate_out"], "anchor_on": 1.0,
            "knot": cfg["knot"], "sub": cfg["rollout_sub"], "early_exit": 1.0 if cfg["early_exit"] else 0.0,
        }
        goals[goal] = {"p": [float(vals[k]) for k in FIELDS], "K": K.tolist(), "P": P.reshape(-1).tolist()}
    return {
        "plant": "double",
        "fields": list(FIELDS),
        "mppi": cfg,
        "profiles": {"full": cfg},
        "force_limit": consts["forceLimit"],
        "goals": goals,
        "source": "shared/constants.json",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--mppi", default="", help="JSON overrides for the mppi config")
    args = ap.parse_args()
    spec = export(json.loads(args.mppi) if args.mppi else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
