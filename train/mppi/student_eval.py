#!/usr/bin/env python3
"""Run the teacher gates (G1 hold/shove vs LQR, G2 swing-up) on a distilled student.

    python -m mppi.student_eval --ckpt ../policies/mppi/student-uuu.pt --n 50
"""

from __future__ import annotations

import argparse
import json

import torch

from mppi.costs import CostConfig, TargetCost
from mppi.dynamics_fast import Plant
from mppi.student import StudentController, load
from mppi.teacher_eval import LQRController, gate_g1, gate_g2


def evaluate(plant: Plant, ctrl, goal: str, n: int = 50, seed: int = 0, steps: int = 600,
             swing_steps: int = 1500, hold_steps: int = 1000, with_g1: bool = True) -> dict:
    flat: dict = {}
    if with_g1:
        ref = LQRController(TargetCost(plant, CostConfig(goal=goal)), plant.force_limit)
        g1 = gate_g1(plant, ctrl, goal, n, steps, seed, ref)
        for k, v in g1.items():
            if isinstance(v, dict):
                flat[f"{k}_survival"] = v["survival"]
                flat[f"{k}_lqr"] = v["lqr_survival"]
        flat["g1_pass"] = g1["pass"]
    sw = gate_g2(plant, ctrl, goal, n, swing_steps, hold_steps, seed)["swing"]
    flat.update({
        "swing_enter": sw["enter_rate"],
        "swing_success": sw["success"],
        "swing_oob": sw["oob"],
        "g2_pass": sw["pass"],
    })
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--goal", default="")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    model, meta = load(args.ckpt)
    goal = args.goal or meta.get("goal", "UUU")
    plant = Plant.from_constants(walls=bool(meta.get("teacher", {}).get("walls", False)))
    report = evaluate(plant, StudentController(model), goal, n=args.n, seed=args.seed)
    print(json.dumps({"goal": goal, **report}, indent=1))
    return 0 if report["g1_pass"] and report["g2_pass"] else 1


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
