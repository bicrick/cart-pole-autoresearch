#!/usr/bin/env python3
"""G2 swing-up gate across cheaper MPPI settings, with their compute cost.

Cost is rollout plant-steps per simulated second (what the browser must
sustain): samples * horizon_steps / rollout_sub / (knot * dt). The 4096-sample
teacher is ~22 M/s.

    python -m mppi.speed_sweep --n 20 --out ../policies/mppi/speed-sweep.jsonl
    python -m mppi.speed_sweep --only '{"n_samples": 1024, "rollout_sub": 2}'
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mppi.config import build_teacher, read_config
from mppi.teacher_eval import gate_g2

HOLD = {"hold_samples": 128, "early_exit": True}
# rollout_sub=2 was tried and fails (0-5% swing-up): 60 Hz semi-implicit Euler
# drifts off the 120 Hz plant during fast swings, so only full-rate rollouts here.
GRID = [
    {"n_samples": 2048, **HOLD},
    {"n_samples": 1024, **HOLD},
    {"n_samples": 2048, "knot": 8, "n_knots": 22, **HOLD},
    {"n_samples": 1024, "knot": 8, "n_knots": 22, **HOLD},
    {"n_samples": 4096, "knot": 8, "n_knots": 22, **HOLD},
]


def steps_per_sim_second(m: dict, dt: float) -> float:
    return m["n_samples"] * m["n_knots"] * m["knot"] / m.get("rollout_sub", 1) / (m["knot"] * dt)


def run_one(config: str, goal: str, over: dict, n: int, seed: int) -> dict:
    raw = read_config(config, goal)
    raw["mppi"].update(over)
    plant, ctrl, meta = build_teacher(raw)
    m = meta["mppi"]
    t0 = time.time()
    sw = gate_g2(plant, ctrl, goal, n, 1500, 1000, seed)["swing"]
    return {
        "goal": goal,
        "mppi": over,
        "Msteps_per_s": round(steps_per_sim_second(m, plant.dt) / 1e6, 2),
        "success": sw["success"],
        "enter": sw["enter_rate"],
        "late": sw["late_enter"],
        "fell": sw["fell_after_enter"],
        "oob": sw["oob"],
        "median_enter_s": sw["median_enter_s"],
        "wall_s": round(time.time() - t0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mppi/configs/uuu.json")
    ap.add_argument("--goal", default="UUU")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", default="", help="one JSON override instead of the grid")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    grid = [json.loads(args.only)] if args.only else GRID
    for over in grid:
        row = run_one(args.config, args.goal, over, args.n, args.seed)
        print(json.dumps(row), flush=True)
        if args.out:
            with Path(args.out).open("a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
