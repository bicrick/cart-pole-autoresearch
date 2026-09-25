#!/usr/bin/env python3
"""Directed A->B switching matrix over the 8 equilibria using per-goal students.

Each pair starts quiet at A (noise 0.02, rates 0.01) and runs the student for
B. Success: enter B's quiet box and stay inside B's fall band for
``hold_steps`` with the cart on the track. Same-goal pairs are plain holds.

    python -m mppi.switch_eval --students ../policies/mppi --n 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from goals_triple import GOAL_IDS
from mppi.dynamics_fast import Plant
from mppi.episodes import EpisodeSpec, near_states, run, swing_metrics
from mppi.student import StudentController, load


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=Path, default=Path("../policies/mppi"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500, help="time allowed to reach B")
    ap.add_argument("--hold-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    plant = Plant.from_constants(walls=False)
    ctrls = {}
    for g in GOAL_IDS:
        p = args.students / f"student-{g.lower()}.pt"
        if p.exists():
            ctrls[g] = StudentController(load(p)[0])
    missing = [g for g in GOAL_IDS if g not in ctrls]
    if missing:
        print(f"missing students: {missing}")

    matrix: dict = {}
    for a in GOAL_IDS:
        matrix[a] = {}
        for b, ctrl in ctrls.items():
            gen = torch.Generator().manual_seed(args.seed)
            s0 = near_states(a, args.n, 0.02, 0.01, gen)
            states, _ = run(plant, ctrl, s0, EpisodeSpec(goal=b, steps=args.steps + args.hold_steps))
            m = swing_metrics(b, states, plant.track_limit, args.hold_steps, plant.dt)
            matrix[a][b] = round(m["success"], 3)
        print(a, matrix[a], flush=True)

    directed = [matrix[a][b] for a in GOAL_IDS for b in ctrls if a != b]
    summary = {
        "pairs": len(directed),
        "mean_success": sum(directed) / max(1, len(directed)),
        "pairs_at_0.9": sum(v >= 0.9 for v in directed),
        "matrix": matrix,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "matrix"}))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
