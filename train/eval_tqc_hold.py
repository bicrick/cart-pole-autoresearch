#!/usr/bin/env python3
"""Eval a TQC UUU hold zip on the quiet-basin contract. Prints W1 gate meters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.triple_gym import TriplePendulumUUUEnv  # noqa: E402
from goals_triple import at_goal, mean_cos_align  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Eval TQC UUU hold checkpoint")
    p.add_argument("--checkpoint", default="policies/tqc-m2-uuu-hold-mac.zip")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--init-noise", type=float, default=0.02)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--goal", "--target-ep", dest="goal", default="UUU")
    p.add_argument("--track-walls", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from sb3_contrib import TQC

    path = Path(args.checkpoint)
    if not path.is_file() and not path.with_suffix(".zip").is_file():
        print(f"MISSING {path}", flush=True)
        return 2
    load = str(path.with_suffix("") if path.suffix == ".zip" else path)
    model = TQC.load(load, device=args.device)
    env = TriplePendulumUUUEnv(
        force_limit=args.force_limit,
        max_steps=args.max_steps,
        track_walls=args.track_walls,
        init_mode="near_target",
        init_noise=args.init_noise,
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        seed=args.seed,
        goal=args.goal,
    )
    surv = []
    goal = []
    align = []
    lens = []
    for i in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + i)
        last = {}
        for _ in range(args.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, last = env.step(act)
            if term or trunc:
                break
        surv.append(1.0 if last.get("is_success") else 0.0)
        goal.append(1.0 if last.get("at_goal_final", last.get("at_goal")) else 0.0)
        lens.append(int(last.get("ep_len", 0)))
        try:
            align.append(
                float(mean_cos_align(env._state.unsqueeze(0), env._goal.unsqueeze(0)).item())
            )
        except Exception:
            align.append(float("nan"))
        _ = at_goal
    srate = float(np.mean(surv))
    grate = float(np.mean(goal))
    arate = float(np.nanmean(align))
    print(
        f"W1 eval n={args.n_episodes} noise={args.init_noise} walls={int(args.track_walls)} "
        f"survival={srate:.3f} at_goal={grate:.3f} align={arate:.3f} "
        f"mean_len={float(np.mean(lens)):.1f}",
        flush=True,
    )
    gate = srate >= 0.80 and grate >= 0.80 and arate >= 0.90
    print("W1 GATE", "PASS" if gate else "FAIL", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
