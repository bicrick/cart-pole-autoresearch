#!/usr/bin/env python3
"""Enter-rate of a swing TQC zip into the catch box this plant owns.

Box: every |wrapped θ error| <= 0.03 and every |ω| <= 0.01, at any step.
Spawn is fawraw M4 bottom (hanging), angle-fall off. Not a hold-survival gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.triple_gym import TriplePendulumUUUEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Swing enter-rate into the quiet catch box")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--goal", default="UUU")
    p.add_argument("--n-episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--init-noise", type=float, default=0.05)
    p.add_argument("--angle-tol", type=float, default=0.03)
    p.add_argument("--rate-tol", type=float, default=0.01)
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _errs(state: torch.Tensor, goal: torch.Tensor) -> tuple[float, float]:
    th = state[[2, 4, 6]]
    tgt = goal
    err = torch.atan2(torch.sin(th - tgt), torch.cos(th - tgt)).abs()
    om = state[[3, 5, 7]].abs()
    return float(err.max().item()), float(om.max().item())


def main() -> int:
    args = parse_args()
    from sb3_contrib import TQC

    path = Path(args.checkpoint)
    if not path.is_file():
        print(f"MISSING {path}", flush=True)
        return 2
    model = TQC.load(str(path), device=args.device)
    env = TriplePendulumUUUEnv(
        force_limit=args.force_limit,
        max_steps=args.max_steps,
        track_walls=True,
        init_mode="bottom",
        init_noise=args.init_noise,
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        angle_fall=False,
        seed=args.seed,
        goal=args.goal,
    )
    entered = []
    best_ang = []
    best_rate_at_best_ang = []
    for i in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + i)
        hit = False
        min_ang = float("inf")
        rate_at = float("inf")
        for _ in range(args.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            ang, rate = _errs(env._state, env._goal)
            if ang < min_ang:
                min_ang = ang
                rate_at = rate
            if ang <= args.angle_tol and rate <= args.rate_tol:
                hit = True
            if term or trunc:
                break
        entered.append(1.0 if hit else 0.0)
        best_ang.append(min_ang)
        best_rate_at_best_ang.append(rate_at)
    erate = float(np.mean(entered))
    print(
        f"swing enter n={args.n_episodes} goal={args.goal} "
        f"enter_rate={erate:.3f} "
        f"median_best_|θ|={float(np.median(best_ang)):.3f} "
        f"median_best_|ω|={float(np.median(best_rate_at_best_ang)):.3f} "
        f"box=|θ|<={args.angle_tol} |ω|<={args.rate_tol}",
        flush=True,
    )
    print("W3 ENTER", "PASS" if erate >= 0.30 else "FAIL", flush=True)
    return 0 if erate >= 0.30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
