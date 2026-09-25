#!/usr/bin/env python3
"""Smoke-eval S1→H1 two-policy handoff on cart-triple UUU (walls plant).

Loads PPO swing + catcher checkpoints (or LQR catcher), rolls out from hang
starts, reports enter-rate / hold success. CPU-friendly; does not start GPU train.

Example:
  python3 scripts/handoff_eval.py \\
    --swing policies/checkpoint-triple-a.pt \\
    --catcher policies/checkpoint-triple-b.pt \\
    --episodes 32 --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))

from handoff import (  # noqa: E402
    TwoPolicyHandoff,
    angle_err,
    in_enter_basin,
    wrap_lqr_catcher,
    wrap_ppo_policy,
    zero_swing,
)
from lqr_uuu import design_lqr_uuu  # noqa: E402
from physics_triple import load_constants, step  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="S1→H1 handoff smoke eval (UUU)")
    p.add_argument("--swing", type=str, default="", help="PPO swing .pt (empty=zero)")
    p.add_argument("--catcher", type=str, default="", help="PPO catcher .pt (empty=lqr)")
    p.add_argument("--catcher-backend", choices=("ppo", "lqr"), default="ppo")
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--track-limit", type=float, default=2.4)
    p.add_argument("--track-walls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--capture-tol", type=float, default=0.03)
    p.add_argument("--capture-vel", type=float, default=0.10)
    p.add_argument("--exit-tol", type=float, default=0.25)
    p.add_argument("--enter-dwell", type=int, default=5)
    p.add_argument("--exit-dwell", type=int, default=5)
    p.add_argument("--success-frac", type=float, default=0.5,
                   help="Fraction of post-enter steps with max|φ|<exit_tol for success")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def hang_state(rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(-1.0, 1.0)
    xd = rng.uniform(-0.8, 0.8)
    th = np.pi + rng.uniform(-0.35, 0.35, size=3)
    thd = rng.uniform(-1.0, 1.0, size=3)
    return np.array([x, xd, th[0], thd[0], th[1], thd[1], th[2], thd[2]], dtype=np.float64)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    constants = dict(load_constants())
    constants["forceLimit"] = float(args.force_limit)
    constants["trackLimit"] = float(args.track_limit)
    constants["trackWalls"] = bool(args.track_walls)

    if args.swing and Path(args.swing).is_file():
        swing = wrap_ppo_policy(args.swing, force_limit=args.force_limit, device=args.device)
        swing_name = args.swing
    else:
        swing = zero_swing
        swing_name = "zero"

    if args.catcher_backend == "lqr" or not args.catcher or not Path(args.catcher).is_file():
        lqr = design_lqr_uuu(force_limit=args.force_limit)
        catcher = wrap_lqr_catcher(lqr)
        catcher_name = "lqr"
    else:
        catcher = wrap_ppo_policy(args.catcher, force_limit=args.force_limit, device=args.device)
        catcher_name = args.catcher

    handoff = TwoPolicyHandoff(
        swing=swing,
        catcher=catcher,
        capture_tol_rad=args.capture_tol,
        capture_vel_rad_s=args.capture_vel,
        exit_tol_rad=args.exit_tol,
        enter_dwell=args.enter_dwell,
        exit_dwell=args.exit_dwell,
        force_limit=args.force_limit,
    )

    entered = 0
    success = 0
    oob = 0
    hold_fracs = []

    for ep in range(args.episodes):
        handoff.reset()
        z = torch.as_tensor(hang_state(rng), dtype=torch.float32).reshape(1, 8)
        saw_enter = False
        post_enter = 0
        good_hold = 0
        ep_oob = False
        for _ in range(args.max_steps):
            z_np = z.squeeze(0).cpu().numpy()
            if not saw_enter and in_enter_basin(
                z_np,
                capture_tol_rad=args.capture_tol,
                capture_vel_rad_s=args.capture_vel,
            ):
                # count basin visit even before dwell completes
                pass
            act = handoff.predict(z_np)
            force_n = float(act[0]) * args.force_limit
            force = torch.tensor([force_n], dtype=torch.float32)
            z = step(z, force, constants=constants)
            if abs(float(z[0, 0].item())) > args.track_limit + 1e-3 and not args.track_walls:
                ep_oob = True
                break
            if handoff.holding:
                saw_enter = True
                post_enter += 1
                if np.max(np.abs(angle_err(z.squeeze(0).cpu().numpy()))) < args.exit_tol:
                    good_hold += 1
        if ep_oob:
            oob += 1
            continue
        if saw_enter:
            entered += 1
            frac = good_hold / max(post_enter, 1)
            hold_fracs.append(frac)
            if frac >= args.success_frac:
                success += 1

    n = args.episodes
    summary = {
        "swing": swing_name,
        "catcher": catcher_name,
        "episodes": n,
        "entered": entered,
        "enter_rate": entered / n,
        "success": success,
        "success_rate": success / n,
        "oob": oob,
        "mean_hold_frac_given_enter": float(np.mean(hold_fracs)) if hold_fracs else 0.0,
        "capture_tol": args.capture_tol,
        "track_walls": args.track_walls,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
    # Non-zero exit only if both ckpts missing (can't smoke). Soft OK otherwise.
    if swing_name == "zero" and catcher_name == "lqr":
        print("NOTE: no PPO ckpts — LQR/zero smoke only", file=sys.stderr)


if __name__ == "__main__":
    main()
