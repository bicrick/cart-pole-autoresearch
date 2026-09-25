#!/usr/bin/env python3
"""Hang rollouts of the classical energy pump.

Reports the quiet catch box and the measured LQR region
(|θ| <= 0.05 and |ω| <= 0.30, cart off the bumper).
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

from energy_pump import EnergyPump  # noqa: E402
from envs.triple_gym import TriplePendulumUUUEnv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Energy-pump arrival from a hang")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--init-noise", type=float, default=0.05)
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--track-limit", type=float, default=2.4)
    p.add_argument("--angle-tol", type=float, default=0.03)
    p.add_argument("--rate-tol", type=float, default=0.01)
    p.add_argument("--lqr-angle", type=float, default=0.05)
    p.add_argument("--lqr-omega", type=float, default=0.30)
    p.add_argument("--target", choices=("poles", "outer", "inner"), default="poles")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _errs(state: torch.Tensor) -> tuple[float, float]:
    th = state[[2, 4, 6]]
    err = torch.atan2(torch.sin(th), torch.cos(th)).abs()
    om = state[[3, 5, 7]].abs()
    return float(err.max().item()), float(om.max().item())


def main() -> int:
    args = parse_args()
    pump = EnergyPump(
        force_limit=args.force_limit,
        track_limit=args.track_limit,
        target=args.target,
    )
    env = TriplePendulumUUUEnv(
        force_limit=args.force_limit,
        max_steps=args.max_steps,
        track_limit=args.track_limit,
        track_walls=True,
        init_mode="bottom",
        init_noise=args.init_noise,
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        angle_fall=False,
        seed=args.seed,
        goal="UUU",
    )
    bumper = args.track_limit - 0.05
    entered = []
    lqr_hit = []
    best_ang = []
    best_rate = []
    rate_in_lqr_angle = []
    for i in range(args.n_episodes):
        env.reset(seed=args.seed + i)
        hit = False
        in_lqr = False
        min_ang = float("inf")
        rate_at = float("inf")
        om_when_close = float("inf")
        for _ in range(args.max_steps):
            act = pump.action_normed(env._state.detach().cpu().numpy())
            _, _, term, trunc, _ = env.step(act)
            ang, rate = _errs(env._state)
            x = abs(float(env._state[0].item()))
            if ang < min_ang:
                min_ang = ang
                rate_at = rate
            if ang <= args.lqr_angle:
                om_when_close = min(om_when_close, rate)
            off_bumper = x < bumper
            if ang <= args.angle_tol and rate <= args.rate_tol:
                hit = True
            if ang <= args.lqr_angle and rate <= args.lqr_omega and off_bumper:
                in_lqr = True
            if term or trunc:
                break
        entered.append(1.0 if hit else 0.0)
        lqr_hit.append(1.0 if in_lqr else 0.0)
        best_ang.append(min_ang)
        best_rate.append(rate_at)
        rate_in_lqr_angle.append(om_when_close)
    ang = np.asarray(best_ang, dtype=np.float64)
    close_om = np.asarray(rate_in_lqr_angle, dtype=np.float64)
    finite = np.isfinite(close_om)
    close_txt = f"{float(np.median(close_om[finite])):.3f}" if finite.any() else "never"
    print(
        f"energy pump target={args.target} n={args.n_episodes} "
        f"enter_rate={float(np.mean(entered)):.3f} "
        f"lqr_region_rate={float(np.mean(lqr_hit)):.3f} "
        f"median_best_|θ|={float(np.median(ang)):.3f} "
        f"median_|ω|_at_best={float(np.median(best_rate)):.3f} "
        f"median_|ω|_when_|θ|<={args.lqr_angle:.2f}={close_txt} "
        f"k_swing={pump.k_swing:.3f} E*={pump.e_star:.3f}",
        flush=True,
    )
    return 0 if float(np.mean(lqr_hit)) >= 0.3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
