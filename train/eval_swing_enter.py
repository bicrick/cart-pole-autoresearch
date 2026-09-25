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
    p.add_argument("--track-limit", type=float, default=2.4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--arrival-omega",
        type=float,
        default=0.0,
        help="If >0, spawn near the goal with this shared link rate instead of from the bottom.",
    )
    return p.parse_args()


def _errs(state: torch.Tensor, goal: torch.Tensor) -> tuple[float, float]:
    from goals_triple import GOAL_ANGLES

    th = state[[2, 4, 6]]
    # env._goal is the equilibrium id, not the three target angles.
    gid = int(goal.reshape(-1)[0].item()) if torch.is_tensor(goal) else int(goal)
    tgt = GOAL_ANGLES[gid].to(device=state.device, dtype=state.dtype)
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
        track_limit=args.track_limit,
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
    lqr_hit = []
    omega_at_lqr_ang = []
    for i in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + i)
        if args.arrival_omega > 0:
            s = torch.zeros(8, device=env.device)
            s[2] = 0.05
            s[4] = 0.05
            s[6] = 0.05
            s[3] = float(args.arrival_omega)
            s[5] = float(args.arrival_omega)
            s[7] = float(args.arrival_omega)
            env._state = s
            env._prev_state = s.clone()
            obs = env._obs_np()
        hit = False
        in_lqr = False
        min_ang = float("inf")
        rate_at = float("inf")
        om_when_close = float("nan")
        bumper = args.track_limit - 0.05
        for _ in range(args.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(act)
            ang, rate = _errs(env._state, env._goal)
            x = abs(float(env._state[0].item()))
            if ang < min_ang:
                min_ang = ang
                rate_at = rate
            if ang <= 0.05 and (om_when_close != om_when_close or rate < om_when_close):
                om_when_close = rate
            if ang <= 0.05 and rate <= 0.30 and x < bumper:
                in_lqr = True
            if ang <= args.angle_tol and rate <= args.rate_tol:
                hit = True
            if term or trunc:
                break
        entered.append(1.0 if hit else 0.0)
        best_ang.append(min_ang)
        best_rate_at_best_ang.append(rate_at)
        lqr_hit.append(1.0 if in_lqr else 0.0)
        omega_at_lqr_ang.append(om_when_close)
    erate = float(np.mean(entered))
    ang = np.asarray(best_ang, dtype=np.float64)
    close_om = np.asarray(omega_at_lqr_ang, dtype=np.float64)
    close_txt = "never" if np.all(np.isnan(close_om)) else f"{float(np.nanmedian(close_om)):.3f}"
    print(
        f"swing enter n={args.n_episodes} goal={args.goal} "
        f"enter_rate={erate:.3f} "
        f"lqr_region_rate={float(np.mean(lqr_hit)):.3f} "
        f"median_best_|θ|={float(np.median(ang)):.3f} "
        f"median_best_|ω|={float(np.median(best_rate_at_best_ang)):.3f} "
        f"median_|ω|_when_|θ|<=0.05={close_txt} "
        f"frac_|θ|<=0.3={float(np.mean(ang <= 0.3)):.3f} "
        f"frac_|θ|<=1.0={float(np.mean(ang <= 1.0)):.3f} "
        f"box=|θ|<={args.angle_tol} |ω|<={args.rate_tol} "
        f"lqr=|θ|<=0.05 |ω|<=0.30",
        flush=True,
    )
    print("W3 ENTER", "PASS" if erate >= 0.30 else "FAIL", flush=True)
    return 0 if erate >= 0.30 else 1


if __name__ == "__main__":
    raise SystemExit(main())
