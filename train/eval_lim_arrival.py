#!/usr/bin/env python3
"""Score a Lim wide specialist on arrival, not quiet-basin survival.

Starts: a hanging neighborhood, then each of the other seven equilibria.
The network's goal stays the checkpoint target (default UUU). Angle-fall is
off, because a foreign equilibrium is not a fall. Walls off, matching the
void plant this run trains on.
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
from goals_triple import GOAL_ANGLES, GOAL_IDS, parse_goal  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Lim wide-start arrival eval")
    p.add_argument("--checkpoint", default="policies/tqc-lim-uuu-wide.zip")
    p.add_argument("--goal", default="UUU")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _plant(args, seed: int) -> TriplePendulumUUUEnv:
    return TriplePendulumUUUEnv(
        force_limit=args.force_limit,
        max_steps=args.max_steps,
        track_walls=False,
        init_mode="bottom",
        init_noise=0.05,
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        ry_scale=1.0,
        cart_barrier_coef=0.0,
        alpha_th=0.0,
        w_up=1.0,
        w_down=1.0,
        sparse_bonus=0.0,
        angle_fall=False,
        seed=seed,
        goal=args.goal,
        device="cpu",
    )


def _set_equilibrium(env: TriplePendulumUUUEnv, name: str) -> None:
    gid = parse_goal(name)
    ang = GOAL_ANGLES[gid]
    env._state = torch.tensor(
        [0.0, 0.0, float(ang[0]), 0.0, float(ang[1]), 0.0, float(ang[2]), 0.0],
        device=env.device,
        dtype=torch.float32,
    )
    env._prev_state = env._state.clone()
    env._step_count = 0


def _roll(env, model, n: int, seed: int, pose: str | None) -> tuple[float, float, float]:
    surv, goal, lens = [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        if pose is not None:
            _set_equilibrium(env, pose)
            obs = env._obs_np()
        last = {}
        for _ in range(env.max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, last = env.step(act)
            if term or trunc:
                break
        surv.append(1.0 if last.get("survival_success") else 0.0)
        goal.append(1.0 if last.get("at_goal_final", last.get("at_goal")) else 0.0)
        lens.append(int(last.get("ep_len", 0)))
    return float(np.mean(surv)), float(np.mean(goal)), float(np.mean(lens))


def main() -> int:
    args = parse_args()
    from sb3_contrib import TQC

    path = Path(args.checkpoint)
    if not path.is_file() and not path.with_suffix(".zip").is_file():
        print(f"MISSING {path}", flush=True)
        return 2
    load = str(path.with_suffix("") if path.suffix == ".zip" else path)
    model = TQC.load(load, device=args.device)
    target = parse_goal(args.goal)
    sources = ["hang"] + [name for i, name in enumerate(GOAL_IDS) if i != target]
    rows = []
    for src in sources:
        env = _plant(args, args.seed)
        pose = None if src == "hang" else src
        srate, grate, mlen = _roll(env, model, args.n_episodes, args.seed, pose)
        ok = srate >= 0.80 and grate >= 0.80
        rows.append(ok)
        print(
            f"arrival {src:>4} -> {args.goal} n={args.n_episodes} "
            f"survival={srate:.3f} at_goal={grate:.3f} mean_len={mlen:.1f} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        env.close()
    passed = all(rows)
    print("ARRIVAL GATE", "PASS" if passed else "FAIL", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
