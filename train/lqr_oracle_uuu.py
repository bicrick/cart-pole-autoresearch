#!/usr/bin/env python3
"""LQR UUU oracle gate — plant + env-contract sanity before BC / TQC FT.

Rolls out ``design_lqr_uuu`` from the quiet basin under the M2 env contract
(fall-kill |θ|>0.6, rates ±0.01, walls ON). Requires ≥95% survival (and reports
final at_goal) over N≥50 episodes at init_noise ∈ {0.01, 0.02}.

On PASS, dumps demo trajectories to policies/lqr-uuu-demos.npz for later BC.

CPU-only. Does NOT start GCP / TQC training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from envs.triple_gym import (  # noqa: E402
    FALL_THRESH_UP,
    QUIET_RATE,
    SURVIVAL_FRAC,
    TriplePendulumUUUEnv,
)
from goals_triple import at_goal  # noqa: E402
from lqr_uuu import design_lqr_uuu, state_from_env  # noqa: E402
from physics_triple import load_constants  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="LQR UUU oracle gate (quiet basin)")
    p.add_argument("--noises", type=str, default="0.01,0.02")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--q-theta", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.01)
    p.add_argument("--pass-threshold", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--demo-out", type=str, default="policies/lqr-uuu-demos.npz")
    p.add_argument("--demo-episodes", type=int, default=80, help="Trajectories to dump on PASS")
    p.add_argument("--demo-noise", type=float, default=0.02, help="IC noise for demo dump")
    p.add_argument("--no-dump", action="store_true")
    p.add_argument(
        "--dump-even-if-fail",
        action="store_true",
        help="Still write surviving demos when the oracle bar is missed (W2 widen).",
    )
    p.add_argument("--track-walls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--goal", "--target-ep", dest="goal", default="UUU")
    return p.parse_args()


def rollout_episode(env: TriplePendulumUUUEnv, ctrl, *, record: bool = False):
    obs, info = env.reset()
    states = []
    actions = []
    for _ in range(env.max_steps):
        z = state_from_env(env)
        a = ctrl.action_normed(z)  # [-1,1]
        if record:
            states.append(z.copy())
            actions.append(float(a[0]) * env.force_limit)  # Newtons
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    survived = bool(info.get("survival_success", False))
    # Fallback if info missing: ep_len + not fell/oob
    if "survival_success" not in info:
        ep_len = int(info.get("ep_len", env._step_count))
        survived = (not info.get("fell", False)) and (not info.get("oob", False)) and (
            ep_len >= int(SURVIVAL_FRAC * env.max_steps)
        )
    goal_ok = bool(info.get("at_goal_final", info.get("at_goal", False)))
    # Prefer true final at_goal from plant
    try:
        goal_ok = bool(
            at_goal(
                env._state.unsqueeze(0),
                env._goal.unsqueeze(0),
            ).item()
        )
    except Exception:
        pass
    ep = {
        "survived": survived,
        "at_goal": goal_ok,
        "ep_len": int(info.get("ep_len", env._step_count)),
        "fell": bool(info.get("fell", False)),
        "oob": bool(info.get("oob", False)),
    }
    if record:
        ep["states"] = np.asarray(states, dtype=np.float32)
        ep["actions"] = np.asarray(actions, dtype=np.float32)
    return ep


def eval_noise(ctrl, noise: float, args) -> dict:
    env = TriplePendulumUUUEnv(
        force_limit=args.force_limit,
        max_steps=args.max_steps,
        track_walls=bool(args.track_walls),
        init_mode="near_target",
        init_noise=float(noise),
        hang_frac=0.0,
        wide_frac=0.0,
        progress_w=0.0,
        angle_fall=True,
        fall_thresh_up=FALL_THRESH_UP,
        seed=args.seed,
        goal=args.goal,
    )
    n_ok = 0
    n_goal = 0
    lens = []
    for i in range(args.n_episodes):
        # Seed RNGs before rollout_episode's reset (quiet-basin sample).
        torch.manual_seed(args.seed + 17 * i + int(noise * 1e4))
        np.random.seed(args.seed + 17 * i + int(noise * 1e4))
        ep = rollout_episode(env, ctrl, record=False)
        lens.append(ep["ep_len"])
        if ep["survived"]:
            n_ok += 1
        if ep["at_goal"]:
            n_goal += 1
    surv_rate = n_ok / float(args.n_episodes)
    goal_rate = n_goal / float(args.n_episodes)
    return {
        "init_noise": float(noise),
        "n_episodes": int(args.n_episodes),
        "survival_rate": surv_rate,
        "at_goal_rate": goal_rate,
        "mean_ep_len": float(np.mean(lens)),
        "pass": surv_rate >= float(args.pass_threshold),
    }


def dump_demos(ctrl, args) -> Path:
    env = TriplePendulumUUUEnv(
        force_limit=args.force_limit,
        max_steps=args.max_steps,
        track_walls=bool(args.track_walls),
        init_mode="near_target",
        init_noise=float(args.demo_noise),
        progress_w=0.0,
        angle_fall=True,
        seed=args.seed + 99,
        goal=args.goal,
    )
    all_states = []
    all_actions = []
    ep_lens = []
    n_surv = 0
    for i in range(args.demo_episodes):
        torch.manual_seed(args.seed + 9000 + i)
        np.random.seed(args.seed + 9000 + i)
        ep = rollout_episode(env, ctrl, record=True)
        if not ep["survived"]:
            continue
        n_surv += 1
        all_states.append(ep["states"])
        all_actions.append(ep["actions"])
        ep_lens.append(len(ep["states"]))
    out = Path(args.demo_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Ragged → store concatenated + offsets
    if not all_states:
        raise RuntimeError("no surviving demos to dump")
    offsets = np.cumsum([0] + ep_lens[:-1]).astype(np.int64)
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    meta = {
        "controller": "lqr_uuu",
        "q_theta": args.q_theta,
        "r": args.r,
        "force_limit": args.force_limit,
        "init_noise": args.demo_noise,
        "quiet_rate": QUIET_RATE,
        "fall_thresh_up": FALL_THRESH_UP,
        "max_steps": args.max_steps,
        "n_episodes_requested": args.demo_episodes,
        "n_episodes_survived": n_surv,
        "n_transitions": int(states.shape[0]),
        "state_dim": 8,
        "action_unit": "newtons",
        "env_contract": "fawraw M2 quiet basin + fall-kill + walls ON",
    }
    np.savez_compressed(
        out,
        states=states,
        actions=actions,
        ep_offsets=offsets,
        ep_lens=np.asarray(ep_lens, dtype=np.int64),
        meta_json=np.asarray(json.dumps(meta)),
    )
    print(f"DEMO DUMP → {out} episodes={n_surv} transitions={states.shape[0]}", flush=True)
    return out


def main() -> int:
    args = parse_args()
    noises = [float(x) for x in args.noises.split(",") if x.strip()]
    print(f"=== LQR {args.goal} oracle gate ===", flush=True)
    print(
        f"env contract: quiet_rate=±{QUIET_RATE}, fall_thresh={FALL_THRESH_UP}, "
        f"survival_frac={SURVIVAL_FRAC}, walls={args.track_walls}, progress_w=0",
        flush=True,
    )
    t0 = time.time()
    consts = dict(load_constants())
    consts["forceLimit"] = float(args.force_limit)
    consts["trackWalls"] = bool(args.track_walls)
    ctrl = design_lqr_uuu(
        force_limit=args.force_limit,
        q_theta=args.q_theta,
        r=args.r,
        constants=consts,
        goal=args.goal,
    )
    print(
        f"LQR designed in {time.time()-t0:.1f}s  K={np.array2string(ctrl.K, precision=3)}",
        flush=True,
    )

    results = []
    all_pass = True
    for n in noises:
        print(f"\n--- init_noise={n}  N={args.n_episodes} ---", flush=True)
        t1 = time.time()
        res = eval_noise(ctrl, n, args)
        res["elapsed_s"] = round(time.time() - t1, 2)
        results.append(res)
        status = "PASS" if res["pass"] else "FAIL"
        print(
            f"  {status}  survival={res['survival_rate']:.3f}  "
            f"at_goal={res['at_goal_rate']:.3f}  "
            f"mean_ep_len={res['mean_ep_len']:.1f}  "
            f"(need survival>={args.pass_threshold})",
            flush=True,
        )
        if not res["pass"]:
            all_pass = False

    print("\n=== SUMMARY ===", flush=True)
    for res in results:
        tag = "PASS" if res["pass"] else "FAIL"
        print(
            f"  noise={res['init_noise']}: {tag} "
            f"survival={res['survival_rate']:.3f} at_goal={res['at_goal_rate']:.3f}",
            flush=True,
        )

    demo_path = None
    if not args.no_dump and (all_pass or args.dump_even_if_fail):
        if not all_pass:
            print("oracle bar missed — dumping surviving rollouts only", flush=True)
        demo_path = str(dump_demos(ctrl, args))
    elif not all_pass:
        print("FAIL — no demo dump. Fix plant/LQR before BC.", flush=True)
    else:
        print("PASS — demo dump skipped (--no-dump)", flush=True)

    summary = {
        "oracle": "PASS" if all_pass else "FAIL",
        "pass_threshold": args.pass_threshold,
        "results": results,
        "demo": demo_path,
        "next_step": "BC from demos → short TQC FT" if all_pass else "debug plant/LQR",
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(("ORACLE PASS" if all_pass else "ORACLE FAIL"), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
