#!/usr/bin/env python3
"""Swing-up gate for n-link MPPI over a grid of settings (the triple's G2 gate).

Episodes start at a hang (near all-up for the all-down goal), run ``--swing-s``
seconds to enter the quiet box (angle errors < 0.03 rad, rates < 0.01 rad/s),
then must stay inside the fall band for ``--hold-s``. One jsonl row per
(goal, samples, horizon, force) with success rates, timing and compute; the
``swing_*`` keys show up under swing/ in ``mppi.tb_watch``.

    python -m mppi.nlink.sweep --n 4 --goals UUUU --samples 8192,16384 --horizon 1.5,2 \
        --force 40,60 --episodes 10 --out ../policies/mppi/quad/sweep.jsonl
    python -m mppi.nlink.sweep --n 3 --goals UUU --samples 1024 --episodes 2 --backend cpu   # sanity
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mppi.nlink.costs import pack
from mppi.nlink.goals import goal_angles, goal_ids
from mppi.nlink.mppi import NLinkMPPI, NLinkMPPIConfig
from mppi.nlink.plant import NLinkPlant, step

FALL_UP, FALL_DOWN = 0.6, 1.5
QUIET_ANGLE, QUIET_RATE = 0.03, 0.01


def starts_for(goal: str, p: NLinkPlant, b: int, gen: torch.Generator) -> torch.Tensor:
    base = "U" * p.n if set(goal) == {"D"} else "D" * p.n
    s = torch.zeros(b, p.dim, dtype=torch.float64)
    s[:, 2::2] = torch.tensor(goal_angles(base), dtype=torch.float64)
    noise = (torch.rand(b, p.dim, generator=gen, dtype=torch.float64) * 2 - 1) * 0.05
    noise[:, 1::2] *= 0.2
    return s + noise


def swing_metrics(goal: str, states: torch.Tensor, p: NLinkPlant, hold_steps: int) -> dict:
    """states [T+1, B, dim] -> G2 metrics (mppi.episodes.swing_metrics for n links)."""
    tgt = torch.tensor(goal_angles(goal), dtype=states.dtype)
    err = (torch.remainder(states[..., 2::2] - tgt + math.pi, 2 * math.pi) - math.pi).abs()
    rate = states[..., 3::2].abs()
    quiet = (err <= QUIET_ANGLE).all(-1) & (rate <= QUIET_RATE).all(-1)
    band = torch.where(tgt.abs() < 1e-6, torch.tensor(FALL_UP), torch.tensor(FALL_DOWN)).to(states.dtype)
    t_total, b = quiet.shape
    entered = quiet.any(0)
    first = torch.where(entered, quiet.float().argmax(0), torch.full((b,), t_total))
    after = torch.arange(t_total).unsqueeze(1) >= first.unsqueeze(0)
    fell = entered & ((err > band).any(-1) & after).any(0)
    late = entered & ((t_total - 1 - first) < hold_steps)
    oob = (states[..., 0].abs() > p.track_limit).any(0)
    success = entered & ~fell & ~late & ~oob
    ok_first = first[entered].float()
    return {
        "swing_success": float(success.float().mean()),
        "swing_enter": float(entered.float().mean()),
        "swing_fell": float(fell.float().mean()),
        "swing_late": float(late.float().mean()),
        "swing_oob": float(oob.float().mean()),
        "swing_median_enter_s": float(ok_first.median() * p.dt) if ok_first.numel() else float("nan"),
        "min_final_err": float(err[-1].max(-1).values.min()),
    }


@torch.no_grad()
def run(goal: str, p: NLinkPlant, ctrl: NLinkMPPI, episodes: int, steps: int, seed: int, desc: str):
    gen = torch.Generator().manual_seed(seed)
    s = starts_for(goal, p, episodes, gen).to(ctrl.device)
    ctrl.reset(episodes)
    out = [s.cpu()]
    for _ in tqdm(range(steps), desc=desc, unit="step", leave=False, dynamic_ncols=True, mininterval=1.0):
        u = ctrl.act(s).double()
        s = step(p, s, u)
        out.append(s.cpu())
    return torch.stack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--goals", default="", help="comma list, or 'all' (default: all-up)")
    ap.add_argument("--samples", default="16384")
    ap.add_argument("--horizon", default="1.5", help="seconds, comma list")
    ap.add_argument("--force", default="40", help="force limits [N], comma list")
    ap.add_argument("--iters", default="1", help="MPPI passes per replan, comma list")
    ap.add_argument("--sigma", default="auto",
                    help="noise std [N], comma list; 'auto' scales the triple's 8 N (40 N limit) with the force limit")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--swing-s", type=float, default=25.0)
    ap.add_argument("--hold-s", type=float, default=8.0)
    ap.add_argument("--knot", type=int, default=4)
    ap.add_argument("--backend", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    goals = goal_ids(args.n) if args.goals == "all" else (args.goals.split(",") if args.goals else ["U" * args.n])
    grid = list(itertools.product(goals, [int(x) for x in args.samples.split(",")],
                                  [float(x) for x in args.horizon.split(",")],
                                  [float(x) for x in args.force.split(",")],
                                  [int(x) for x in args.iters.split(",")],
                                  args.sigma.split(",")))
    device = "cuda" if args.backend == "cuda" else "cpu"
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    for i, (goal, samples, horizon, force, iters, sig) in enumerate(
            tqdm(grid, desc="configs", unit="cfg", dynamic_ncols=True)):
        p = NLinkPlant.load(args.n, force_limit=force)
        n_knots = max(2, round(horizon / (args.knot * p.dt)))
        # The triple's 8 / 25 N were tuned at 40 N; "auto" scales them with authority.
        sigma = 8.0 * force / 40.0 if sig == "auto" else float(sig)
        cfg = NLinkMPPIConfig(n_samples=samples, n_knots=n_knots, knot=args.knot, sigma=sigma,
                              sigma_wide=sigma * 25.0 / 8.0, n_iters=iters, backend=args.backend,
                              seed=args.seed + i)
        pk = pack(p, goal, knot=args.knot, gate_in=cfg.gate_in, gate_out=cfg.gate_out)
        ctrl = NLinkMPPI(pk, p.n, force, cfg, device)
        swing, hold = round(args.swing_s / p.dt), round(args.hold_s / p.dt)
        t0 = time.time()
        states = run(goal, p, ctrl, args.episodes, swing + hold, args.seed, f"{goal} K={samples} H={horizon}s F={force:g} it={iters} sig={sigma:g}")
        wall = time.time() - t0
        m = swing_metrics(goal, states, p, hold)
        rollout_steps = args.episodes * samples * iters * n_knots * args.knot * (swing + hold) / args.knot
        row = {"iter": i, "n": args.n, "goal": goal, "samples": samples, "horizon_s": horizon, "force": force,
               "mppi_iters": iters, "sigma": sigma,
               "episodes": args.episodes, "unstable_rate": pk.unstable_rate, **m, "wall_s": round(wall, 1),
               # upper bounds: replans in hold mode use hold_samples, not `samples`
               "Gsteps_max": round(rollout_steps / 1e9, 2), "Msteps_per_s_max": round(rollout_steps / wall / 1e6, 1)}
        tqdm.write(f"{goal} K={samples} H={horizon}s F={force:g} it={iters} sig={sigma:g}: swing-up {m['swing_success']:.0%} "
                   f"(enter {m['swing_enter']:.0%}, fell {m['swing_fell']:.0%}, oob {m['swing_oob']:.0%}, "
                   f"median {m['swing_median_enter_s']:.1f}s) | {wall:.0f}s, <= {row['Msteps_per_s_max']} M rollout steps/s")
        if out:
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
