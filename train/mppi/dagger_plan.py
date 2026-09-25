#!/usr/bin/env python3
"""DAgger on the MPPI teacher's plan update (see ``plannet``).

Each episode carries two plans: the executed one and the teacher's own. At
every knot boundary the teacher refines its own shifted plan from the visited
state (the label), and the student predicts ``PlanNet(obs, warm)`` from the
executed plan shifted to ``warm``. The episode executes the teacher's plan
with probability ``beta`` (per episode) and the student's otherwise.

Labels come from the teacher's own plan because one MPPI pass started from a
poor student plan jumps to whichever wide sample happens to win: repeat labels
of the same (state, warm) differ by ~22 N, versus ~2 N from its own plan.

Leaving the track respawns at a hang with a zero plan, like the web demo.
Random +-40 N shoves (0.1-0.3 s) and starts near every equilibrium widen
coverage for drags and goal switches.

    python -m mppi.dagger_plan --config mppi/configs/uuu.json --out ../policies/mppi/plan-uuu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from goals_triple import GOAL_IDS
from mppi.config import load_teacher
from mppi.dagger import shove_schedule
from mppi.dynamics_fast import step
from mppi.episodes import hang_states, near_states, wide_states
from mppi.plannet import PLAN_SCALE, PlanController, PlanNet, from_teacher, load, mirror, save, shift
from mppi.student_eval import evaluate
from physics_triple import observe

KINDS = ("hang", "near", "wide", "eq")


def sample_starts(goal: str, n: int, gen: torch.Generator, mix: dict) -> torch.Tensor:
    p = torch.tensor([mix.get(k, 0.0) for k in KINDS], dtype=torch.float)
    pick = torch.multinomial(p / p.sum(), n, replacement=True, generator=gen)
    out = torch.empty(n, 8)
    others = [g for g in GOAL_IDS if g != goal]
    for i, kind in enumerate(KINDS):
        idx = (pick == i).nonzero().flatten()
        m = idx.numel()
        if not m:
            continue
        if kind == "hang":
            out[idx] = hang_states(m, 0.1, gen)
        elif kind == "near":
            out[idx] = near_states(goal, m, 0.15, 0.5, gen)
        elif kind == "wide":
            out[idx] = wide_states(m, gen)
        else:
            for j in idx.tolist():
                g = others[int(torch.randint(len(others), (1,), generator=gen))]
                out[j] = near_states(g, 1, 0.1, 0.3, gen)[0]
    return out


@torch.no_grad()
def collect(plant, teacher, model: PlanNet | None, starts, steps, beta, extra, gen, bar=None):
    """``bar`` (tqdm) advances per knot; its postfix shows respawns and, for
    student-driven episodes, the mean |student - teacher| first-knot force."""
    b = starts.shape[0]
    knot = teacher.cfg.knot
    fmax = plant.force_limit
    keep_teacher = torch.rand(b, generator=gen) < beta if model is not None else torch.ones(b, dtype=torch.bool)
    s = starts.clone()
    plan = torch.zeros(b, teacher.cfg.n_knots)
    tplan = torch.zeros_like(plan)
    fresh = torch.ones(b, dtype=torch.bool)
    obs_l, warm_l, lab_l = [], [], []
    respawns, gap_sum, gap_n = 0, 0.0, 0
    for t in range(steps):
        if t % knot == 0:
            keep = fresh.unsqueeze(1)
            warm = torch.where(keep, plan, shift(plan))
            tplan = teacher.optimize(s, torch.where(keep, tplan, shift(tplan)))
            fresh[:] = False
            obs = observe(s)
            obs_l.append(obs)
            warm_l.append(warm)
            lab_l.append(tplan)
            if model is None:
                plan = tplan
            else:
                mine = model(obs, warm)
                driven = ~keep_teacher
                if driven.any():
                    gap_sum += float((mine[driven, 0] - tplan[driven, 0]).abs().sum())
                    gap_n += int(driven.sum())
                plan = torch.where(keep_teacher.unsqueeze(1), tplan, mine)
            if bar is not None:
                bar.update(1)
                post = {"respawns": respawns}
                if gap_n:
                    post["gap_N"] = f"{gap_sum / gap_n:.1f}"
                bar.set_postfix(post, refresh=False)
        u = teacher.force(s, plan[:, 0])
        s = step(plant, s, (u + extra[t]).clamp(-fmax, fmax))
        bad = ~torch.isfinite(s).all(-1) | (s[:, 0].abs() > plant.track_limit)
        if bad.any():
            respawns += int(bad.sum())
            s[bad] = hang_states(int(bad.sum()), 0.1, gen)
            plan[bad] = 0.0
            tplan[bad] = 0.0
            fresh |= bad
    return torch.cat(obs_l), torch.cat(warm_l), torch.cat(lab_l)


def knot_weights(n: int) -> torch.Tensor:
    w = torch.full((n,), 0.5)
    w[:8] = 1.0
    w[0] = 4.0
    return w / w.mean()


def train(model: PlanNet, O, W, Y, epochs: int, lr: float, batch: int, gen, desc: str = "train") -> float:
    Om, Wm, Ym = mirror(O, W, Y)
    O, W, Y = torch.cat((O, Om)), torch.cat((W, Wm)), torch.cat((Y, Ym))
    target = (Y - W) / PLAN_SCALE
    wk = knot_weights(Y.shape[1])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    n = O.shape[0]
    steps = epochs * ((n + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
    model.train()
    last = float("nan")
    bar = tqdm(range(epochs), desc=desc, unit="ep", leave=False, dynamic_ncols=True)
    for _ in bar:
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i : i + batch]
            err = model.delta(O[j], W[j]) - target[j]
            loss = (err * err * wk).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * j.numel()
        last = tot / n
        bar.set_postfix(loss=f"{last:.4f}")
    bar.close()
    model.eval()
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--goal", default="")
    ap.add_argument("--out", required=True, help="output prefix, e.g. ../policies/mppi/plan-uuu")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=32, help="episodes per iteration")
    ap.add_argument("--batch", type=int, default=32, help="parallel episodes")
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--betas", default="1,0.5,0.25,0.1,0")
    ap.add_argument("--p-shove", type=float, default=0.4)
    ap.add_argument("--mix", default='{"hang": 0.5, "near": 0.15, "wide": 0.2, "eq": 0.15}')
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.set_grad_enabled(True)

    plant, teacher, meta = load_teacher(args.config, args.goal or None)
    goal = meta["goal"]
    mix = json.loads(args.mix)
    betas = [float(b) for b in args.betas.split(",")]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path = out.with_name(out.name + "-data.npz")
    ckpt = out.with_name(out.name + ".pt")
    log_path = out.with_name(out.name + "-log.jsonl")
    gen = torch.Generator().manual_seed(args.seed)
    teacher.gen.manual_seed(args.seed + 1)

    model = from_teacher(teacher, tuple(int(h) for h in args.hidden.split(",")))
    O = torch.empty(0, 11)
    W = torch.empty(0, model.n_knots)
    Y = torch.empty(0, model.n_knots)
    start = 0
    if args.resume and ckpt.exists() and data_path.exists():
        model, prev = load(ckpt)
        d = np.load(data_path)
        O, W, Y = (torch.from_numpy(d[k]) for k in ("O", "W", "Y"))
        start = int(prev.get("iter", -1)) + 1

    batches = list(range(0, args.episodes, args.batch))
    knots = len(batches) * ((args.steps + teacher.cfg.knot - 1) // teacher.cfg.knot)
    overall = tqdm(range(start, args.iters), desc=f"{goal} DAgger", unit="iter", initial=start,
                   total=args.iters, position=0, dynamic_ncols=True)
    for it in overall:
        t0 = time.time()
        beta = betas[min(it, len(betas) - 1)]
        drive = None if it == 0 else model
        bar = tqdm(total=knots, desc=f"iter {it} collect (beta {beta:g})", unit="knot", leave=False,
                   position=1, dynamic_ncols=True)
        for _ in batches:
            starts = sample_starts(goal, args.batch, gen, mix)
            extra = shove_schedule(args.batch, args.steps, gen, args.p_shove)
            with torch.no_grad():
                o, w, y = collect(plant, teacher, drive, starts, args.steps, beta, extra, gen, bar)
            O, W, Y = torch.cat((O, o)), torch.cat((W, w)), torch.cat((Y, y))
        bar.close()
        np.savez_compressed(data_path, O=O.numpy(), W=W.numpy(), Y=Y.numpy())
        t1 = time.time()
        loss = train(model, O, W, Y, args.epochs, args.lr, 2048, gen, desc=f"iter {it} train")
        t2 = time.time()
        overall.set_description(f"{goal} DAgger (evaluating)")
        with torch.no_grad():
            report = evaluate(plant, PlanController(model), goal, n=args.eval_n, seed=1234)
        row = {"iter": it, "beta": beta, "samples": int(O.shape[0]), "loss": loss,
               "collect_s": round(t1 - t0), "train_s": round(t2 - t1), "eval_s": round(time.time() - t2), **report}
        save(model, ckpt, {"iter": it, "goal": goal, "teacher": meta, "eval": report})
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        overall.set_description(f"{goal} DAgger")
        overall.set_postfix(swing=f"{report['swing_success']:.0%}", enter=f"{report['swing_enter']:.0%}")
        tqdm.write(
            f"iter {it}: swing-up {report['swing_success']:.0%} (reach upright {report['swing_enter']:.0%}, "
            f"off track {report['swing_oob']:.0%}) | fit loss {loss:.4f} | {int(O.shape[0])} labels | "
            f"collect {t1 - t0:.0f}s train {t2 - t1:.0f}s"
        )
    overall.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
