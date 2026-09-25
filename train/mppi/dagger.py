#!/usr/bin/env python3
"""DAgger: distill the MPPI teacher into a ``Student`` MLP.

Labels are the teacher's residual on its gated LQR anchor; the student has the
same anchor built in and learns only that residual.

Each iteration runs B parallel episodes. The executed force comes from the
teacher with probability ``beta`` per episode and from the student otherwise.
The teacher replans from every visited state regardless of who drove, so every
state gets a label. When the student drives, the teacher's plan is
warm-started from the student's own open-loop residuals, which keeps the
label on the student's side of left/right swing choices.

Random mid-episode shoves (+-40 N, 0.1-0.3 s) widen coverage.

    python -m mppi.dagger --config mppi/configs/uuu.json --out ../policies/mppi/student-uuu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mppi.config import load_teacher
from mppi.dynamics_fast import Plant, step
from mppi.episodes import hang_states, near_states, wide_states
from mppi.student import Student, StudentController, from_teacher, load, mirror, save
from mppi.student_eval import evaluate
from physics_triple import observe


def sample_starts(goal: str, n: int, gen: torch.Generator, mix: dict) -> torch.Tensor:
    kinds = ["hang", "near", "wide"]
    p = torch.tensor([mix[k] for k in kinds], dtype=torch.float)
    pick = torch.multinomial(p / p.sum(), n, replacement=True, generator=gen)
    parts = torch.empty(n, 8)
    for i, kind in enumerate(kinds):
        idx = (pick == i).nonzero().flatten()
        if not idx.numel():
            continue
        m = idx.numel()
        if kind == "hang":
            parts[idx] = hang_states(m, 0.1, gen)
        elif kind == "near":
            parts[idx] = near_states(goal, m, 0.15, 0.5, gen)
        else:
            parts[idx] = wide_states(m, gen)
    return parts


def shove_schedule(b: int, steps: int, gen: torch.Generator, p_shove: float) -> torch.Tensor:
    """Per-env extra force [T,B]."""
    extra = torch.zeros(steps, b)
    for i in range(b):
        if torch.rand(1, generator=gen).item() >= p_shove:
            continue
        for _ in range(int(torch.randint(1, 3, (1,), generator=gen))):
            start = int(torch.randint(0, max(1, steps - 60), (1,), generator=gen))
            dur = int(torch.randint(12, 37, (1,), generator=gen))
            sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
            extra[start : start + dur, i] = 40.0 * sign
    return extra


@torch.no_grad()
def student_plan(model: Student, teacher, plant: Plant, states: torch.Tensor) -> torch.Tensor:
    """Student's open-loop behaviour from ``states`` as a teacher residual plan [B,T]."""
    cfg = teacher.cfg
    s = states.clone()
    out = []
    for _ in range(cfg.n_knots):
        u = model.force(s)
        out.append(model.residual(s))
        for _ in range(cfg.knot):
            s = step(plant, s, u)
    return torch.stack(out, dim=1)


@torch.no_grad()
def collect(plant, teacher, model: Student | None, starts, steps, beta, extra, gen):
    b = starts.shape[0]
    cfg = teacher.cfg
    teacher_drives = torch.rand(b, generator=gen) < beta if model is not None else torch.ones(b, dtype=torch.bool)
    teacher.reset(b)
    s = starts.clone()
    xs, ys = [], []
    fmax = plant.force_limit
    for t in range(steps):
        if t % cfg.knot == 0:
            U = teacher.uff
            if t > 0:
                U = torch.cat((U[:, 1:], torch.zeros_like(U[:, -1:])), 1)
            if model is not None and (~teacher_drives).any():
                warm = student_plan(model, teacher, plant, s)
                U = torch.where(teacher_drives.unsqueeze(1), U, warm)
            teacher.uff = teacher.optimize(s, U)
        residual = teacher.uff[:, 0]
        label = teacher.force(s, residual)
        xs.append(observe(s))
        ys.append(residual / (2.0 * fmax))
        u = label if model is None else torch.where(teacher_drives, label, model.force(s))
        s = step(plant, s, (u + extra[t]).clamp(-fmax, fmax))
        bad = ~torch.isfinite(s).all(-1) | (s[:, 0].abs() > 3.5)
        if bad.any():
            s[bad] = sample_starts_like(s[bad], gen)
    return torch.cat(xs), torch.cat(ys)


def sample_starts_like(s: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    return hang_states(s.shape[0], 0.1, gen)


def train_student(model: Student, X: torch.Tensor, Y: torch.Tensor, epochs: int, lr: float, batch: int, gen) -> float:
    Xm, Ym = mirror(X, Y)
    X = torch.cat((X, Xm))
    Y = torch.cat((Y, Ym)).clamp(-1.0, 1.0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    n = X.shape[0]
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i : i + batch]
            loss = ((model(X[j]) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * j.numel()
        last = tot / n
    model.eval()
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--goal", default="")
    ap.add_argument("--out", required=True, help="output prefix, e.g. ../policies/mppi-uuu")
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=32, help="episodes per iteration")
    ap.add_argument("--batch", type=int, default=16, help="parallel episodes")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--beta-decay", type=float, default=0.6)
    ap.add_argument("--p-shove", type=float, default=0.5)
    ap.add_argument("--mix", default='{"hang": 0.5, "near": 0.25, "wide": 0.25}')
    ap.add_argument("--hidden", default="256,256")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    plant, teacher, meta = load_teacher(args.config, args.goal or None)
    goal = meta["goal"]
    mix = json.loads(args.mix)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data_path = out.with_name(out.name + "-data.npz")
    ckpt = out.with_name(out.name + ".pt")
    log_path = out.with_name(out.name + "-log.jsonl")
    gen = torch.Generator().manual_seed(args.seed)

    hidden = tuple(int(h) for h in args.hidden.split(","))
    model = from_teacher(teacher, hidden)
    X = torch.empty(0, 11)
    Y = torch.empty(0)
    start_iter = 0
    if args.resume and ckpt.exists() and data_path.exists():
        model, prev = load(ckpt)
        d = np.load(data_path)
        X, Y = torch.from_numpy(d["X"]), torch.from_numpy(d["Y"])
        start_iter = int(prev.get("iter", -1)) + 1

    for it in range(start_iter, args.iters):
        t0 = time.time()
        beta = args.beta0 * (args.beta_decay**it)
        drive = None if it == 0 and not args.resume else model
        xs, ys = [], []
        for _ in range(0, args.episodes, args.batch):
            starts = sample_starts(goal, args.batch, gen, mix)
            extra = shove_schedule(args.batch, args.steps, gen, args.p_shove)
            x, y = collect(plant, teacher, drive, starts, args.steps, beta, extra, gen)
            xs.append(x)
            ys.append(y)
        X = torch.cat([X] + xs)
        Y = torch.cat([Y] + ys)
        np.savez_compressed(data_path, X=X.numpy(), Y=Y.numpy())
        loss = train_student(model, X, Y, args.epochs, args.lr, 4096, gen)
        report = evaluate(plant, StudentController(model), goal, n=args.eval_n, seed=1234)
        row = {"iter": it, "beta": beta, "samples": int(X.shape[0]), "loss": loss, "wall_s": time.time() - t0, **report}
        save(model, ckpt, {"iter": it, "goal": goal, "teacher": meta, "eval": report})
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
