#!/usr/bin/env python3
"""Learned terminal value for a shorter MPPI horizon.

Label for a visited teacher state s_t: the teacher's realized running cost over
the next ``--tail`` seconds plus the analytic terminal cost there, i.e. the part
of the 1.5 s horizon that a shorter horizon drops. The fit is plain regression
on log1p(V), so the distillation label problem (plan-dependent actions) does
not apply. MPPI uses it via ``MPPIConfig.terminal_value``.

    python -m mppi.value --config mppi/configs/uuu.json --out ../policies/mppi/value-uuu.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from mppi.config import build_teacher, read_config
from mppi.dagger import shove_schedule
from mppi.dagger_plan import sample_starts
from mppi.dynamics_fast import step
from mppi.student import OBS_SCALE
from physics_triple import observe


class ValueNet(nn.Module):
    def __init__(self, hidden=(128, 128)):
        super().__init__()
        self.hidden = tuple(hidden)
        layers: list[nn.Module] = []
        d = 11
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", OBS_SCALE.clone())

    def log_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs / self.scale).squeeze(-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Cost-to-go for states [..., 8]."""
        return torch.expm1(self.log_value(observe(states.float())).clamp(0.0, 16.0))


def save(model: ValueNet, path: Path, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "hidden": model.hidden, "meta": meta}, path)


def load(path: str | Path) -> ValueNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ValueNet(tuple(ck["hidden"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


@torch.no_grad()
def collect(plant, teacher, starts, steps, extra):
    """Teacher episodes -> states [T+1,B,8], per-step running cost [T,B], alive mask [T+1,B]."""
    b = starts.shape[0]
    teacher.reset(b)
    s = starts.clone()
    fmax = plant.force_limit
    states, costs, alive = [s], [], [torch.ones(b, dtype=torch.bool)]
    ok = torch.ones(b, dtype=torch.bool)
    zero = torch.zeros(b)
    for t in range(steps):
        u = (teacher.act(s) + extra[t]).clamp(-fmax, fmax)
        s = step(plant, s, u)
        costs.append(teacher.cost.running(s, u, zero))
        ok = ok & torch.isfinite(s).all(-1) & (s[:, 0].abs() <= plant.track_limit)
        states.append(s)
        alive.append(ok.clone())
    return torch.stack(states), torch.stack(costs), torch.stack(alive)


def labels(teacher, states, costs, alive, tail: int):
    """(states [N,8], V [N]) for every step with a full, alive tail window."""
    t_total = costs.shape[0]
    csum = torch.cat((torch.zeros(1, costs.shape[1]), costs.cumsum(0)))
    xs, vs = [], []
    for t in range(0, t_total - tail + 1):
        ok = alive[t + tail]
        if not ok.any():
            continue
        run = csum[t + tail] - csum[t]
        term = teacher.cost.terminal(states[t + tail])
        xs.append(states[t][ok])
        vs.append((run + term)[ok])
    return torch.cat(xs), torch.cat(vs)


def fit(X, V, hidden, epochs, lr, gen) -> tuple[ValueNet, float]:
    model = ValueNet(hidden)
    obs = observe(X.float())
    y = torch.log1p(V.clamp_min(0.0)).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = obs.shape[0]
    batch = 4096
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * ((n + batch - 1) // batch), eta_min=lr * 0.05)
    bar = tqdm(range(epochs), desc="fit V", unit="ep", dynamic_ncols=True)
    last = float("nan")
    for _ in bar:
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch):
            j = perm[i : i + batch]
            loss = ((model.log_value(obs[j]) - y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * j.numel()
        last = tot / n
        bar.set_postfix(loss=f"{last:.4f}")
    return model.eval(), last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mppi/configs/uuu.json")
    ap.add_argument("--goal", default="UUU")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2400)
    ap.add_argument("--tail", type=float, default=1.0, help="seconds of horizon the value replaces")
    ap.add_argument("--p-shove", type=float, default=0.4)
    ap.add_argument("--mix", default='{"hang": 0.5, "near": 0.15, "wide": 0.25, "eq": 0.1}')
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    raw = read_config(args.config, args.goal)
    raw["mppi"].update({"hold_samples": 128, "early_exit": True})
    plant, teacher, meta = build_teacher(raw)
    tail = round(args.tail / plant.dt)
    gen = torch.Generator().manual_seed(args.seed)
    mix = json.loads(args.mix)
    out = Path(args.out)
    data = out.with_name(out.stem + "-data.pt")
    t0 = time.time()
    if data.exists():
        d = torch.load(data)
        X, V = d["X"], d["V"]
    else:
        xs, vs = [], []
        for _ in tqdm(range(0, args.episodes, args.batch), desc="teacher episodes", unit="batch", dynamic_ncols=True):
            starts = sample_starts(args.goal, args.batch, gen, mix)
            extra = shove_schedule(args.batch, args.steps, gen, args.p_shove)
            st, c, al = collect(plant, teacher, starts, args.steps, extra)
            x, v = labels(teacher, st, c, al, tail)
            xs.append(x)
            vs.append(v)
        X, V = torch.cat(xs), torch.cat(vs)
        torch.save({"X": X, "V": V}, data)
    torch.set_grad_enabled(True)
    model, loss = fit(X, V, tuple(int(h) for h in args.hidden.split(",")), args.epochs, args.lr, gen)
    save(model, out, {"goal": args.goal, "tail_s": args.tail, "labels": int(X.shape[0]), "loss": loss, "teacher": meta})
    print(json.dumps({"out": str(out), "labels": int(X.shape[0]), "fit_loss_log1p": loss, "wall_s": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    raise SystemExit(main())
