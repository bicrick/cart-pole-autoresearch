#!/usr/bin/env python3
"""Behavior cloning from LQR UUU demos → MLP hold policy (fawraw M2 env contract).

Demos (policies/lqr-uuu-demos.npz):
  states (N,8) raw plant; actions (N,) Newtons.
We train on gym features: observe(state) → 11-D, action_norm = Newtons / force_limit ∈ [-1,1]
so the policy matches TriplePendulumUUUEnv / TQC actor I/O.

Arch default [128,128] (TQC pi). Loss: Huber (or MSE). Saves policies/bc-lqr-uuu.pt.
Eval: survival / at_goal @ init_noise 0.01 and 0.02, N≥50 (same as LQR oracle).

CPU-only. Does NOT start TQC / GCP.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
from physics_triple import observe  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="BC from LQR UUU demos")
    p.add_argument("--demos", type=str, default="policies/lqr-uuu-demos.npz")
    p.add_argument("--out", type=str, default="policies/bc-lqr-uuu.pt")
    p.add_argument("--arch", type=str, default="128,128")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--loss", type=str, default="huber", choices=("huber", "mse"))
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--noises", type=str, default="0.01,0.02")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--pass-threshold", type=float, default=0.90)
    p.add_argument("--track-walls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--goal", "--target-ep", dest="goal", default="UUU")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-train", action="store_true", help="Load --out and eval only")
    return p.parse_args()


def _parse_arch(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class BCActor(nn.Module):
    """Deterministic MLP: obs → action in [-1,1] (tanh). Matches TQC pi width."""

    def __init__(self, obs_dim: int = 11, act_dim: int = 1, arch: list[int] | None = None):
        super().__init__()
        arch = arch or [128, 128]
        layers: list[nn.Module] = []
        d = obs_dim
        for h in arch:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.arch = list(arch)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(obs_np, dtype=torch.float32).reshape(1, -1)
        a = self.forward(x).cpu().numpy().reshape(-1)
        return a.astype(np.float32)


def states_to_obs(states: np.ndarray) -> np.ndarray:
    """Raw 8-D plant → 11-D gym observe features."""
    t = torch.as_tensor(states, dtype=torch.float32)
    return observe(t).detach().cpu().numpy().astype(np.float32)


def load_demos(path: Path, force_limit: float) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.load(path, allow_pickle=True)
    states = np.asarray(raw["states"], dtype=np.float32)
    actions_N = np.asarray(raw["actions"], dtype=np.float32).reshape(-1)
    meta = {}
    if "meta_json" in raw.files:
        try:
            meta = json.loads(str(raw["meta_json"].item() if hasattr(raw["meta_json"], "item") else raw["meta_json"]))
        except Exception:
            meta = {}
    fl = float(meta.get("force_limit", force_limit))
    obs = states_to_obs(states)
    act = (actions_N / fl).astype(np.float32).reshape(-1, 1)
    act = np.clip(act, -1.0, 1.0)
    print(
        f"demos: N={obs.shape[0]} obs_dim={obs.shape[1]} "
        f"act_N=[{actions_N.min():.3g},{actions_N.max():.3g}] "
        f"act_norm=[{act.min():.3g},{act.max():.3g}] force_limit={fl}",
        flush=True,
    )
    return obs, act, {**meta, "force_limit": fl}


def train_bc(args) -> tuple[BCActor, dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    obs, act, demo_meta = load_demos(Path(args.demos), args.force_limit)
    n = obs.shape[0]
    idx = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _loader(ii, shuffle):
        ds = TensorDataset(
            torch.as_tensor(obs[ii], dtype=torch.float32),
            torch.as_tensor(act[ii], dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=False)

    train_loader = _loader(tr_idx, True)
    val_loader = _loader(val_idx, False)

    arch = _parse_arch(args.arch)
    model = BCActor(obs_dim=obs.shape[1], act_dim=1, arch=arch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == "huber":
        crit = nn.SmoothL1Loss()
    else:
        crit = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(float(loss.item()))
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                va_losses.append(float(crit(model(xb), yb).item()))
        tr_m = float(np.mean(tr_losses))
        va_m = float(np.mean(va_losses))
        history.append({"epoch": ep, "train_loss": tr_m, "val_loss": va_m})
        if va_m < best_val:
            best_val = va_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep == 1 or ep % 10 == 0 or ep == args.epochs:
            print(
                f"epoch {ep:3d}/{args.epochs}  train={tr_m:.5f}  val={va_m:.5f}  "
                f"best_val={best_val:.5f}",
                flush=True,
            )

    assert best_state is not None
    model.load_state_dict(best_state)
    train_meta = {
        "algo": "BC",
        "source_demos": str(args.demos),
        "demo_meta": demo_meta,
        "arch": arch,
        "obs_dim": int(obs.shape[1]),
        "act_dim": 1,
        "action_space": "[-1,1] (Newtons/force_limit)",
        "input": "11-D observe(state)",
        "loss": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_val_loss": best_val,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "force_limit": float(demo_meta.get("force_limit", args.force_limit)),
        "goal": getattr(args, "goal", "UUU"),
        "env_contract": "fawraw M2 quiet basin + fall-kill + walls ON",
        "train_seconds": round(time.time() - t0, 2),
        "history_tail": history[-5:],
    }
    return model, train_meta


def save_ckpt(model: BCActor, meta: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "obs_dim": model.obs_dim,
        "act_dim": model.act_dim,
        "arch": model.arch,
        "meta": meta,
    }
    torch.save(payload, path)
    print(f"CKPT → {path}", flush=True)


def load_ckpt(path: Path, device: torch.device) -> tuple[BCActor, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    arch = list(payload.get("arch", [128, 128]))
    model = BCActor(
        obs_dim=int(payload.get("obs_dim", 11)),
        act_dim=int(payload.get("act_dim", 1)),
        arch=arch,
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, dict(payload.get("meta") or {})


def rollout_episode(env: TriplePendulumUUUEnv, model: BCActor) -> dict:
    obs, info = env.reset()
    for _ in range(env.max_steps):
        a = model.act(obs)
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    survived = bool(info.get("survival_success", False))
    if "survival_success" not in info:
        ep_len = int(info.get("ep_len", env._step_count))
        survived = (not info.get("fell", False)) and (not info.get("oob", False)) and (
            ep_len >= int(SURVIVAL_FRAC * env.max_steps)
        )
    try:
        goal_ok = bool(
            at_goal(env._state.unsqueeze(0), env._goal.unsqueeze(0)).item()
        )
    except Exception:
        goal_ok = bool(info.get("at_goal_final", info.get("at_goal", False)))
    return {
        "survived": survived,
        "at_goal": goal_ok,
        "ep_len": int(info.get("ep_len", env._step_count)),
        "fell": bool(info.get("fell", False)),
        "oob": bool(info.get("oob", False)),
    }


def eval_noise(model: BCActor, noise: float, args) -> dict:
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
        goal=getattr(args, "goal", "UUU"),
    )
    n_ok = 0
    n_goal = 0
    lens = []
    for i in range(args.n_episodes):
        torch.manual_seed(args.seed + 17 * i + int(noise * 1e4))
        np.random.seed(args.seed + 17 * i + int(noise * 1e4))
        ep = rollout_episode(env, model)
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


def main() -> int:
    args = parse_args()
    print(f"=== BC LQR {args.goal} ===", flush=True)
    print(
        f"env contract: quiet_rate=±{QUIET_RATE}, fall_thresh={FALL_THRESH_UP}, "
        f"survival_frac={SURVIVAL_FRAC}, walls={args.track_walls}",
        flush=True,
    )

    out = Path(args.out)
    if args.skip_train:
        model, train_meta = load_ckpt(out, torch.device(args.device))
        print(f"loaded {out}", flush=True)
    else:
        model, train_meta = train_bc(args)
        save_ckpt(model, train_meta, out)

    results = []
    all_pass = True
    if not args.skip_eval:
        noises = [float(x) for x in args.noises.split(",") if x.strip()]
        for n in noises:
            print(f"\n--- eval init_noise={n}  N={args.n_episodes} ---", flush=True)
            t1 = time.time()
            res = eval_noise(model, n, args)
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

    summary = {
        "bc": "PASS" if (all_pass and results) else ("SKIP_EVAL" if not results else "FAIL"),
        "pass_threshold": args.pass_threshold,
        "ckpt": str(out),
        "results": results,
        "train": {
            "best_val_loss": train_meta.get("best_val_loss"),
            "arch": train_meta.get("arch"),
            "train_seconds": train_meta.get("train_seconds"),
        },
        "ready_for_tqc_ft": bool(all_pass and results),
        "note": "GPU TQC FT needs separate user green-light; do not start GCP from here.",
    }
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(("BC PASS" if (all_pass and results) else "BC FAIL/SKIP"), flush=True)
    return 0 if (all_pass and results) else (0 if args.skip_eval else 1)


if __name__ == "__main__":
    raise SystemExit(main())
