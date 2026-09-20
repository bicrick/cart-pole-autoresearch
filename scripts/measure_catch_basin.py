#!/usr/bin/env python3
"""Measure UUU catcher basin (fawraw M4-style grid).

Grid: angle offsets {0.1,0.2,0.3,0.4,0.5} × link-vel {0,1,2,3}.
Success = survive the full horizon without |x|>track_limit AND spend
≥ success_frac of steps with max|φ| < exit_tol.

Catcher backends:
  --catcher lqr          (default; ResearchSquare Qθ=100 R=0.01)
  --catcher tqc --model path/to.zip

Does NOT start GPU training. Prefer --device cpu on the box.
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

from handoff import angle_err  # noqa: E402
from lqr_uuu import design_lqr_uuu  # noqa: E402
from physics_triple import load_constants, observe, step  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Catch-basin measure for UUU hold")
    p.add_argument("--catcher", choices=("lqr", "tqc"), default="lqr")
    p.add_argument("--model", type=str, default="", help="TQC .zip path")
    p.add_argument("--force-limit", type=float, default=40.0)
    p.add_argument("--q-theta", type=float, default=100.0)
    p.add_argument("--r", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--success-frac", type=float, default=0.8)
    p.add_argument("--exit-tol", type=float, default=0.25)
    p.add_argument("--track-limit", type=float, default=2.4)
    p.add_argument("--offsets", type=str, default="0.1,0.2,0.3,0.4,0.5")
    p.add_argument("--vels", type=str, default="0,1,2,3")
    p.add_argument("--link", type=int, default=1, choices=(1, 2, 3))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="")
    return p.parse_args()


def make_catcher(args):
    if args.catcher == "lqr":
        ctrl = design_lqr_uuu(
            force_limit=args.force_limit, q_theta=args.q_theta, r=args.r
        )
        return lambda z: ctrl.force(z)

    if not args.model:
        raise SystemExit("--model required for --catcher tqc")
    from sb3_contrib import TQC

    model = TQC.load(args.model, device=args.device)

    def _force(z_np: np.ndarray) -> float:
        z = torch.as_tensor(z_np, dtype=torch.float32).reshape(1, 8)
        obs = observe(z).detach().cpu().numpy().astype(np.float32)
        act, _ = model.predict(obs, deterministic=True)
        a = float(np.asarray(act).reshape(-1)[0])
        return float(np.clip(a, -1.0, 1.0) * args.force_limit)

    return _force


def rollout(catcher_force, z0, constants, args) -> float:
    """Return fraction of steps spent near upright; 0.0 if OOB."""
    z = torch.as_tensor(z0, dtype=torch.float32).reshape(1, 8)
    good = 0
    for _ in range(args.max_steps):
        u = catcher_force(z.squeeze(0).cpu().numpy())
        force = torch.tensor([u], dtype=torch.float32)
        z = step(z, force, constants=constants)
        if abs(float(z[0, 0].item())) > args.track_limit:
            return 0.0
        if np.max(np.abs(angle_err(z.squeeze(0).cpu().numpy()))) < args.exit_tol:
            good += 1
    return good / float(args.max_steps)


def main():
    args = parse_args()
    constants = dict(load_constants())
    constants["forceLimit"] = float(args.force_limit)
    catcher = make_catcher(args)
    offsets = [float(x) for x in args.offsets.split(",") if x.strip()]
    vels = [float(x) for x in args.vels.split(",") if x.strip()]
    th_i = {1: 2, 2: 4, 3: 6}[args.link]
    w_i = {1: 3, 2: 5, 3: 7}[args.link]
    need = args.success_frac

    table = {}
    print(f"catcher={args.catcher} link={args.link} max_steps={args.max_steps} need>={need}")
    print("offset\\vel", " ".join(f"{v:>6g}" for v in vels))
    for off in offsets:
        row = []
        for vel in vels:
            scores = []
            for sign in (1.0, -1.0):
                z0 = np.zeros(8, dtype=np.float64)
                z0[th_i] = sign * off
                z0[w_i] = sign * vel
                scores.append(rollout(catcher, z0, constants, args))
            frac = float(np.mean(scores))
            ok = 1.0 if frac >= need else 0.0
            table[f"{off:g}|{vel:g}"] = {"frac": frac, "ok": ok}
            row.append(ok)
        print(f"{off:6g}", " ".join(f"{s:6.2f}" for s in row))

    n_ok = sum(1 for v in table.values() if v["ok"] >= 1.0)
    print(f"basin cells ok: {n_ok}/{len(table)}")
    if args.out:
        Path(args.out).write_text(
            json.dumps({"catcher": args.catcher, "table": table}, indent=2) + "\n"
        )
        print("wrote", args.out)


if __name__ == "__main__":
    main()
