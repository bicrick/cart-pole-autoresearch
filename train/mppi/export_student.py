#!/usr/bin/env python3
"""Export a distilled student to the browser policy JSON.

The fixed input-scaling layer is folded into the first hidden layer, so the
MLP is plain linear/tanh layers on raw 11-D ``observe()`` features. The spec
also carries the gated LQR ``anchor`` and ``residual_scale``; the web computes
``clip(anchor(obs) + residual_scale * tanh(mlp(obs)))``.

    python -m mppi.export_student --ckpt ../policies/mppi/student-uuu.pt --web
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from export_hold import layers_from_modules, spec_from_layers
from mppi.student import load
from physics_triple import load_constants

ROOT = Path(__file__).resolve().parents[2]
WEB_FILES = {g: f"policy-triple-{g.lower()}.json" for g in ("DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU")}
WEB_FILES["DDD"] = "policy-triple.json"


def folded_modules(model) -> list[nn.Module]:
    mods = list(model.net)
    scale, first = mods[0], mods[1]
    fused = nn.Linear(first.in_features, first.out_features)
    with torch.no_grad():
        fused.weight.copy_(first.weight @ scale.weight)
        fused.bias.copy_(first.bias + first.weight @ scale.bias)
    return [fused] + mods[2:]


def export(ckpt: Path, goal: str | None = None) -> dict:
    model, meta = load(ckpt)
    goal = goal or meta.get("goal", "UUU")
    layers = layers_from_modules(folded_modules(model))
    ev = meta.get("eval", {})
    note = (
        f"MPPI-teacher DAgger student for {goal}: swing-up, hold, and recovery in one net. "
        f"11-D observe, tanh action x force_limit. iter={meta.get('iter')} "
        f"swing_success={ev.get('swing_success')}"
    )
    spec = spec_from_layers(
        layers,
        obs_dim=11,
        hidden=list(model.hidden),
        force_limit=model.force_limit,
        constants=load_constants(),
        source=str(ckpt),
        note=note,
        goal=goal,
    )
    # Trained from hangs, wide poses, and shoves: the web keeps the current
    # state on goal switch instead of teleporting, and skips the scripted swing.
    spec["global"] = True
    spec["anchor"] = model.anchor.spec()
    spec["residual_scale"] = model.residual_scale
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--goal", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--web", action="store_true", help="also write web/public/<goal file>")
    args = ap.parse_args()
    spec = export(args.ckpt, args.goal or None)
    goal = spec["goals"][0]
    out = args.out or args.ckpt.with_suffix(".json")
    out.write_text(json.dumps(spec))
    print(f"wrote {out}")
    if args.web:
        web = ROOT / "web" / "public" / WEB_FILES[goal]
        web.write_text(json.dumps(spec))
        print(f"wrote {web}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
