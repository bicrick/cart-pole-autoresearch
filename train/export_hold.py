#!/usr/bin/env python3
"""Export the UUU hold actor (BC .pt or TQC zip) to browser policy JSON.

Writes policies/policy-triple-hold.json and web/public/policy-triple.json.
The demo reads 11-D observe() (x, xd, sin/cos×3, ω×3), no goal one-hot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = Path(__file__).resolve().parent
for p in (_ROOT, _TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bc_lqr_uuu import load_ckpt  # noqa: E402
from physics_triple import load_constants  # noqa: E402

OUT_POLICIES = _ROOT / "policies" / "policy-triple-hold.json"
OUT_WEB = _ROOT / "web" / "public" / "policy-triple.json"
OBS_LAYOUT = ["x", "xd", "sin1", "cos1", "sin2", "cos2", "sin3", "cos3", "w1", "w2", "w3"]


def parse_args():
    p = argparse.ArgumentParser(description="Export UUU hold policy for the web demo")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "policies" / "bc-lqr-uuu.pt",
        help="BC .pt (default) or TQC .zip",
    )
    p.add_argument("--out", type=Path, default=OUT_POLICIES)
    p.add_argument("--web-out", type=Path, default=OUT_WEB)
    p.add_argument("--force-limit", type=float, default=None)
    p.add_argument("--goal", type=str, default="UUU", help="Equilibrium this specialist holds (UUU, DDD, ...)")
    return p.parse_args()


def layers_from_modules(modules) -> list[dict]:
    layers: list[dict] = []
    for module in modules:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "weight": module.weight.detach().cpu().tolist(),
                    "bias": module.bias.detach().cpu().tolist(),
                }
            )
        elif isinstance(module, nn.ReLU):
            layers.append({"type": "relu"})
        elif isinstance(module, nn.Tanh):
            layers.append({"type": "tanh"})
    return layers


def spec_from_layers(layers, *, obs_dim, hidden, force_limit, constants, source, note, goal="UUU"):
    physics = dict(constants)
    physics["forceLimit"] = float(force_limit)
    physics["obsDim"] = int(obs_dim)
    return {
        "obs_dim": int(obs_dim),
        "obs_layout": OBS_LAYOUT,
        "goals": [goal],
        "num_goals": 1,
        "n_links": 3,
        "hidden": hidden,
        "act_dim": 1,
        "force_limit": float(force_limit),
        "physics": physics,
        "layers": layers,
        "source": source,
        "note": note,
    }


def export_bc(path: Path, force_limit: float | None, constants: dict, goal: str = "UUU") -> dict:
    model, meta = load_ckpt(path, torch.device("cpu"))
    fl = float(force_limit or meta.get("force_limit") or constants.get("forceLimit", 40.0))
    hidden = list(model.arch)
    layers = layers_from_modules(model.net)
    return spec_from_layers(
        layers,
        obs_dim=model.obs_dim,
        hidden=hidden,
        force_limit=fl,
        constants=constants,
        source=str(path),
        note=f"BC LQR {goal} hold. 11-D observe, tanh action × force_limit. No goal conditioning.",
        goal=goal,
    )


def export_tqc(path: Path, force_limit: float | None, constants: dict, goal: str = "UUU") -> dict:
    from sb3_contrib import TQC

    load = str(path.with_suffix("") if path.suffix == ".zip" else path)
    model = TQC.load(load, device="cpu")
    actor = model.policy.actor
    modules = list(actor.latent_pi) + [actor.mu]
    layers = layers_from_modules(modules)
    fl = float(force_limit or constants.get("forceLimit", 40.0))
    hidden = [m.out_features for m in actor.latent_pi if isinstance(m, nn.Linear)]
    return spec_from_layers(
        layers,
        obs_dim=int(actor.latent_pi[0].in_features),
        hidden=hidden,
        force_limit=fl,
        constants=constants,
        source=str(path),
        note=f"TQC {goal} hold actor (latent_pi + mu). 11-D observe, tanh action × force_limit.",
        goal=goal,
    )


def main() -> int:
    args = parse_args()
    constants = load_constants()
    ckpt = args.checkpoint
    if not ckpt.is_file() and ckpt.suffix == ".zip" and ckpt.with_suffix("").is_file():
        ckpt = ckpt.with_suffix("")
    if not ckpt.is_file():
        print(f"MISSING {ckpt}", flush=True)
        return 2
    suffix = ckpt.suffix.lower()
    if suffix == ".pt":
        spec = export_bc(ckpt, args.force_limit, constants, args.goal)
    elif suffix in (".zip", "") or "tqc" in ckpt.name:
        spec = export_tqc(ckpt, args.force_limit, constants, args.goal)
    else:
        print(f"unknown checkpoint type: {ckpt}", flush=True)
        return 2
    text = json.dumps(spec)
    for dest in (args.out, args.web_out):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        print(f"exported {dest} obs_dim={spec['obs_dim']} source={spec['source']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
