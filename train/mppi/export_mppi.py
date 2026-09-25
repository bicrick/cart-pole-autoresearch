#!/usr/bin/env python3
"""Export the MPPI teacher (all 8 goals) for the in-browser controller.

Writes ``web/public/mppi/triple.json``: the MPPI config plus, per goal, the
packed scalar parameters (``fast_rollout.pack_params`` layout), LQR gain K and
cost-to-go P that ``web/src/mppi/rollout.js`` consumes.

    python -m mppi.export_mppi [--config mppi/configs/uuu.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from goals_triple import GOAL_IDS
from mppi.config import build_teacher, read_config
from mppi.fast_rollout import _P_FIELDS, pack_params

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "mppi" / "triple.json"


def export(config: str) -> dict:
    goals = {}
    cfg = None
    for goal in GOAL_IDS:
        plant, ctrl, meta = build_teacher(read_config(config, goal))
        scal, K, P = pack_params(plant, ctrl.cost, ctrl.cfg)
        goals[goal] = {"p": scal.tolist(), "K": K.tolist(), "P": P.reshape(-1).tolist()}
        cfg = meta["mppi"]
    keys = ("n_samples", "n_knots", "knot", "sigma", "sigma_wide", "wide_frac", "noise_beta", "ess", "n_iters",
            "gate_in", "gate_out")
    return {
        "fields": list(_P_FIELDS),
        "mppi": {k: cfg[k] for k in keys},
        "force_limit": plant.force_limit,
        "goals": goals,
        "source": config,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mppi/configs/uuu.json")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    spec = export(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
