#!/usr/bin/env python3
"""Export the MPPI teacher (all 8 goals) for the in-browser controller.

Writes ``web/public/mppi/triple.json``: per goal, the packed scalar parameters
(``fast_rollout.pack_params`` layout), LQR gain K and cost-to-go P that
``web/src/mppi/rollout.js`` consumes, plus the MPPI config of each profile.
The browser picks a profile (``web/src/mppi/load.js``) and patches the
knot / sub / early-exit fields of ``p`` from it.

Profiles (UUU G2, N=20, see ``speed_sweep``):
  full  4096 samples, replan every 4 steps   22 M rollout steps/s   90% (enter 100%)
  lite  2048 samples, replan every 8 steps    5 M rollout steps/s   85% (enter 95%)
Both drop to 128 samples while quiet at the target (hold mode).

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

HOLD = {"hold_samples": 128, "early_exit": True}
PROFILES = {
    "full": {**HOLD},
    "lite": {"n_samples": 2048, "knot": 8, "n_knots": 22, **HOLD},
}
KEYS = ("n_samples", "n_knots", "knot", "sigma", "sigma_wide", "wide_frac", "noise_beta", "ess", "n_iters",
        "gate_in", "gate_out", "rollout_sub", "early_exit", "hold_samples", "hold_d", "hold_spin")


def profile_cfg(config: str, over: dict) -> dict:
    raw = read_config(config, "UUU")
    raw["mppi"].update(over)
    _, _, meta = build_teacher(raw)
    return {k: meta["mppi"][k] for k in KEYS}


def export(config: str, overrides: dict | None = None, profiles: dict | None = None) -> dict:
    """``overrides`` apply on top of every profile (tests pin one config this way)."""
    profiles = profiles or PROFILES
    base = {**profiles["full"], **(overrides or {})}
    goals = {}
    for goal in GOAL_IDS:
        raw = read_config(config, goal)
        raw["mppi"].update(base)
        plant, ctrl, _ = build_teacher(raw)
        scal, K, P = pack_params(plant, ctrl.cost, ctrl.cfg)
        goals[goal] = {"p": scal.tolist(), "K": K.tolist(), "P": P.reshape(-1).tolist()}
    return {
        "fields": list(_P_FIELDS),
        "mppi": profile_cfg(config, base),
        "profiles": {name: profile_cfg(config, {**over, **(overrides or {})}) for name, over in profiles.items()},
        "force_limit": plant.force_limit,
        "goals": goals,
        "source": config,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mppi/configs/uuu.json")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--mppi", default="", help="JSON overrides applied to every profile")
    args = ap.parse_args()
    spec = export(args.config, json.loads(args.mppi) if args.mppi else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(spec))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
