#!/usr/bin/env python3
"""Export the n-link MPPI controller for the browser (the quad by default, or the single).

Per goal the n-link kernel's scalars (``nlink.costs.FIELDS``) followed by the
per-link rows m, l, c, S, target (``p`` = scal ++ links.ravel()), the target-LQR
gain K and cost-to-go P from ``nlink.lqr``. ``web/src/mppi/rollout-nlink.js``
and ``gpu-kernel-nlink.js`` read that layout; the page plant is
``web/src/physics-nlink.js`` (same step as ``nlink.plant.step``).

Writes ``web/public/mppi/quad.json`` (n = 4) or ``single.json`` (n = 1).

    python -m mppi.export_mppi_nlink [--n 4] [--mppi '{"n_samples": 8192}']
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mppi.nlink.costs import FIELDS, pack
from mppi.nlink.goals import goal_ids
from mppi.nlink.plant import NLinkPlant

ROOT = Path(__file__).resolve().parents[2]
NAMES = {1: "single", 2: "double", 3: "triple", 4: "quad"}
# double.json / triple.json are the hand-written kernels' specs; never overwrite them.
OUTPUTS = {1: "single", 4: "quad"}

# From the L4 sweep (policies/mppi/quad): 1.25 s horizon, sigma 8 / 25 N at 40 N.
HORIZON_S = 1.25
MPPI = {
    "knot": 4, "wide_frac": 0.25,
    "noise_beta": 0.7, "ess": 32.0, "n_iters": 1, "gate_in": 0.02, "gate_out": 0.10,
    "rollout_sub": 1, "early_exit": True, "hold_samples": 128, "hold_d": 0.01, "hold_spin": 0.5,
}
# full: WebGPU (the quad's 32k replans in ~13 ms on an M2 Pro); lite: phones / CPU workers.
SAMPLES = {
    1: {"full": 1024, "lite": 512},
    4: {"full": 32768, "lite": 4096},
}


def link_fields(n: int) -> list[str]:
    return [f"{row}{i + 1}" for row in ("m", "l", "c", "S", "t") for i in range(n)]


def export(n: int = 4, overrides: dict | None = None) -> dict:
    p = NLinkPlant.load(n)
    sizes = SAMPLES.get(n, SAMPLES[4])
    # The 8 / 25 N noise was tuned at 40 N; scale it with the force limit.
    sigma = 8.0 * p.force_limit / 40.0
    cfg = {
        **MPPI,
        "n_samples": sizes["full"],
        "sigma": sigma,
        "sigma_wide": sigma * 25.0 / 8.0,
        "n_knots": max(2, round(HORIZON_S / (MPPI["knot"] * p.dt))),
        **(overrides or {}),
    }
    goals = {}
    for goal in goal_ids(n):
        pk = pack(p, goal, knot=cfg["knot"], gate_in=cfg["gate_in"], gate_out=cfg["gate_out"],
                  early_exit=cfg["early_exit"])
        goals[goal] = {
            "p": [float(v) for v in pk.scal] + [float(v) for v in pk.links.ravel()],
            "K": pk.K.tolist(),
            "P": pk.P.reshape(-1).tolist(),
            "unstable_rate": pk.unstable_rate,
        }
    return {
        "plant": NAMES.get(n, f"n{n}"),
        "n_links": n,
        "fields": list(FIELDS) + link_fields(n),
        "mppi": cfg,
        "profiles": {k: {**cfg, "n_samples": v} for k, v in sizes.items()},
        "force_limit": p.force_limit,
        "goals": goals,
        "source": f"shared/constants-{NAMES.get(n, n)}.json",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mppi", default="", help="JSON overrides for the mppi config")
    args = ap.parse_args()
    spec = export(args.n, json.loads(args.mppi) if args.mppi else None)
    name = OUTPUTS.get(args.n, f"nlink{args.n}")
    out = args.out or ROOT / "web" / "public" / "mppi" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(spec['goals'])} goals, "
          f"{spec['mppi']['n_knots']} knots)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
