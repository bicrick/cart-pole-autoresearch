#!/usr/bin/env python3
"""The browser's n-link MPPI pieces (run in node) against the Python reference.

1. Costs: ``web/src/mppi/rollout-nlink.js`` vs ``kernel_cpu.rollout_costs`` (both
   f64) on random starts and plans, anchor and early exit on.
2. Dynamics: with the anchor off, the JS kernel's final state and
   ``web/src/physics-nlink.js`` stepped with the same forces vs ``plant.step``.

    python -m mppi.nlink.test_web [--n 4]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from mppi.export_mppi_nlink import export
from mppi.nlink.costs import IDX
from mppi.nlink.kernel_cpu import rollout_costs
from mppi.nlink.plant import NLinkPlant, step

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web" / "src"

JS = r"""
import { makeRolloutNlink } from "%(rollout)s";
import { step } from "%(physics)s";
import fs from "fs";
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const n = job.n, dim = 2 + 2 * n;
const roll = makeRolloutNlink(n);
const run = (p) => job.starts.map((s0, i) => {
  const fin = new Float64Array(dim);
  const cost = roll(Float64Array.from(p), Float64Array.from(job.K), Float64Array.from(job.P),
    Float64Array.from(s0), 0, Float64Array.from(job.plans[i]), 0, job.plans[i].length, fin);
  return { cost, fin: Array.from(fin) };
});
const keys = ["x", "xd"];
for (let i = 1; i <= n; i += 1) keys.push(`th${i}`, `th${i}d`);
const out = { anchored: run(job.p).map((r) => r.cost), free: run(job.pFree).map((r) => r.fin) };
out.physics = job.starts.map((s0, i) => {
  let s = Object.fromEntries(keys.map((k, j) => [k, s0[j]]));
  for (const r of job.plans[i]) for (let q = 0; q < job.knot; q += 1) s = step(s, r, null, job.constants);
  return keys.map((k) => s[k]);
});
console.log(JSON.stringify(out));
"""


def run_node(job: dict) -> dict:
    with tempfile.TemporaryDirectory() as t:
        script = Path(t) / "nlink.mjs"
        script.write_text(JS % {"rollout": (WEB / "mppi/rollout-nlink.js").as_uri(),
                                "physics": (WEB / "physics-nlink.js").as_uri()})
        jf = Path(t) / "job.json"
        jf.write_text(json.dumps(job))
        res = subprocess.run(["node", str(script), str(jf)], check=True, capture_output=True, text=True)
        return json.loads(res.stdout)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--samples", type=int, default=256)
    args = ap.parse_args()
    n = args.n
    spec = export(n)
    plant = NLinkPlant.load(n)
    constants = json.loads((ROOT / "shared" / f"constants-{spec['plant']}.json").read_text()) if n == 4 else None
    knot, T = spec["mppi"]["knot"], spec["mppi"]["n_knots"]
    ns = len(IDX)
    rng = np.random.default_rng(0)
    ok = True
    for goal in ("U" * n, "DDUU"[:n] if n >= 4 else "D" * n):
        g = spec["goals"][goal]
        p = np.array(g["p"])
        scal, links = p[:ns], p[ns:].reshape(5, n)
        K, P = np.array(g["K"]), np.array(g["P"]).reshape(2 + 2 * n, 2 + 2 * n)
        starts = np.zeros((args.samples, 2 + 2 * n))
        starts[:, 0] = rng.uniform(-0.5, 0.5, args.samples)
        starts[:, 2::2] = rng.uniform(-math.pi, math.pi, (args.samples, n))
        starts[:, 3::2] = rng.uniform(-4, 4, (args.samples, n))
        plans = rng.normal(0, 8, (args.samples, T))
        free = scal.copy()
        free[IDX["anchor_on"]] = 0.0
        free[IDX["early_exit"]] = 0.0
        job = {"n": n, "p": p.tolist(), "pFree": np.concatenate([free, links.ravel()]).tolist(), "K": K.tolist(),
               "P": P.reshape(-1).tolist(), "starts": starts.tolist(), "plans": plans.tolist(), "knot": knot,
               "constants": constants}
        out = run_node(job)

        ref = np.array([rollout_costs(scal, links, K, P, starts[i:i + 1], plans[i:i + 1], 1)[0]
                        for i in range(args.samples)])
        rel = np.abs(np.array(out["anchored"]) - ref) / np.maximum(np.abs(ref), 1.0)
        top = lambda c: set(np.argsort(c)[:32].tolist())
        overlap = len(top(ref) & top(np.array(out["anchored"])))
        print(f"{goal} cost js vs kernel_cpu: rel err median {np.median(rel):.1e}, p95 {np.percentile(rel, 95):.1e}, "
              f"top-32 overlap {overlap}/32")
        ok = ok and np.median(rel) < 1e-9 and overlap >= 31

        s = torch.tensor(starts)
        u = torch.tensor(plans)
        for j in range(T):
            for _ in range(knot):
                s = step(plant, s, u[:, j])
        s = s.numpy()
        for name in ("free", "physics"):
            err = np.abs(np.array(out[name]) - s).max(1)
            print(f"{goal} {name} final state vs plant.step: median {np.median(err):.1e}, p90 {np.percentile(err, 90):.1e}")
            # 1.25 s of a chaotic plant: judge the bulk, not the worst roundoff blow-up.
            ok = ok and np.median(err) < 1e-6
    print("NLINK WEB PARITY OK" if ok else "NLINK WEB PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
