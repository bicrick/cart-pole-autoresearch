#!/usr/bin/env python3
"""The in-browser MPPI (``web/src/mppi``, run in node) against the Python teacher.

1. Rollout parity: JS ``rolloutCost`` vs numba ``fast_rollout.rollout_costs``
   on random states and plans, for every goal.
2. Closed loop: the full JS controller (in-thread evaluator) drives the web
   physics from a hang; episodes are scored with the teacher's G2 swing metric.

    python -m mppi.test_web_mppi --goal UUU --episodes 8 --samples 4096
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from goals_triple import GOAL_IDS
from mppi.episodes import hang_states, near_states, swing_metrics
from mppi.export_mppi import export
from mppi.fast_rollout import rollout_costs

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "src"

ROLL_JS = r"""
import { rolloutCost } from "%(rollout)s";
import fs from "fs";
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const g = job.params;
const p = Float64Array.from(g.p), K = Float64Array.from(g.K), P = Float64Array.from(g.P);
const out = job.starts.map((s, i) => rolloutCost(p, K, P, Float64Array.from(s), 0, Float64Array.from(job.plans[i]), 0, job.plans[i].length));
console.log(JSON.stringify(out));
"""

LOOP_JS = r"""
import { createMppiController } from "%(controller)s";
import { createLocalEvaluator } from "%(pool)s";
import { step } from "%(physics)s";
import fs from "fs";
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ctrl = createMppiController(job.spec, createLocalEvaluator(job.spec), { samples: job.samples, adapt: false, seed: job.seed });
const keys = ["x","xd","th1","th1d","th2","th2d","th3","th3d"];
let s = Object.fromEntries(keys.map((k, i) => [k, job.start[i]]));
const states = [keys.map((k) => s[k])];
const tick = () => new Promise((r) => setImmediate(r));
for (let t = 0; t < job.steps; t += 1) {
  while (!ctrl.prepare(s, job.goal)) await tick();
  const u = ctrl.act(null, s);
  s = step(s, u, null);
  states.push(keys.map((k) => s[k]));
}
console.log(JSON.stringify({ states }));
"""


def run_node(src: str, job: dict, tmp: Path, name: str) -> dict | list:
    script = tmp / f"{name}.mjs"
    script.write_text(src)
    jf = tmp / f"{name}.json"
    jf.write_text(json.dumps(job))
    out = subprocess.run(["node", str(script), str(jf)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def uris() -> dict:
    return {k: (WEB / f).as_uri() for k, f in {
        "rollout": "mppi/rollout.js", "controller": "mppi/controller.js", "pool": "mppi/pool.js",
        "physics": "physics-triple.js"}.items()}


def rollout_parity(spec: dict, tmp: Path) -> bool:
    g = np.random.default_rng(0)
    n, T = 256, spec["mppi"]["n_knots"]
    ok = True
    for goal in GOAL_IDS:
        prm = spec["goals"][goal]
        starts = np.zeros((n, 8))
        starts[:, 0] = g.uniform(-1.5, 1.5, n)
        starts[:, 1] = g.uniform(-2, 2, n)
        for i in (2, 4, 6):
            starts[:, i] = g.uniform(-math.pi, math.pi, n)
        for i in (3, 5, 7):
            starts[:, i] = g.uniform(-6, 6, n)
        plans = g.normal(0, 20, (n, T))
        ref = rollout_costs(np.array(prm["p"]), np.array(prm["K"]), np.array(prm["P"]).reshape(8, 8), starts, plans)
        js = np.array(run_node(ROLL_JS % uris(), {"params": prm, "starts": starts.tolist(), "plans": plans.tolist()},
                               tmp, f"roll-{goal}"))
        rel = np.abs(js - ref) / np.maximum(np.abs(ref), 1.0)
        # Chaotic 1.5 s horizons amplify last-bit differences (sin/cos libm); compare the bulk.
        print(f"rollout {goal}: median rel err {np.median(rel):.1e}, p95 {np.percentile(rel, 95):.1e}")
        ok = ok and np.median(rel) < 1e-9 and np.percentile(rel, 95) < 1e-4
    return ok


def closed_loop(spec: dict, goal: str, episodes: int, samples: int, steps: int, hold: int, tmp: Path) -> dict:
    gen = torch.Generator().manual_seed(7)
    starts = near_states("UUU", episodes, 0.05, 0.01, gen) if goal == "DDD" else hang_states(episodes, 0.05, gen)

    def one(i: int):
        job = {"spec": spec, "goal": goal, "samples": samples, "steps": steps, "seed": 1000 + i,
               "start": starts[i].double().tolist()}
        return run_node(LOOP_JS % uris(), job, tmp, f"loop-{i}")["states"]

    with ThreadPoolExecutor(max_workers=episodes) as ex:
        runs = list(ex.map(one, range(episodes)))
    states = torch.tensor(runs, dtype=torch.float32).permute(1, 0, 2)
    return swing_metrics(goal, states, 2.4, hold, spec["goals"][goal]["p"][12])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal", default="UUU")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4096)
    ap.add_argument("--swing-steps", type=int, default=1500)
    ap.add_argument("--hold-steps", type=int, default=1000)
    ap.add_argument("--skip-loop", action="store_true")
    args = ap.parse_args()
    spec = export("mppi/configs/uuu.json")
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        ok = rollout_parity(spec, tmp)
        print("ROLLOUT PARITY OK" if ok else "ROLLOUT PARITY FAIL", flush=True)
        if not args.skip_loop:
            m = closed_loop(spec, args.goal, args.episodes, args.samples, args.swing_steps + args.hold_steps,
                            args.hold_steps, tmp)
            print(json.dumps({"goal": args.goal, "samples": args.samples, **m}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
