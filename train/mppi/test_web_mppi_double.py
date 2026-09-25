#!/usr/bin/env python3
"""The in-browser MPPI on the double pendulum (``web/src/mppi``, run in node).

1. Dynamics parity: with the anchor off, ``rollout-double.js`` must land on the
   same final state as stepping the page plant (``physics.js``) with the same
   forces, for exact and rotation-updated trig.
2. Closed loop: the pipelined controller drives ``physics.js`` from a hang
   (from near UU for DD); scored like the triple's G2 swing-up gate.

    python -m mppi.test_web_mppi_double --episodes 8
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

from mppi.export_mppi_double import FIELDS, GOALS, export

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "src"
KEYS = ["x", "xd", "th1", "th1d", "th2", "th2d"]
FALL_UP, FALL_DOWN, QUIET_ANGLE, QUIET_RATE = 0.6, 1.5, 0.03, 0.01

PARITY_JS = r"""
import { rolloutCostDouble } from "%(rollout)s";
import { setFastTrig } from "%(triple)s";
import { step } from "%(physics)s";
import fs from "fs";
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const p = Float64Array.from(job.p), K = Float64Array.from(job.K), P = Float64Array.from(job.P);
const keys = ["x","xd","th1","th1d","th2","th2d"];
const out = {};
for (const fast of [false, true]) {
  setFastTrig(fast);
  out[fast ? "fast" : "exact"] = job.starts.map((s0, i) => {
    const fin = new Float64Array(6);
    rolloutCostDouble(p, K, P, Float64Array.from(s0), 0, Float64Array.from(job.plans[i]), 0, job.plans[i].length, fin);
    return Array.from(fin);
  });
}
out.physics = job.starts.map((s0, i) => {
  let s = Object.fromEntries(keys.map((k, j) => [k, s0[j]]));
  for (const r of job.plans[i]) for (let q = 0; q < job.knot; q += 1) s = step(s, r, null);
  return keys.map((k) => s[k]);
});
console.log(JSON.stringify(out));
"""

LOOP_JS = r"""
import { createMppiController } from "%(controller)s";
import { createLocalEvaluator } from "%(pool)s";
import { step } from "%(physics)s";
import fs from "fs";
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const ctrl = createMppiController(job.spec, createLocalEvaluator(job.spec), {
  samples: job.samples, adapt: false, seed: job.seed, step,
});
const keys = ["x","xd","th1","th1d","th2","th2d"];
let s = Object.fromEntries(keys.map((k, i) => [k, job.start[i]]));
const states = [keys.map((k) => s[k])];
const tick = () => new Promise((r) => setImmediate(r));
for (let t = 0; t < job.steps; t += 1) {
  while (!ctrl.prepare(s, job.goal)) await tick();
  s = step(s, ctrl.act(null, s), null);
  states.push(keys.map((k) => s[k]));
}
console.log(JSON.stringify({ states }));
"""


def uris() -> dict:
    return {k: (WEB / f).as_uri() for k, f in {
        "rollout": "mppi/rollout-double.js", "triple": "mppi/rollout.js", "controller": "mppi/controller.js",
        "pool": "mppi/pool.js", "physics": "physics.js"}.items()}


def run_node(src: str, job: dict, tmp: Path, name: str):
    script = tmp / f"{name}.mjs"
    script.write_text(src % uris())
    jf = tmp / f"{name}.json"
    jf.write_text(json.dumps(job))
    return json.loads(subprocess.run(["node", str(script), str(jf)], check=True, capture_output=True,
                                     text=True).stdout)


def parity(spec: dict, tmp: Path) -> bool:
    g = np.random.default_rng(0)
    n, T = 128, spec["mppi"]["n_knots"]
    prm = dict(spec["goals"]["UU"])
    p = list(prm["p"])
    p[FIELDS.index("anchor_on")] = 0.0
    p[FIELDS.index("early_exit")] = 0.0
    starts = np.zeros((n, 6))
    starts[:, 0] = g.uniform(-0.5, 0.5, n)
    starts[:, 2] = g.uniform(-math.pi, math.pi, n)
    starts[:, 4] = g.uniform(-math.pi, math.pi, n)
    starts[:, 3] = g.uniform(-4, 4, n)
    starts[:, 5] = g.uniform(-4, 4, n)
    plans = g.normal(0, 6, (n, T))
    out = run_node(PARITY_JS, {"p": p, "K": prm["K"], "P": prm["P"], "starts": starts.tolist(),
                               "plans": plans.tolist(), "knot": spec["mppi"]["knot"]}, tmp, "parity")
    ref = np.array(out["physics"])
    ok = True
    for mode in ("exact", "fast"):
        fin = np.array(out[mode])
        err = np.abs(fin - ref)
        # 1.5 s of a chaotic plant: compare the bulk, not the worst roundoff blow-up.
        med = float(np.median(err.max(1)))
        print(f"dynamics vs physics.js ({mode} trig): median max-state err {med:.1e}, "
              f"p90 {np.percentile(err.max(1), 90):.1e}")
        ok = ok and med < 1e-8
    return ok


def metrics(goal: str, states: np.ndarray, hold: int, dt: float) -> dict:
    """states [T+1, B, 6] -> the triple G2 swing metric, two links."""
    tgt = np.array(GOALS[goal])
    ang = states[..., [2, 4]]
    err = np.abs((ang - tgt + np.pi) % (2 * np.pi) - np.pi)
    rate = np.abs(states[..., [3, 5]])
    quiet = (err <= QUIET_ANGLE).all(-1) & (rate <= QUIET_RATE).all(-1)
    band = np.where(np.abs(tgt) < 1e-6, FALL_UP, FALL_DOWN)
    t_total, b = quiet.shape
    entered = quiet.any(0)
    first = np.where(entered, quiet.argmax(0), t_total)
    after = np.arange(t_total)[:, None] >= first[None, :]
    fell = entered & ((err > band).any(-1) & after).any(0)
    late = entered & ((t_total - 1 - first) < hold)
    oob = (np.abs(states[..., 0]) > 2.4).any(0)
    success = entered & ~fell & ~late & ~oob
    return {
        "enter": float(entered.mean()),
        "success": float(success.mean()),
        "fell": float(fell.mean()),
        "oob": float(oob.mean()),
        "median_enter_s": float(np.median(first[entered]) * dt) if entered.any() else float("nan"),
    }


def closed_loop(spec: dict, goal: str, episodes: int, samples: int, steps: int, hold: int, tmp: Path) -> dict:
    g = np.random.default_rng(7)
    base = np.array([0, 0, 0.0, 0, 0.0, 0]) if goal == "DD" else np.array([0, 0, math.pi, 0, math.pi, 0])
    starts = base + g.uniform(-0.05, 0.05, (episodes, 6)) * np.array([1, 0.2, 1, 0.2, 1, 0.2])

    def one(i: int):
        job = {"spec": spec, "goal": goal, "samples": samples, "steps": steps, "seed": 1000 + i,
               "start": starts[i].tolist()}
        return run_node(LOOP_JS, job, tmp, f"loop-{goal}-{i}")["states"]

    with ThreadPoolExecutor(max_workers=episodes) as ex:
        runs = list(ex.map(one, range(episodes)))
    states = np.transpose(np.array(runs), (1, 0, 2))
    return metrics(goal, states, hold, spec["goals"][goal]["p"][FIELDS.index("dt")])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", default="UU,UD,DU,DD")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--samples", type=int, default=0, help="default: the exported n_samples")
    ap.add_argument("--swing-steps", type=int, default=1500)
    ap.add_argument("--hold-steps", type=int, default=1000)
    ap.add_argument("--mppi", default="", help="JSON overrides for the exported mppi config")
    ap.add_argument("--skip-loop", action="store_true")
    args = ap.parse_args()
    spec = export(json.loads(args.mppi) if args.mppi else None)
    samples = args.samples or spec["mppi"]["n_samples"]
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        ok = parity(spec, tmp)
        print("DOUBLE DYNAMICS PARITY OK" if ok else "DOUBLE DYNAMICS PARITY FAIL", flush=True)
        if not args.skip_loop:
            for goal in args.goals.split(","):
                m = closed_loop(spec, goal, args.episodes, samples, args.swing_steps + args.hold_steps,
                                args.hold_steps, tmp)
                print(json.dumps({"goal": goal, "samples": samples, **{k: round(v, 3) for k, v in m.items()}}),
                      flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
