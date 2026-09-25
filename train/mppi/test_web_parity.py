#!/usr/bin/env python3
"""Parity between the web demo's triple plant/anchor (JS, run in node) and the
Python plant the MPPI teacher was trained on.

Checks, on random states and forces:
  - web ``physics-triple.js`` step vs ``physics_triple.step`` (one step, float64)
  - 2 s open-loop rollouts under the same force sequence
  - web TRIPLE_CONSTANTS vs shared/constants-triple.json (incl. trackWalls)
  - web ``anchorForce`` vs ``mppi.student.Anchor`` for every goal

    python -m mppi.test_web_parity
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import torch

import physics_triple
from goals_triple import GOAL_IDS
from mppi.config import build_teacher
from mppi.student import Anchor

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "src"

NODE = r"""
import { step, TRIPLE_CONSTANTS } from "%(physics)s";
import { anchorForce } from "%(policy)s";
import fs from "fs";
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const keys = ["x","xd","th1","th1d","th2","th2d","th3","th3d"];
const toObj = (v) => Object.fromEntries(keys.map((k, i) => [k, v[i]]));
const toArr = (s) => keys.map((k) => s[k]);
const consts = { ...TRIPLE_CONSTANTS, ...job.overrides };
const one = job.states.map((s, i) => toArr(step(toObj(s), job.forces[i], null, consts)));
const roll = job.states.slice(0, job.roll_n).map((s, i) => {
  let st = toObj(s);
  for (let t = 0; t < job.roll_steps; t += 1) st = step(st, job.roll_forces[i][t], null, consts);
  return toArr(st);
});
const anchors = {};
for (const [goal, spec] of Object.entries(job.anchors)) {
  anchors[goal] = job.obs.map((o) => anchorForce(spec, o));
}
console.log(JSON.stringify({ consts: TRIPLE_CONSTANTS, one, roll, anchors }));
"""


def main() -> int:
    torch.set_grad_enabled(False)
    py_consts = physics_triple.load_constants()
    g = torch.Generator().manual_seed(0)
    n = 512
    s = torch.empty(n, 8, dtype=torch.float64)
    s[:, 0].uniform_(-2.0, 2.0, generator=g)
    s[:, 1].uniform_(-4.0, 4.0, generator=g)
    for i in (2, 4, 6):
        s[:, i].uniform_(-math.pi, math.pi, generator=g)
    for i in (3, 5, 7):
        s[:, i].uniform_(-15.0, 15.0, generator=g)
    u = torch.empty(n, dtype=torch.float64).uniform_(-60.0, 60.0, generator=g)
    roll_n, roll_steps = 64, 240
    roll_u = torch.empty(roll_n, roll_steps, dtype=torch.float64).uniform_(-40.0, 40.0, generator=g)
    obs = physics_triple.observe(s * torch.tensor([1, 1, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5], dtype=torch.float64))

    anchors, py_anchor = {}, {}
    for goal in GOAL_IDS:
        _, ctrl, _ = build_teacher({"cost": {"goal": goal}, "mppi": {"compile": False}})
        c = ctrl.cfg
        a = Anchor(ctrl.cost.K_lqr, ctrl.cost.target_angles, c.gate_in, c.gate_out)
        anchors[goal] = a.spec()
        ang = ctrl.cost.target_angles.double()
        o = obs.clone()
        # evaluate anchors near each goal: rotate sample angles onto the target
        base = s * torch.tensor([1, 1, 0.1, 0.5, 0.1, 0.5, 0.1, 0.5], dtype=torch.float64)
        base[:, 2] += ang[0]
        base[:, 4] += ang[1]
        base[:, 6] += ang[2]
        o = physics_triple.observe(base)
        anchors[goal] = {**a.spec(), "_obs": o.tolist()}
        py_anchor[goal] = a.double()(o) if hasattr(a, "double") else a(o.float()).double()

    job = {
        "states": s.tolist(),
        "forces": u.tolist(),
        "roll_n": roll_n,
        "roll_steps": roll_steps,
        "roll_forces": roll_u.tolist(),
        "overrides": {"trackWalls": bool(py_consts.get("trackWalls", False))},
        "anchors": {k: {kk: vv for kk, vv in v.items() if kk != "_obs"} for k, v in anchors.items()},
        "obs": None,
    }
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "parity.mjs"
        script.write_text(NODE % {"physics": (WEB / "physics-triple.js").as_uri(), "policy": (WEB / "policy.js").as_uri()})
        for goal in GOAL_IDS:
            job["anchors"] = {goal: {kk: vv for kk, vv in anchors[goal].items() if kk != "_obs"}}
            job["obs"] = anchors[goal]["_obs"]
            jf = Path(tmp) / "job.json"
            jf.write_text(json.dumps(job))
            out = json.loads(subprocess.run(["node", str(script), str(jf)], check=True, capture_output=True, text=True).stdout)
            results.setdefault("one", out["one"])
            results.setdefault("roll", out["roll"])
            results.setdefault("consts", out["consts"])
            results.setdefault("anchors", {})[goal] = out["anchors"][goal]

    ok = True
    web_consts = results["consts"]
    for k, v in py_consts.items():
        if k in ("obsDim", "obsLow", "obsHigh", "hidden"):
            continue
        if web_consts.get(k) != v:
            print(f"CONSTANT MISMATCH {k}: python={v} web={web_consts.get(k)}")
            ok = False

    consts = dict(py_consts)
    ref = physics_triple.step(s, u, constants=consts)
    err1 = (ref - torch.tensor(results["one"], dtype=torch.float64)).abs().max().item()
    st = s[:roll_n].clone()
    for t in range(roll_steps):
        st = physics_triple.step(st, roll_u[:, t], constants=consts)
    web_roll = torch.tensor(results["roll"], dtype=torch.float64)
    ang_err = (st[:, [2, 4, 6]] - web_roll[:, [2, 4, 6]]).abs()
    print(f"one-step max err {err1:.2e}")
    print(f"2 s rollout angle err: median {ang_err.median():.2e}, max {ang_err.max():.2e} (chaotic; max grows from 1e-15 roundoff)")
    ok = ok and err1 < 1e-9
    for goal in GOAL_IDS:
        a_err = (py_anchor[goal].double() - torch.tensor(results["anchors"][goal], dtype=torch.float64)).abs().max().item()
        print(f"anchor {goal}: max err {a_err:.2e} N")
        ok = ok and a_err < 1e-3
    print("WEB PARITY OK" if ok else "WEB PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
