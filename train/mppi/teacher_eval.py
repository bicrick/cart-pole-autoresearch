#!/usr/bin/env python3
"""Teacher gates for the MPPI controller.

G1 hold: the target's LQR is the reference. On this plant no controller we
     could find (multi-start iLQR, CEM residual search) holds independent
     +-0.10 rad per-link starts much above ~0.4, so a fixed 0.95 bar there is
     physically out of reach. G1 therefore requires
       - quiet starts (noise 0.03, rates 0.01): survival >= 0.95
       - near starts (noise 0.05 / 0.10, rates 0.3): survival >= LQR's
       - +40 N shoves of 0.05 / 0.10 / 0.25 s from a quiet start: survival >= LQR's
     Survival means never past the fall band and never off the 2.4 m track.
G2 swing: from a hang (noise 0.05, walls off; DDD starts near UUU), enter the quiet box
     (|theta| <= 0.03, |omega| <= 0.01) and stay in the fall band for
     ``hold_steps``, cart inside the track. Pass: success >= 0.9.

    python -m mppi.teacher_eval --config mppi/configs/uuu.json --gate all --n 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mppi.config import build_teacher, read_config
from mppi.costs import wrap
from mppi.episodes import EpisodeSpec, Shove, hang_states, hold_metrics, near_states, run, swing_metrics

QUIET_BAR = 0.95
G2_BAR = 0.90
# Rates are float32 means (19/20 -> 0.94999999); compare with slack.
BAR_EPS = 1e-6
SHOVE_STEPS = (6, 12, 30)


class LQRController:
    """Ungated target LQR, the G1 reference."""

    def __init__(self, cost, force_limit: float):
        self.K = cost.K_lqr
        self.tgt = cost.target_angles
        self.fmax = force_limit

    def reset(self, batch: int) -> None:
        pass

    def act(self, s: torch.Tensor) -> torch.Tensor:
        e = s.clone()
        for j, i in enumerate((2, 4, 6)):
            e[:, i] = wrap(s[:, i] - self.tgt[j])
        return (-(e * self.K).sum(-1)).clamp(-self.fmax, self.fmax)


def g1_cases(goal: str, n: int, steps: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    yield "quiet_0.03", near_states(goal, n, 0.03, 0.01, g), EpisodeSpec(goal=goal, steps=steps)
    for noise in (0.05, 0.10):
        yield f"near_{noise:.2f}", near_states(goal, n, noise, 0.3, g), EpisodeSpec(goal=goal, steps=steps)
    for k in SHOVE_STEPS:
        s0 = near_states(goal, n, 0.02, 0.01, g)
        yield f"shove_40N_{k / 120:.2f}s", s0, EpisodeSpec(goal=goal, steps=steps, shoves=[Shove(120, k, 40.0)])


def gate_g1(plant, ctrl, goal: str, n: int, steps: int, seed: int, ref=None) -> dict:
    out = {}
    for name, s0, spec in g1_cases(goal, n, steps, seed):
        states, _ = run(plant, ctrl, s0, spec)
        out[name] = hold_metrics(goal, states, plant.track_limit)
        if ref is not None:
            rs, _ = run(plant, ref, s0, spec)
            out[name]["lqr_survival"] = hold_metrics(goal, rs, plant.track_limit)["survival"]
    out["pass"] = g1_pass(out)
    return out


def g1_pass(out: dict) -> bool:
    ok = out["quiet_0.03"]["survival"] >= QUIET_BAR - BAR_EPS
    for name, m in out.items():
        if isinstance(m, dict) and "lqr_survival" in m and name != "quiet_0.03":
            ok = ok and m["survival"] >= m["lqr_survival"] - BAR_EPS
    return bool(ok)


def swing_start(goal: str, n: int, g: torch.Generator) -> torch.Tensor:
    """Hang for every target except DDD, which starts near UUU (bring it down)."""
    if goal.upper() == "DDD":
        return near_states("UUU", n, 0.05, 0.01, g)
    return hang_states(n, 0.05, g)


def gate_g2(plant, ctrl, goal: str, n: int, swing_steps: int, hold_steps: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    s0 = swing_start(goal, n, g)
    states, forces = run(plant, ctrl, s0, EpisodeSpec(goal=goal, steps=swing_steps + hold_steps))
    m = swing_metrics(goal, states, plant.track_limit, hold_steps, plant.dt)
    m["force_saturated_frac"] = float((forces.abs() >= plant.force_limit - 1e-3).float().mean())
    m["pass"] = m["success"] >= G2_BAR - BAR_EPS
    return {"swing": m, "pass": m["pass"]}


def _merge(parts: list[dict]) -> dict:
    """Episode-weighted mean of per-chunk metrics."""
    out: dict = {}
    for key in parts[0]:
        if key == "pass":
            continue
        subs = [p[key] for p in parts]
        n = sum(s["n"] for s in subs)
        m = {"n": n}
        for f in subs[0]:
            if f == "n":
                continue
            vals = [s[f] for s in subs]
            if isinstance(vals[0], bool):
                m[f] = all(vals)
            elif f.startswith("max_"):
                m[f] = max(vals)
            elif f.startswith("min_"):
                m[f] = min(vals)
            else:
                m[f] = sum(v * s["n"] for v, s in zip(vals, subs)) / n
        out[key] = m
    if "swing" in out:
        out["swing"]["pass"] = out["swing"]["success"] >= G2_BAR - BAR_EPS
        out["pass"] = out["swing"]["pass"]
    else:
        out["pass"] = g1_pass(out)
    return out


def apply_overrides(m: dict, args) -> dict:
    """Speed knobs from the CLI onto a raw mppi config dict (in place)."""
    if args.mppi:
        m.update(json.loads(args.mppi))
    if args.samples:
        m["n_samples"] = args.samples
    if args.rollout_sub:
        m["rollout_sub"] = args.rollout_sub
    if args.replan_every:
        horizon = m.get("n_knots", 45) * m.get("knot", 4)
        m["knot"] = args.replan_every
        m["n_knots"] = max(2, round(horizon / args.replan_every))
    if args.hold_samples >= 0:
        m["hold_samples"] = args.hold_samples
    if args.early_exit:
        m["early_exit"] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--goal", default="")
    ap.add_argument("--gate", choices=("g1", "g2", "all"), default="all")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--batch", type=int, default=10, help="episodes run in parallel")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--swing-steps", type=int, default=1500)
    ap.add_argument("--hold-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--mppi", default="", help='JSON overrides for the mppi config, e.g. \'{"n_samples": 1024}\'')
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--rollout-sub", type=int, default=0)
    ap.add_argument("--replan-every", type=int, default=0, help="knot length in steps; keeps the horizon")
    ap.add_argument("--hold-samples", type=int, default=-1)
    ap.add_argument("--early-exit", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    raw = read_config(args.config, args.goal or None)
    apply_overrides(raw["mppi"], args)
    plant, ctrl, meta = build_teacher(raw)
    goal = meta["goal"]
    ref = LQRController(ctrl.cost, plant.force_limit)
    report = {"config": args.config, "goal": goal, "teacher": meta}
    t0 = time.time()
    chunks = [(i, min(args.batch, args.n - i)) for i in range(0, args.n, args.batch)]

    def merged(fn):
        return _merge([fn(b, args.seed + i) for i, b in chunks])

    if args.gate in ("g1", "all"):
        report["g1"] = merged(lambda b, s: gate_g1(plant, ctrl, goal, b, args.steps, s, ref))
        print(json.dumps({"g1": report["g1"]}, indent=1), flush=True)
    if args.gate in ("g2", "all"):
        report["g2"] = merged(lambda b, s: gate_g2(plant, ctrl, goal, b, args.swing_steps, args.hold_steps, s))
        print(json.dumps({"g2": report["g2"]}, indent=1), flush=True)
    report["wall_s"] = time.time() - t0
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=1))
    ok = all(report[k]["pass"] for k in ("g1", "g2") if k in report)
    print(f"{goal} teacher {'PASS' if ok else 'FAIL'} ({report['wall_s']:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
