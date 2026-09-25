#!/usr/bin/env python3
"""Parity of the numba rollout against the torch rollout, and its speed."""

from __future__ import annotations

import time

import torch

from mppi.config import build_teacher
from mppi.episodes import hang_states, near_states


def main() -> None:
    torch.set_grad_enabled(False)
    raw = {"cost": {"goal": "UUU"}, "mppi": {"n_samples": 4096}}
    _, fast, _ = build_teacher({**raw, "mppi": {**raw["mppi"], "backend": "numba"}})
    _, ref, _ = build_teacher({**raw, "mppi": {**raw["mppi"], "backend": "torch", "compile": False}})
    ref.cost = fast.cost
    g = torch.Generator().manual_seed(0)
    for name, s0 in (("near", near_states("UUU", 2, 0.03, 0.1, g)), ("hang", hang_states(2, 0.05, g))):
        U = ref._noise(2)
        a = ref.rollout_costs(s0, U)
        b = fast.rollout_costs(s0, U)
        rel = ((a - b).abs() / a.abs().clamp_min(1.0)).flatten()
        rank = (a.argsort(1)[:, :32] == b.argsort(1)[:, :32]).float().mean()
        print(f"{name}: rel err median {rel.median():.1e} p99 {rel.quantile(0.99):.1e}; top-32 order match {rank:.2f}")
    s0 = hang_states(1, 0.05, g)
    U = ref._noise(1)
    fast.rollout_costs(s0, U)
    for label, ctrl in (("numba", fast),):
        t0 = time.perf_counter()
        for _ in range(20):
            ctrl.rollout_costs(s0, U)
        print(f"{label}: 4096 x 180-step rollout {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms")
    t0 = time.perf_counter()
    fast.reset(1)
    for _ in range(40):
        fast.act(s0)
    print(f"numba MPPI act: {(time.perf_counter() - t0) / 10 * 1e3:.1f} ms per replan (incl. accept pass)")


if __name__ == "__main__":
    main()
