#!/usr/bin/env python3
"""Generic N-link step vs the hand-written plants: n=2 ``physics.step``, n=3 ``physics_triple.step``.

    python -m mppi.nlink.test_plant
"""

from __future__ import annotations

import math
from dataclasses import replace

import torch

import physics
import physics_triple
from mppi.nlink.plant import NLinkPlant, step


def random_states(n: int, count: int, gen: torch.Generator) -> torch.Tensor:
    s = torch.empty(count, 2 + 2 * n, dtype=torch.float64)
    s[:, 0].uniform_(-2, 2, generator=gen)
    s[:, 1].uniform_(-4, 4, generator=gen)
    s[:, 2::2].uniform_(-math.pi, math.pi, generator=gen)
    s[:, 3::2].uniform_(-12, 12, generator=gen)
    return s


def check(n: int, ref_step, consts: dict, jitter: float) -> bool:
    g = torch.Generator().manual_seed(n)
    p = replace(NLinkPlant.from_constants(consts), jitter=jitter)
    s = random_states(n, 512, g)
    u = torch.empty(512, dtype=torch.float64).uniform_(-60, 60, generator=g)
    one = (step(p, s, u) - ref_step(s, u, constants=consts)).abs().max().item()
    a, b = s[:64].clone(), s[:64].clone()
    for _ in range(120):
        f = torch.empty(64, dtype=torch.float64).uniform_(-p.force_limit, p.force_limit, generator=g)
        a, b = step(p, a, f), ref_step(b, f, constants=consts)
    roll = (a - b).abs().max(-1).values
    print(f"n={n}: one-step max err {one:.1e}; 1 s rollout max-state err median {roll.median():.1e}")
    return one < 1e-9 and roll.median() < 1e-6


def main() -> int:
    ok = check(2, physics.step, physics.load_constants(), jitter=0.0)
    ok = check(3, physics_triple.step, physics_triple.load_constants(), jitter=1e-8) and ok
    print("NLINK PLANT PARITY OK" if ok else "NLINK PLANT PARITY FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
