#!/usr/bin/env python3
"""Parity of ``mppi.dynamics_fast.step`` against ``physics_triple.step``."""

from __future__ import annotations

import math

import torch

import physics_triple
from mppi.dynamics_fast import Plant, step


def _random_states(n: int, gen: torch.Generator) -> torch.Tensor:
    s = torch.empty(n, 8, dtype=torch.float64)
    s[:, 0].uniform_(-2.0, 2.0, generator=gen)
    s[:, 1].uniform_(-4.0, 4.0, generator=gen)
    for i in (2, 4, 6):
        s[:, i].uniform_(-math.pi, math.pi, generator=gen)
    for i in (3, 5, 7):
        s[:, i].uniform_(-15.0, 15.0, generator=gen)
    return s


def main() -> None:
    gen = torch.Generator().manual_seed(0)
    for walls in (False, True):
        consts = dict(physics_triple.load_constants())
        consts["trackWalls"] = walls
        plant = Plant.from_constants(consts)
        s = _random_states(4096, gen)
        u = torch.empty(4096, dtype=torch.float64).uniform_(-60.0, 60.0, generator=gen)
        ref = physics_triple.step(s, u, constants=consts)
        got = step(plant, s, u)
        err = (ref - got).abs().max().item()
        assert err < 1e-9, f"walls={walls} float64 max err {err}"

        ref_roll = s[:256].clone()
        got_roll = s[:256].clone().float()
        uu = u[:256]
        for _ in range(120):
            ref_roll = physics_triple.step(ref_roll, uu, constants=consts)
            got_roll = step(plant, got_roll, uu.float())
        drift = (ref_roll[:, [0, 2, 4, 6]].float() - got_roll[:, [0, 2, 4, 6]]).abs()
        print(f"walls={walls} one-step f64 err {err:.2e}; 1 s f32 rollout median drift {drift.median():.2e}")
    print("dynamics_fast parity OK")


if __name__ == "__main__":
    main()
