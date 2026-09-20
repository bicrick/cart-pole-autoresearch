#!/usr/bin/env python3
"""Assert quiet-basin IC scales: init_noise controls θ, ω, x/xd (no legacy floors)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from envs.triple_gym import _sample_state  # noqa: E402
from goals_triple import GOAL_INDEX  # noqa: E402
from train_triple import random_states  # noqa: E402


def _scales(init_noise: float):
    n_ang = max(float(init_noise), 1e-3)
    n_x = min(0.5, max(1e-3, n_ang * 2.0))
    n_xd = min(0.5, max(1e-3, n_ang * 2.0))
    n_w = min(0.8, max(1e-3, n_ang * 5.0))
    return n_ang, n_x, n_xd, n_w


def test_proportional_scales():
    # With init_noise=0.02, legacy floors forced ω±0.1 and cart±0.05 — must NOT bind.
    n_ang, n_x, n_xd, n_w = _scales(0.02)
    assert abs(n_ang - 0.02) < 1e-9
    assert abs(n_x - 0.04) < 1e-9, f"expected n_x=0.04 got {n_x} (legacy floor was 0.05)"
    assert abs(n_xd - 0.04) < 1e-9
    assert abs(n_w - 0.10) < 1e-9, f"expected n_w=0.10 got {n_w}"
    # Smaller still: floors are 1e-3 only
    n_ang, n_x, n_xd, n_w = _scales(0.001)
    assert n_x == 1e-3 or abs(n_x - 0.002) < 1e-12
    assert n_w == max(1e-3, 0.005)
    # M2 0.05
    n_ang, n_x, n_xd, n_w = _scales(0.05)
    assert abs(n_ang - 0.05) < 1e-9
    assert abs(n_x - 0.10) < 1e-9
    assert abs(n_w - 0.25) < 1e-9


def test_gym_samples_respect_noise():
    torch.manual_seed(0)
    noise = 0.05
    n_ang, n_x, n_xd, n_w = _scales(noise)
    xs, xds, ths, ws = [], [], [], []
    for _ in range(200):
        s = _sample_state("near_target", hang_frac=0.0, wide_frac=0.0, noise=noise, device=torch.device("cpu"))
        xs.append(abs(float(s[0])))
        xds.append(abs(float(s[1])))
        ths.extend([abs(float(s[2])), abs(float(s[4])), abs(float(s[6]))])
        ws.extend([abs(float(s[3])), abs(float(s[5])), abs(float(s[7]))])
    assert max(xs) <= n_x + 1e-5, f"cart |x| max {max(xs)} > n_x {n_x}"
    assert max(xds) <= n_xd + 1e-5
    assert max(ths) <= n_ang + 1e-5
    assert max(ws) <= n_w + 1e-5
    # Must actually use the quiet range (not legacy ±0.5 / ±0.8)
    assert max(xs) < 0.3, "samples look like legacy ±0.5 cart basin"
    assert max(ws) < 0.5, "samples look like legacy ±0.8 ω basin"
    print(
        f"gym near_target noise={noise}: max|x|={max(xs):.4f} (cap {n_x}) "
        f"max|ω|={max(ws):.4f} (cap {n_w}) max|θ|={max(ths):.4f} (cap {n_ang})"
    )


def test_random_states_near_target():
    torch.manual_seed(1)
    noise = 0.02
    n_ang, n_x, n_xd, n_w = _scales(noise)
    goals = torch.full((64,), int(GOAL_INDEX["UUU"]), dtype=torch.long)
    s = random_states(
        64,
        device=torch.device("cpu"),
        constants={},
        goals=goals,
        init_mode="near_target",
        init_noise=noise,
        hang_start_p=0.0,
    )
    assert s.shape == (64, 8)
    assert float(s[:, 0].abs().max()) <= n_x + 1e-5
    assert float(s[:, 1].abs().max()) <= n_xd + 1e-5
    # angles near 0 for UUU
    for i in (2, 4, 6):
        assert float(s[:, i].abs().max()) <= n_ang + 1e-5
    for i in (3, 5, 7):
        assert float(s[:, i].abs().max()) <= n_w + 1e-5
    # Legacy floor would force some |ω| up to 0.1 always available; with noise=0.02
    # n_w=0.1 exactly via ×5 — check cart is below legacy 0.05 floor when noise smaller:
    n_ang2, n_x2, _, n_w2 = _scales(0.01)
    assert n_x2 < 0.05, "quiet-basin must allow cart scale < legacy 0.05 floor"
    assert n_w2 < 0.1 or abs(n_w2 - 0.05) < 1e-9
    s2 = random_states(
        128,
        device=torch.device("cpu"),
        constants={},
        goals=torch.full((128,), int(GOAL_INDEX["UUU"]), dtype=torch.long),
        init_mode="near_target",
        init_noise=0.01,
        hang_start_p=0.0,
    )
    assert float(s2[:, 0].abs().max()) <= n_x2 + 1e-5
    assert float(s2[:, 3].abs().max()) <= n_w2 + 1e-5
    print(
        f"random_states near_target noise=0.02: max|x|={float(s[:,0].abs().max()):.4f} "
        f"max|ω|={float(s[:,3].abs().max()):.4f} (caps x={n_x} ω={n_w})"
    )
    print(
        f"random_states near_target noise=0.01: max|x|={float(s2[:,0].abs().max()):.4f} "
        f"< legacy floor 0.05; max|ω|={float(s2[:,3].abs().max()):.4f} (cap {n_w2})"
    )


if __name__ == "__main__":
    test_proportional_scales()
    test_gym_samples_respect_noise()
    test_random_states_near_target()
    print("OK quiet-basin IC asserts passed")
