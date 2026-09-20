#!/usr/bin/env python3
"""Assert fawraw quiet-basin ICs: θ/x ~ ±init_noise; xd/ω FIXED ±0.01."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TRAIN = Path(__file__).resolve().parent
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from envs.triple_gym import QUIET_RATE, _sample_state  # noqa: E402
from goals_triple import GOAL_INDEX  # noqa: E402
from train_triple import random_states  # noqa: E402


def _scales(init_noise: float):
    n_ang = max(float(init_noise), 1e-3)
    n_x = n_ang
    n_rate = float(QUIET_RATE)  # 0.01 fixed
    return n_ang, n_x, n_rate


def test_quiet_rate_contract():
    for noise in (0.01, 0.02, 0.05):
        n_ang, n_x, n_rate = _scales(noise)
        assert abs(n_ang - noise) < 1e-9 or noise < 1e-3
        assert abs(n_x - n_ang) < 1e-9
        assert abs(n_rate - 0.01) < 1e-12, f"rates must be fixed 0.01, got {n_rate}"
        # Must NOT be the old ×5 / ×2 scaling
        assert n_rate != min(0.8, max(1e-3, n_ang * 5.0)) or noise == 0.002
        old_w = min(0.8, max(1e-3, n_ang * 5.0))
        assert n_rate < old_w or noise <= 0.002, (
            f"quiet rate {n_rate} should be << old ×5 scale {old_w} at noise={noise}"
        )
    print("quiet-rate contract: θ/x±noise, xd/ω=±0.01 fixed OK")


def test_gym_samples_respect_noise():
    torch.manual_seed(0)
    noise = 0.05
    n_ang, n_x, n_rate = _scales(noise)
    xs, xds, ths, ws = [], [], [], []
    for _ in range(200):
        s = _sample_state(
            "near_target", hang_frac=0.0, wide_frac=0.0, noise=noise, device=torch.device("cpu")
        )
        xs.append(abs(float(s[0])))
        xds.append(abs(float(s[1])))
        ths.extend([abs(float(s[2])), abs(float(s[4])), abs(float(s[6]))])
        ws.extend([abs(float(s[3])), abs(float(s[5])), abs(float(s[7]))])
    assert max(xs) <= n_x + 1e-5, f"cart |x| max {max(xs)} > n_x {n_x}"
    assert max(xds) <= n_rate + 1e-5, f"|xd| max {max(xds)} > {n_rate}"
    assert max(ths) <= n_ang + 1e-5
    assert max(ws) <= n_rate + 1e-5, f"|ω| max {max(ws)} > {n_rate} (old ×5 leaked?)"
    assert max(ws) < 0.05, "samples look like legacy ×5 ω basin"
    print(
        f"gym near_target noise={noise}: max|x|={max(xs):.4f} (cap {n_x}) "
        f"max|ω|={max(ws):.4f} (cap {n_rate}) max|θ|={max(ths):.4f} (cap {n_ang})"
    )


def test_random_states_near_target():
    torch.manual_seed(1)
    for noise in (0.01, 0.02, 0.05):
        n_ang, n_x, n_rate = _scales(noise)
        s = random_states(
            128,
            device=torch.device("cpu"),
            constants={},
            goals=torch.full((128,), int(GOAL_INDEX["UUU"]), dtype=torch.long),
            init_mode="near_target",
            init_noise=noise,
            hang_start_p=0.0,
        )
        assert float(s[:, 0].abs().max()) <= n_x + 1e-5
        assert float(s[:, 1].abs().max()) <= n_rate + 1e-5
        for i in (2, 4, 6):
            assert float(s[:, i].abs().max()) <= n_ang + 1e-5
        for i in (3, 5, 7):
            assert float(s[:, i].abs().max()) <= n_rate + 1e-5
        print(
            f"random_states noise={noise}: max|x|={float(s[:,0].abs().max()):.4f} "
            f"max|ω|={float(s[:,3].abs().max()):.4f} (caps x={n_x} ω={n_rate})"
        )


def test_fall_kill_flag():
    from envs.triple_gym import TriplePendulumUUUEnv

    env = TriplePendulumUUUEnv(init_noise=0.01, max_steps=100, progress_w=0.0, seed=0)
    obs, _ = env.reset(seed=0)
    # Force a fallen state
    env._state[2] = 0.7  # θ1 past 0.6
    obs, r, term, trunc, info = env.step([0.0])
    assert term is True, "expected fall termination"
    assert info.get("fell") is True
    assert info.get("is_success") is False
    print("fall-kill |θ|>0.6 → terminated OK")


if __name__ == "__main__":
    test_quiet_rate_contract()
    test_gym_samples_respect_noise()
    test_random_states_near_target()
    test_fall_kill_flag()
    print("OK quiet-basin IC asserts passed")
