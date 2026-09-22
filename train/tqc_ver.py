"""Baek VER left–right replay flip for the 11-D UUU hold observer.

Planar cart-triple is equivariant under (x, xd, θ, ω, F) → (−x, −xd, −θ, −ω, −F).
cos θ is even. Reward is even. Each collected transition is stored twice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer

# 11-D observe: x, xd, sin1, cos1, sin2, cos2, sin3, cos3, w1, w2, w3
FLIP_HOLD_OBS_IDX = (0, 1, 2, 4, 6, 8, 9, 10)


def flip_hold_obs(obs: np.ndarray) -> np.ndarray:
    out = np.array(obs, copy=True)
    out[..., list(FLIP_HOLD_OBS_IDX)] *= -1.0
    return out


def flip_hold_action(action: np.ndarray) -> np.ndarray:
    return np.array(action, copy=True) * -1.0


class VERReplayBuffer(ReplayBuffer):
    """Store each transition and its left–right reflection (Baek VER)."""

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        super().add(obs, next_obs, action, reward, done, infos)
        super().add(
            flip_hold_obs(obs),
            flip_hold_obs(next_obs),
            flip_hold_action(action),
            reward,
            done,
            infos,
        )
