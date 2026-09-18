"""Discrete plant equilibria and UVFA-style goal encodings.

Convention (matches physics): theta = 0 is upright.
Xin / IFAC 2008 cart-double-pendulum equilibria:
  UU: th1=0,   th2=0
  UD: th1=0,   th2=pi
  DU: th1=pi,  th2=0
  DD: th1=pi,  th2=pi
"""

from __future__ import annotations

import math

import torch

GOAL_IDS = ("UU", "UD", "DU", "DD")
GOAL_INDEX = {name: i for i, name in enumerate(GOAL_IDS)}
NUM_GOALS = len(GOAL_IDS)

# Target angles (th1*, th2*) for each discrete goal.
GOAL_ANGLES = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, math.pi],
        [math.pi, 0.0],
        [math.pi, math.pi],
    ],
    dtype=torch.float32,
)

# State feature dim + one-hot(4) + target sin/cos(4).
STATE_OBS_DIM = 8
GOAL_ONEHOT_DIM = NUM_GOALS
GOAL_SINCCOS_DIM = 4
OBS_DIM = STATE_OBS_DIM + GOAL_ONEHOT_DIM + GOAL_SINCCOS_DIM  # 16

OBS_LAYOUT = (
    "x, xd, sinθ1, cosθ1, sinθ2, cosθ2, θ1d, θ2d, "
    "onehot_UU, onehot_UD, onehot_DU, onehot_DD, "
    "sinθ1*, cosθ1*, sinθ2*, cosθ2*"
)

# Extra bounds for goal channels appended after the 8 state features.
GOAL_OBS_LOW = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]
GOAL_OBS_HIGH = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def goal_angles(goal_ids: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """Map integer goal ids [...,] -> target angles [..., 2]."""
    table = GOAL_ANGLES.to(device=device or goal_ids.device, dtype=dtype or torch.float32)
    return table[goal_ids.long()]


def goal_encoding(goal_ids: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """One-hot + target sin/cos. goal_ids: [...] -> [..., 8]."""
    device = device or goal_ids.device
    dtype = dtype or torch.float32
    ids = goal_ids.long()
    onehot = torch.nn.functional.one_hot(ids, NUM_GOALS).to(dtype=dtype)
    angles = goal_angles(ids, device=device, dtype=dtype)
    th1 = angles[..., 0]
    th2 = angles[..., 1]
    sincos = torch.stack(
        (torch.sin(th1), torch.cos(th1), torch.sin(th2), torch.cos(th2)),
        dim=-1,
    )
    return torch.cat((onehot, sincos), dim=-1)


def sample_goals(n: int, device, dtype=torch.long) -> torch.Tensor:
    return torch.randint(0, NUM_GOALS, (n,), device=device, dtype=dtype)


def angle_align(th: torch.Tensor, th_star: torch.Tensor) -> torch.Tensor:
    """cos(th - th*) via wrap-friendly trig identity."""
    return torch.cos(th) * torch.cos(th_star) + torch.sin(th) * torch.sin(th_star)


def nearest_goal(state: torch.Tensor) -> torch.Tensor:
    """Integer goal id of the equilibrium nearest to current angles."""
    th1 = state[..., 2]
    th2 = state[..., 4]
    # alignments: [..., 4]
    table = GOAL_ANGLES.to(device=state.device, dtype=state.dtype)
    a1 = angle_align(th1.unsqueeze(-1), table[:, 0])
    a2 = angle_align(th2.unsqueeze(-1), table[:, 1])
    return (a1 + a2).argmax(dim=-1)


def at_goal(state: torch.Tensor, goal_ids: torch.Tensor, cos_thresh: float = 0.95) -> torch.Tensor:
    """Sparse success: both links aligned with the requested equilibrium."""
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    a1 = angle_align(state[..., 2], angles[..., 0])
    a2 = angle_align(state[..., 4], angles[..., 1])
    return (a1 > cos_thresh) & (a2 > cos_thresh)


def goal_reward(state, force, goal_ids, constants, sparse_bonus: float = 1.0):
    """Dense cos-alignment to goal + soft cart centering + action cost + sparse hit."""
    x = state[..., 0]
    xd = state[..., 1]
    th1 = state[..., 2]
    th1d = state[..., 3]
    th2 = state[..., 4]
    th2d = state[..., 5]
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    align = angle_align(th1, angles[..., 0]) + angle_align(th2, angles[..., 1])
    center = 0.02 * (x * x + 0.1 * xd * xd)
    spin = 0.002 * (th1d * th1d + th2d * th2d)
    effort = 0.001 * (force / constants["forceLimit"]) ** 2
    sparse = sparse_bonus * at_goal(state, goal_ids).to(state.dtype)
    return align - center - spin - effort + sparse


def conditioned_obs(state_obs: torch.Tensor, goal_ids: torch.Tensor) -> torch.Tensor:
    """Concatenate state features with goal encoding -> [..., 16]."""
    return torch.cat((state_obs, goal_encoding(goal_ids, device=state_obs.device, dtype=state_obs.dtype)), dim=-1)
