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


def sample_goals(n: int, device, dtype=torch.long, allowed=None, probs=None) -> torch.Tensor:
    """Sample goal ids.

    - probs: length-NUM_GOALS weights (any device); categorical over all goals.
    - allowed: uniform over an explicit id list/tensor (legacy hard curriculum).
    - else: uniform over all goals.
    """
    if probs is not None:
        p = probs if torch.is_tensor(probs) else torch.tensor(probs, dtype=torch.float32)
        p = p.to(device=device, dtype=torch.float32).clamp_min(0)
        s = p.sum()
        if s <= 0:
            p = torch.ones(NUM_GOALS, device=device, dtype=torch.float32) / NUM_GOALS
        else:
            p = p / s
        return torch.multinomial(p, n, replacement=True).to(dtype=dtype)
    if allowed is None:
        return torch.randint(0, NUM_GOALS, (n,), device=device, dtype=dtype)
    if not torch.is_tensor(allowed):
        allowed = torch.tensor(list(allowed), device=device, dtype=torch.long)
    else:
        allowed = allowed.to(device=device, dtype=torch.long)
    idx = torch.randint(0, allowed.numel(), (n,), device=device)
    return allowed[idx].to(dtype=dtype)


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


def goal_reward(
    state,
    force,
    goal_ids,
    constants,
    sparse_bonus: float = 1.0,
    align_w: float = 1.5,
    energy_w: float = 0.15,
    spin_w: float = 0.0003,
    center_w: float = 0.06,
    center_hold_w: float = 0.18,
    track_limit: float = 4.0,
    reward_clip: float = 8.0,
    oob_penalty: float = 20.0,
):
    """Dense goal reward with bounded penalties (see docs/paper-training-lessons.md).

    - align_w * (cosΔθ1 + cosΔθ2): denser tracking of the requested equilibrium.
    - energy_w * (align − soft kinetic): Xin/Spong-style push toward goal potential
      at rest; kinetic is soft-capped so swing-up pumping is not crushed.
    - Cart / spin penalties use soft-capped magnitudes so a drifting cart cannot
      produce −10k episode returns (root cause of the collapsed GCP eval curve).
    - center_w always pulls toward x=0; center_hold_w ramps up as links align so
      "balanced at the rail" is not a free lunch (Xin regulates cart with energy).
    - oob_penalty on |x| > track_limit (applied AFTER reward_clip so void-death
      stays catastrophic; train also marks done).
    """
    x = state[..., 0]
    xd = state[..., 1]
    th1 = state[..., 2]
    th1d = state[..., 3]
    th2 = state[..., 4]
    th2d = state[..., 5]
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    a1 = angle_align(th1, angles[..., 0])
    a2 = angle_align(th2, angles[..., 1])
    align = a1 + a2

    # Soft-bound cart penalty; stronger when already near the goal pose.
    x_eff = x.clamp(-track_limit, track_limit)
    xd_eff = xd.clamp(-20.0, 20.0)
    hold = (0.5 * (a1.clamp(min=0.0) + a2.clamp(min=0.0))).clamp(0.0, 1.0)
    center_coef = center_w + center_hold_w * hold
    center = center_coef * (x_eff * x_eff + 0.2 * xd_eff * xd_eff)
    oob = (x.abs() > track_limit).to(state.dtype)

    # Soft-cap spin so energy pumping (Spong) is not dominated by ω².
    spin_raw = th1d * th1d + th2d * th2d
    spin = spin_w * spin_raw.clamp(max=200.0)

    effort = 0.001 * (force / constants["forceLimit"]) ** 2
    sparse = sparse_bonus * at_goal(state, goal_ids).to(state.dtype)

    # Energy-to-goal: favor alignment and modest kinetic near the target.
    kin_soft = (0.05 * xd_eff * xd_eff + 0.02 * spin_raw).clamp(max=40.0)
    energy = energy_w * (align - 0.25 * kin_soft)

    rew = align_w * align + energy - center - spin - effort + sparse
    if reward_clip is not None and reward_clip > 0:
        rew = rew.clamp(-reward_clip, reward_clip)
    # Void-death after clip so a −20 terminal hit is not softened to −8.
    rew = rew - float(oob_penalty) * oob
    return rew


def conditioned_obs(state_obs: torch.Tensor, goal_ids: torch.Tensor) -> torch.Tensor:
    """Concatenate state features with goal encoding -> [..., 16]."""
    return torch.cat((state_obs, goal_encoding(goal_ids, device=state_obs.device, dtype=state_obs.dtype)), dim=-1)
