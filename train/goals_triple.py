"""Discrete triple-pendulum equilibria and UVFA-style goal encodings.

Convention (matches physics_triple): theta = 0 is upright.
Eight equilibria (each link Up=0 or Down=π), ordered DDD … UUU with
MSB = link1 (same letter order as double UU/UD/DU/DD):

  DDD DDU DUD DUU UDD UDU UUD UUU
"""

from __future__ import annotations

import math

import torch

GOAL_IDS = ("DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU")
GOAL_INDEX = {name: i for i, name in enumerate(GOAL_IDS)}
NUM_GOALS = len(GOAL_IDS)  # 8
NUM_TRANSITIONS = NUM_GOALS * (NUM_GOALS - 1)  # 56

# Target angles (th1*, th2*, th3*). U=0, D=π; bits th1|th2|th3 with D=0 U=1
# in the GOAL_IDS order above (DDD=000 … UUU=111 under that bit map).
_PI = math.pi
GOAL_ANGLES = torch.tensor(
    [
        [_PI, _PI, _PI],  # DDD
        [_PI, _PI, 0.0],  # DDU
        [_PI, 0.0, _PI],  # DUD
        [_PI, 0.0, 0.0],  # DUU
        [0.0, _PI, _PI],  # UDD
        [0.0, _PI, 0.0],  # UDU
        [0.0, 0.0, _PI],  # UUD
        [0.0, 0.0, 0.0],  # UUU
    ],
    dtype=torch.float32,
)

# State features (11) + one-hot(8) + target sin/cos(6).
STATE_OBS_DIM = 11
GOAL_ONEHOT_DIM = NUM_GOALS
GOAL_SINCCOS_DIM = 6
OBS_DIM = STATE_OBS_DIM + GOAL_ONEHOT_DIM + GOAL_SINCCOS_DIM  # 25

OBS_LAYOUT = (
    "x, xd, sinθ1, cosθ1, sinθ2, cosθ2, sinθ3, cosθ3, θ1d, θ2d, θ3d, "
    "onehot_DDD, onehot_DDU, onehot_DUD, onehot_DUU, "
    "onehot_UDD, onehot_UDU, onehot_UUD, onehot_UUU, "
    "sinθ1*, cosθ1*, sinθ2*, cosθ2*, sinθ3*, cosθ3*"
)

GOAL_OBS_LOW = [0.0] * NUM_GOALS + [-1.0] * GOAL_SINCCOS_DIM
GOAL_OBS_HIGH = [1.0] * NUM_GOALS + [1.0] * GOAL_SINCCOS_DIM


def goal_angles(goal_ids: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """Map integer goal ids [...,] -> target angles [..., 3]."""
    table = GOAL_ANGLES.to(device=device or goal_ids.device, dtype=dtype or torch.float32)
    return table[goal_ids.long()]


def goal_encoding(goal_ids: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    """One-hot + target sin/cos. goal_ids: [...] -> [..., 14]."""
    device = device or goal_ids.device
    dtype = dtype or torch.float32
    ids = goal_ids.long()
    onehot = torch.nn.functional.one_hot(ids, NUM_GOALS).to(dtype=dtype)
    angles = goal_angles(ids, device=device, dtype=dtype)
    th1 = angles[..., 0]
    th2 = angles[..., 1]
    th3 = angles[..., 2]
    sincos = torch.stack(
        (
            torch.sin(th1),
            torch.cos(th1),
            torch.sin(th2),
            torch.cos(th2),
            torch.sin(th3),
            torch.cos(th3),
        ),
        dim=-1,
    )
    return torch.cat((onehot, sincos), dim=-1)


def sample_goals(n: int, device, dtype=torch.long, allowed=None, probs=None) -> torch.Tensor:
    """Sample goal ids (same API as goals.py)."""
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


def mean_cos_align(state: torch.Tensor, goal_ids: torch.Tensor) -> torch.Tensor:
    """Mean cos-align of three links vs goal ∈ [-1, 1]."""
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    a1 = angle_align(state[..., 2], angles[..., 0])
    a2 = angle_align(state[..., 4], angles[..., 1])
    a3 = angle_align(state[..., 6], angles[..., 2])
    return (a1 + a2 + a3) / 3.0


def nearest_goal(state: torch.Tensor) -> torch.Tensor:
    """Integer goal id of the equilibrium nearest to current angles."""
    th1 = state[..., 2]
    th2 = state[..., 4]
    th3 = state[..., 6]
    table = GOAL_ANGLES.to(device=state.device, dtype=state.dtype)
    a1 = angle_align(th1.unsqueeze(-1), table[:, 0])
    a2 = angle_align(th2.unsqueeze(-1), table[:, 1])
    a3 = angle_align(th3.unsqueeze(-1), table[:, 2])
    return (a1 + a2 + a3).argmax(dim=-1)


def at_goal(state: torch.Tensor, goal_ids: torch.Tensor, cos_thresh: float = 0.95) -> torch.Tensor:
    """Sparse success: all three links aligned with the requested equilibrium."""
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    a1 = angle_align(state[..., 2], angles[..., 0])
    a2 = angle_align(state[..., 4], angles[..., 1])
    a3 = angle_align(state[..., 6], angles[..., 2])
    return (a1 > cos_thresh) & (a2 > cos_thresh) & (a3 > cos_thresh)


def _link_is_up(th_star: torch.Tensor) -> torch.Tensor:
    """True when target is upright (near 0), False when hanging (near ±π)."""
    c = torch.cos(th_star)
    return c > 0.0


def product_reward(
    state,
    force,
    goal_ids,
    constants,
    track_limit: float = 4.0,
    oob_penalty: float = 20.0,
    # Lim KIEE 2025 soft coeffs (prefer over harsh MDPI-double)
    ru_coef: float = 0.001,
    ry_coef: float = 0.3,
    ry_scale: float = None,
    rw1_coef: float = 0.015,
    rw2_coef: float = 0.009,
    rw3_coef: float = 0.005,
    # Baek-style α floors so one bad link does not zero the product
    alpha_u: float = 0.0,
    alpha_y: float = 0.0,
    alpha_th: float = 0.0,
    alpha_w: float = 0.0,
    # Adaptive UP×5 / DOWN×1 geometric weights on angle terms
    w_up: float = 5.0,
    w_down: float = 1.0,
    # Cart barrier (fawraw): (x/limit)^8 penalty
    cart_barrier_coef: float = 50.0,
    reward_clip: float = 0.0,
    sparse_bonus: float = 0.0,
    # Optional additive height shaping (product otherwise ignores energy_w)
    energy_w: float = 0.0,
    # fawraw/Baek progress: dense bonus for Δ mean cos-align toward goal
    progress_w: float = 0.0,
    prev_state=None,
    **_ignored,
):
    """Lim-style product of [0,1] terms on absolute/world angles + rates.

    R = R_u * R_y * R_θ1 * R_θ2 * R_θ3 * R_ω1 * R_ω2 * R_ω3
    with optional Baek α floors and UP/DOWN geometric angle weights.
    Cart barrier subtracted after the product (anti rail-slide).
    Optional progress_w * (align_now − align_prev) when prev_state is given.
    """
    x = state[..., 0]
    th1 = state[..., 2]
    th1d = state[..., 3]
    th2 = state[..., 4]
    th2d = state[..., 5]
    th3 = state[..., 6]
    th3d = state[..., 7]
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    fl = float(constants["forceLimit"])
    y_scale = float(ry_scale) if ry_scale is not None else float(track_limit)

    # [0,1] factors (Lim)
    r_u = torch.exp(-float(ru_coef) * (force ** 2))
    r_y = torch.exp(-float(ry_coef) * (x.abs() / max(y_scale, 1e-6)))
    # World/absolute angles (our plant stores per-link absolute θ)
    r_th1 = 0.5 + 0.5 * angle_align(th1, angles[..., 0])
    r_th2 = 0.5 + 0.5 * angle_align(th2, angles[..., 1])
    r_th3 = 0.5 + 0.5 * angle_align(th3, angles[..., 2])
    # World rates (per-link absolute ω)
    r_w1 = torch.exp(-float(rw1_coef) * th1d.abs())
    r_w2 = torch.exp(-float(rw2_coef) * th2d.abs())
    r_w3 = torch.exp(-float(rw3_coef) * th3d.abs())

    # Baek floors: α + (1-α) * term
    def floor(term, alpha):
        a = float(alpha)
        if a <= 0.0:
            return term
        a = min(max(a, 0.0), 1.0)
        return a + (1.0 - a) * term

    r_u = floor(r_u, alpha_u)
    r_y = floor(r_y, alpha_y)
    r_th1 = floor(r_th1, alpha_th)
    r_th2 = floor(r_th2, alpha_th)
    r_th3 = floor(r_th3, alpha_th)
    r_w1 = floor(r_w1, alpha_w)
    r_w2 = floor(r_w2, alpha_w)
    r_w3 = floor(r_w3, alpha_w)

    # Adaptive UP/DOWN geometric weights on angle terms
    up1 = _link_is_up(angles[..., 0])
    up2 = _link_is_up(angles[..., 1])
    up3 = _link_is_up(angles[..., 2])
    w1 = torch.where(up1, torch.full_like(r_th1, float(w_up)), torch.full_like(r_th1, float(w_down)))
    w2 = torch.where(up2, torch.full_like(r_th2, float(w_up)), torch.full_like(r_th2, float(w_down)))
    w3 = torch.where(up3, torch.full_like(r_th3, float(w_up)), torch.full_like(r_th3, float(w_down)))
    wsum = (w1 + w2 + w3).clamp_min(1e-6)
    # Weighted geometric mean of angle terms, then multiply other factors
    # prod(r_i^{w_i})^{1/W} keeps scale in [0,1]
    log_th = (
        w1 * torch.log(r_th1.clamp_min(1e-8))
        + w2 * torch.log(r_th2.clamp_min(1e-8))
        + w3 * torch.log(r_th3.clamp_min(1e-8))
    ) / wsum
    r_th = torch.exp(log_th)

    rew = r_u * r_y * r_th * r_w1 * r_w2 * r_w3

    if sparse_bonus and sparse_bonus > 0:
        rew = rew + float(sparse_bonus) * at_goal(state, goal_ids).to(state.dtype)

    # Optional potential-height bonus toward goal (helps UUU swing-up under product).
    # Uses mean angle-align to target ∈ [-1,1]; scale by energy_w (default 0 = off).
    if energy_w and float(energy_w) > 0:
        a1 = angle_align(th1, angles[..., 0])
        a2 = angle_align(th2, angles[..., 1])
        a3 = angle_align(th3, angles[..., 2])
        height = (a1 + a2 + a3) / 3.0
        rew = rew + float(energy_w) * height

    # Progress shaping (fawraw): dense Δ cos-align toward goal.
    if progress_w and float(progress_w) > 0 and prev_state is not None:
        align_now = mean_cos_align(state, goal_ids)
        align_prev = mean_cos_align(prev_state, goal_ids)
        rew = rew + float(progress_w) * (align_now - align_prev)

    # Cart barrier (fawraw): strong near-rail penalty
    if cart_barrier_coef and cart_barrier_coef > 0 and track_limit > 0:
        frac = (x.abs() / float(track_limit)).clamp(max=1.0)
        rew = rew - float(cart_barrier_coef) * frac.pow(8)

    if reward_clip is not None and reward_clip > 0:
        rew = rew.clamp(-float(reward_clip), float(reward_clip))

    oob = (x.abs() > track_limit).to(state.dtype)
    rew = rew - float(oob_penalty) * oob
    return rew


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
    w_up: float = 1.0,
    w_down: float = 1.0,
    cart_barrier_coef: float = 0.0,
    vel_cost_coef: float = 0.0,
    progress_w: float = 0.0,
    prev_state=None,
    **_ignored,
):
    """Dense additive goal reward (legacy double-style; align over 3 links).

    Optional adaptive UP/DOWN angle weights and cart barrier for anti-DDD /
    anti-rail experiments without switching to product mode.
    """
    x = state[..., 0]
    xd = state[..., 1]
    th1 = state[..., 2]
    th1d = state[..., 3]
    th2 = state[..., 4]
    th2d = state[..., 5]
    th3 = state[..., 6]
    th3d = state[..., 7]
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    a1 = angle_align(th1, angles[..., 0])
    a2 = angle_align(th2, angles[..., 1])
    a3 = angle_align(th3, angles[..., 2])
    up1 = _link_is_up(angles[..., 0])
    up2 = _link_is_up(angles[..., 1])
    up3 = _link_is_up(angles[..., 2])
    w1 = torch.where(up1, torch.full_like(a1, float(w_up)), torch.full_like(a1, float(w_down)))
    w2 = torch.where(up2, torch.full_like(a2, float(w_up)), torch.full_like(a2, float(w_down)))
    w3 = torch.where(up3, torch.full_like(a3, float(w_up)), torch.full_like(a3, float(w_down)))
    wsum = (w1 + w2 + w3).clamp_min(1e-6)
    align = (w1 * a1 + w2 * a2 + w3 * a3) * (3.0 / wsum)  # keep ~[-3,3] scale

    x_eff = x.clamp(-track_limit, track_limit)
    xd_eff = xd.clamp(-20.0, 20.0)
    hold = (a1.clamp(min=0.0) + a2.clamp(min=0.0) + a3.clamp(min=0.0)) / 3.0
    hold = hold.clamp(0.0, 1.0)
    center_coef = center_w + center_hold_w * hold
    center = center_coef * (x_eff * x_eff + 0.2 * xd_eff * xd_eff)
    oob = (x.abs() > track_limit).to(state.dtype)

    spin_raw = th1d * th1d + th2d * th2d + th3d * th3d
    spin = spin_w * spin_raw.clamp(max=300.0)
    if vel_cost_coef and vel_cost_coef > 0:
        spin = spin + float(vel_cost_coef) * spin_raw.clamp(max=300.0)

    effort = 0.001 * (force / constants["forceLimit"]) ** 2
    sparse = sparse_bonus * at_goal(state, goal_ids).to(state.dtype)

    kin_soft = (0.05 * xd_eff * xd_eff + 0.02 * spin_raw).clamp(max=60.0)
    energy = energy_w * (align - 0.25 * kin_soft)

    rew = align_w * align + energy - center - spin - effort + sparse
    if progress_w and float(progress_w) > 0 and prev_state is not None:
        align_now = mean_cos_align(state, goal_ids)
        align_prev = mean_cos_align(prev_state, goal_ids)
        rew = rew + float(progress_w) * (align_now - align_prev)
    if cart_barrier_coef and cart_barrier_coef > 0 and track_limit > 0:
        frac = (x.abs() / float(track_limit)).clamp(max=1.0)
        rew = rew - float(cart_barrier_coef) * frac.pow(8)
    if reward_clip is not None and reward_clip > 0:
        rew = rew.clamp(-reward_clip, reward_clip)
    rew = rew - float(oob_penalty) * oob
    return rew


def compute_reward(state, force, goal_ids, constants, reward_mode: str = "product", **kwargs):
    """Dispatch product (Lim) or additive (legacy) reward."""
    mode = (reward_mode or "product").lower()
    if mode == "product":
        return product_reward(state, force, goal_ids, constants, **kwargs)
    return goal_reward(state, force, goal_ids, constants, **kwargs)


def conditioned_obs(state_obs: torch.Tensor, goal_ids: torch.Tensor) -> torch.Tensor:
    """Concatenate state features with goal encoding -> [..., 25]."""
    return torch.cat(
        (state_obs, goal_encoding(goal_ids, device=state_obs.device, dtype=state_obs.dtype)),
        dim=-1,
    )
