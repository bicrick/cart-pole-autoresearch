#!/usr/bin/env python3
"""Batched goal-conditioned PPO for cart + double pendulum. Smoke on CPU, scale on GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from goals import (
    GOAL_IDS,
    GOAL_INDEX,
    NUM_GOALS,
    OBS_DIM,
    angle_align,
    at_goal,
    conditioned_obs,
    goal_angles,
    goal_reward,
    nearest_goal,
    sample_goals,
)
from physics import _MAX_ANG_VEL, _MAX_CART_VEL, load_constants, normalize_obs, observe, step
from ppo import ActorCritic, export_actor, tanh_action

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policies" / "policy.json"
WEB_POLICY_PATH = ROOT / "web" / "public" / "policy.json"


def _cpu_percent_self() -> float:
    """Rough process CPU% over a short sleep using /proc (no psutil)."""
    try:
        with open("/proc/self/stat", "r", encoding="utf-8") as f:
            parts = f.read().split()
        ut1, st1 = int(parts[13]), int(parts[14])
        with open("/proc/stat", "r", encoding="utf-8") as f:
            total1 = sum(int(x) for x in f.readline().split()[1:])
        time.sleep(0.05)
        with open("/proc/self/stat", "r", encoding="utf-8") as f:
            parts = f.read().split()
        ut2, st2 = int(parts[13]), int(parts[14])
        with open("/proc/stat", "r", encoding="utf-8") as f:
            total2 = sum(int(x) for x in f.readline().split()[1:])
        d_proc = (ut2 + st2) - (ut1 + st1)
        d_tot = max(1, total2 - total1)
        ncpu = os.cpu_count() or 1
        return 100.0 * d_proc / d_tot * ncpu
    except Exception:
        return float("nan")


def _gpu_stats():
    """Return (util_percent, mem_used_mb, mem_total_mb) or NaNs."""
    if not torch.cuda.is_available():
        return float("nan"), float("nan"), float("nan")
    mem_used = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip().split(",")
        util = float(out[0].strip())
        used = float(out[1].strip())
        total = float(out[2].strip())
        return util, used, total
    except Exception:
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 2)
        return float("nan"), max(mem_used, mem_reserved), total



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent", type=float, default=0.01)
    parser.add_argument("--episode-len", type=int, default=800)
    parser.add_argument("--impulse-p", type=float, default=0.005)
    parser.add_argument(
        "--her-ratio",
        type=float,
        default=0.3,
        help="Fraction of failed episodes to relabel with nearest achieved equilibrium",
    )
    parser.add_argument(
        "--track-limit",
        type=float,
        default=4.0,
        help="|x| beyond this ends the episode (Duan/Gustafsson-style track fail)",
    )
    parser.add_argument("--reward-clip", type=float, default=8.0)
    parser.add_argument("--align-w", type=float, default=1.5)
    parser.add_argument("--energy-w", type=float, default=0.15)
    parser.add_argument("--spin-w", type=float, default=0.0003)
    parser.add_argument("--center-w", type=float, default=0.06,
                        help="Always-on |x| centering weight")
    parser.add_argument("--center-hold-w", type=float, default=0.18,
                        help="Extra |x| weight when links are aligned (kill rail-parking)")
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=80,
        help="First N updates train only --warmup-goal (0 disables)",
    )
    parser.add_argument(
        "--warmup-goal",
        default="UU",
        choices=list(GOAL_IDS),
        help="Single goal during warmup curriculum",
    )
    parser.add_argument(
        "--near-goal-p",
        type=float,
        default=0.5,
        help="Probability a reset spawns angles near the assigned goal",
    )
    parser.add_argument(
        "--hang-start-p",
        type=float,
        default=0.0,
        help="Probability a reset spawns near hanging (DD) — swing-up / energy-pump edge",
    )
    parser.add_argument(
        "--wrong-eq-p",
        type=float,
        default=0.0,
        help="Probability a reset spawns near a different discrete equilibrium than the goal",
    )
    parser.add_argument(
        "--goal-switch-p",
        type=float,
        default=0.0,
        help="Per-step P(change goal mid-episode without reset) — interactive toggle edge",
    )
    parser.add_argument(
        "--uu-bias",
        type=float,
        default=0.0,
        help="After hard warmup, P(warmup-goal) for anneal window (0=uniform). Soft curriculum.",
    )
    parser.add_argument(
        "--anneal-updates",
        type=int,
        default=0,
        help="Post-warmup updates keeping --uu-bias (0 with uu-bias>0 => rest of run)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "policies" / "checkpoint.pt")
    parser.add_argument("--out", type=Path, default=POLICY_PATH)
    parser.add_argument("--logdir", type=Path, default=ROOT / "runs")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human slug for the TensorBoard folder (after the timestamp)",
    )
    return parser.parse_args()


def default_run_name(args) -> str:
    """Descriptive TensorBoard run slug from key knobs."""
    parts = []
    ckpt = getattr(args, "checkpoint", None)
    parts.append("ft" if ckpt and Path(ckpt).is_file() else "cold")
    parts.append(f"e{args.num_envs}")
    parts.append(f"r{args.rollout}")
    parts.append("hardwalls")
    if getattr(args, "center_hold_w", 0) and args.center_hold_w > 0:
        parts.append("center")
        if args.center_hold_w != 0.18:
            parts.append(f"ch{args.center_hold_w:g}".replace(".", ""))
    if getattr(args, "uu_bias", 0):
        parts.append(f"uub{args.uu_bias:g}".replace(".", "p"))
    her = getattr(args, "her_ratio", None)
    if her is not None:
        parts.append(f"her{her:g}".replace(".", ""))
    if getattr(args, "hang_start_p", 0):
        parts.append(f"hang{args.hang_start_p:g}".replace(".", ""))
    if getattr(args, "wrong_eq_p", 0):
        parts.append(f"xeq{args.wrong_eq_p:g}".replace(".", ""))
    if getattr(args, "goal_switch_p", 0):
        parts.append(f"gsw{args.goal_switch_p:g}".replace(".", "p"))
    if getattr(args, "warmup_updates", 0):
        parts.append(f"wu{args.warmup_updates}{args.warmup_goal}")
    return "-".join(parts)


def reward_kwargs(args):
    return dict(
        align_w=args.align_w,
        energy_w=args.energy_w,
        spin_w=args.spin_w,
        center_w=args.center_w,
        center_hold_w=args.center_hold_w,
        track_limit=args.track_limit,
        reward_clip=args.reward_clip,
    )


def random_states(
    n,
    device,
    constants,
    mild=False,
    goals=None,
    near_goal_p=0.0,
    hang_start_p=0.0,
    wrong_eq_p=0.0,
):
    """Reset distribution.

    Priority on each env (exclusive): near-goal capture → hang (DD-ish swing-up)
    → wrong discrete equilibrium (anywhere↔anywhere edge) → uniform / mild.
    """
    x = torch.empty(n, device=device).uniform_(-2.0, 2.0)
    xd = torch.empty(n, device=device).uniform_(-3.0, 3.0)
    th1 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th2 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th1d = torch.empty(n, device=device).uniform_(-8.0, 8.0)
    th2d = torch.empty(n, device=device).uniform_(-8.0, 8.0)
    if mild:
        k = max(1, n // 5)
        x[:k] = torch.empty(k, device=device).uniform_(-0.4, 0.4)
        xd[:k] = torch.empty(k, device=device).uniform_(-0.4, 0.4)
        th1[:k] = torch.empty(k, device=device).uniform_(-0.25, 0.25)
        th2[:k] = torch.empty(k, device=device).uniform_(-0.25, 0.25)
        th1d[:k] = torch.empty(k, device=device).uniform_(-0.5, 0.5)
        th2d[:k] = torch.empty(k, device=device).uniform_(-0.5, 0.5)

    claimed = torch.zeros(n, dtype=torch.bool, device=device)

    # Local capture near the requested goal.
    if goals is not None and near_goal_p > 0:
        hit = (torch.rand(n, device=device) < near_goal_p) & ~claimed
        if hit.any():
            angles = goal_angles(goals, device=device, dtype=torch.float32)
            n_hit = int(hit.sum())
            x[hit] = torch.empty(n_hit, device=device).uniform_(-0.5, 0.5)
            xd[hit] = torch.empty(n_hit, device=device).uniform_(-0.5, 0.5)
            th1[hit] = angles[hit, 0] + torch.empty(n_hit, device=device).uniform_(-0.3, 0.3)
            th2[hit] = angles[hit, 1] + torch.empty(n_hit, device=device).uniform_(-0.3, 0.3)
            th1d[hit] = torch.empty(n_hit, device=device).uniform_(-0.8, 0.8)
            th2d[hit] = torch.empty(n_hit, device=device).uniform_(-0.8, 0.8)
            claimed |= hit

    # Hang / near-DD: forces energy pumping toward whatever goal is assigned.
    if hang_start_p > 0:
        hit = (torch.rand(n, device=device) < hang_start_p) & ~claimed
        if hit.any():
            n_hit = int(hit.sum())
            x[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            xd[hit] = torch.empty(n_hit, device=device).uniform_(-0.8, 0.8)
            # θ=0 upright; hang is ±π. Small noise so it's not a singular rest.
            th1[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
            th2[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
            th1d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            th2d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            claimed |= hit

    # Start at a *different* discrete equilibrium than the goal (switch / recover edge).
    if goals is not None and wrong_eq_p > 0:
        hit = (torch.rand(n, device=device) < wrong_eq_p) & ~claimed
        if hit.any():
            n_hit = int(hit.sum())
            other = torch.randint(0, 4, (n,), device=device)
            # Reroll collisions with the assigned goal.
            same = other == goals
            if same.any():
                other = torch.where(same, (other + 1 + torch.randint(0, 3, (n,), device=device)) % 4, other)
            angles = goal_angles(other, device=device, dtype=torch.float32)
            x[hit] = torch.empty(n_hit, device=device).uniform_(-0.6, 0.6)
            xd[hit] = torch.empty(n_hit, device=device).uniform_(-0.6, 0.6)
            th1[hit] = angles[hit, 0] + torch.empty(n_hit, device=device).uniform_(-0.25, 0.25)
            th2[hit] = angles[hit, 1] + torch.empty(n_hit, device=device).uniform_(-0.25, 0.25)
            th1d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            th2d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)

    return torch.stack((x, xd, th1, th1d, th2, th2d), dim=-1)


def apply_impulses(state, p):
    n = state.shape[0]
    hit = torch.rand(n, device=state.device) < p
    if not hit.any():
        return state
    kick = state.clone()
    kick[:, 1] += torch.empty(n, device=state.device).uniform_(-4.0, 4.0)
    kick[:, 3] += torch.empty(n, device=state.device).uniform_(-6.0, 6.0)
    kick[:, 5] += torch.empty(n, device=state.device).uniform_(-6.0, 6.0)
    # Same soft caps as physics.step so a kick cannot leave unbounded velocities.
    kick[:, 1].clamp_(-_MAX_CART_VEL, _MAX_CART_VEL)
    kick[:, 3].clamp_(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    kick[:, 5].clamp_(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    return torch.where(hit.unsqueeze(-1), kick, state)


def make_obs(state, goal_ids, constants):
    return normalize_obs(conditioned_obs(observe(state), goal_ids), constants)


def save_policy(model, constants, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_actor(model, constants)
    path.write_text(json.dumps(payload))
    WEB_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_POLICY_PATH.write_text(json.dumps(payload))


def final_states_for_her(state_t, done_t):
    """Prefer state at last done in the window; else last rollout state."""
    t_len, n_envs, _ = state_t.shape
    device = state_t.device
    final_state = state_t[-1]
    any_done = done_t.any(dim=0)
    if not any_done.any():
        return final_state
    idxs = torch.arange(t_len, device=device).unsqueeze(1).expand_as(done_t)
    masked = torch.where(done_t.bool(), idxs, torch.full_like(idxs, -1))
    last_done = masked.max(dim=0).values.clamp(min=0)
    gather_idx = last_done.view(1, n_envs, 1).expand(1, n_envs, state_t.shape[-1])
    final_from_done = state_t.gather(0, gather_idx).squeeze(0)
    return torch.where(any_done.unsqueeze(-1), final_from_done, final_state)


@torch.no_grad()
def her_relabel_inplace(
    obs_t, rew_t, val_t, state_t, force_t, goal_t, done_t, model, constants, her_ratio, rkw
):
    """
    For a random subset of failed envs, replace the goal with the nearest
    equilibrium at episode/rollout end (HER). Rewrite goal channels in obs,
    recompute rewards, refresh critic values. Actions stay put so PPO ratios
    remain defined; goal channels change — approximate but teaches other goals.
    """
    if her_ratio <= 0:
        return obs_t, rew_t, goal_t, val_t

    t_len, n_envs, _ = state_t.shape
    device = state_t.device
    final_state = final_states_for_her(state_t, done_t)
    achieved = nearest_goal(final_state)
    requested = goal_t[-1]
    succeeded = at_goal(final_state, requested)
    relabel = (torch.rand(n_envs, device=device) < her_ratio) & (~succeeded) & (achieved != requested)
    if not relabel.any():
        return obs_t, rew_t, goal_t, val_t

    new_goals = torch.where(relabel, achieved, requested)
    goals_full = goal_t.clone()
    goals_full[:, relabel] = new_goals[relabel].unsqueeze(0).expand(t_len, -1)

    rel_states = state_t[:, relabel]
    rel_forces = force_t[:, relabel]
    rel_goals = goals_full[:, relabel]
    t_h, n_h = rel_states.shape[0], rel_states.shape[1]
    flat_obs = normalize_obs(
        conditioned_obs(observe(rel_states.reshape(-1, 6)), rel_goals.reshape(-1)),
        constants,
    ).reshape(t_h, n_h, OBS_DIM)
    flat_rew = goal_reward(
        rel_states.reshape(-1, 6),
        rel_forces.reshape(-1),
        rel_goals.reshape(-1),
        constants,
        **rkw,
    ).reshape(t_h, n_h)
    flat_val = model.critic(flat_obs.reshape(-1, OBS_DIM)).squeeze(-1).reshape(t_h, n_h)

    obs_t = obs_t.clone()
    rew_t = rew_t.clone()
    val_t = val_t.clone()
    obs_t[:, relabel] = flat_obs
    rew_t[:, relabel] = flat_rew
    val_t[:, relabel] = flat_val
    return obs_t, rew_t, goals_full, val_t


@torch.no_grad()
def rollout_eval(model, constants, device, rkw, n=64, steps=400, track_limit=4.0):
    """Evaluate each discrete goal separately; return overall + per-goal metrics."""
    model.eval()
    per_goal = {}
    all_rew = []
    all_align = []
    for gid, name in enumerate(GOAL_IDS):
        state = random_states(n, device, constants, mild=False)
        goals = torch.full((n,), gid, device=device, dtype=torch.long)
        total = torch.zeros(n, device=device)
        align_sum = torch.zeros(n, device=device)
        hits = torch.zeros(n, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        for _ in range(steps):
            obs = make_obs(state, goals, constants)
            force = tanh_action(model.deterministic(obs), constants["forceLimit"]).squeeze(-1)
            state = step(state, force, constants=constants)
            rew = goal_reward(state, force, goals, constants, **rkw)
            # Stop accruing after track fail so eval reward stays interpretable.
            total += torch.where(alive, rew, torch.zeros_like(rew))
            angles = goal_angles(goals, device=device, dtype=state.dtype)
            align = 0.5 * (
                angle_align(state[:, 2], angles[:, 0]) + angle_align(state[:, 4], angles[:, 1])
            )
            align_sum += torch.where(alive, align, torch.zeros_like(align))
            hits += torch.where(alive, at_goal(state, goals).float(), torch.zeros(n, device=device))
            alive = alive & (state[:, 0].abs() <= track_limit)
        mean_rew = float(total.mean())
        mean_align = float(align_sum.mean() / steps)
        mean_hit = float(hits.mean() / steps)
        per_goal[name] = {"reward": mean_rew, "align": mean_align, "at_goal": mean_hit}
        all_rew.append(mean_rew)
        all_align.append(mean_align)
    model.train()
    return sum(all_rew) / NUM_GOALS, sum(all_align) / NUM_GOALS, per_goal


def allowed_goals_for_update(args, update: int):
    """Legacy hard allow-list (pure warmup). Prefer goal_probs_for_update."""
    if args.warmup_updates > 0 and update <= args.warmup_updates:
        return [GOAL_INDEX[args.warmup_goal]]
    return None


def goal_probs_for_update(args, update: int):
    """Curriculum mix: hard UU warmup, then optional soft UU bias, then uniform.

    Returns (probs_tensor_or_None, allowed_list_or_None). When probs is set,
    sample_goals uses multinomial; when only allowed is set, uniform over that
    subset; both None => uniform over all four goals.
    """
    n = NUM_GOALS
    warm = int(args.warmup_updates)
    bias = float(args.uu_bias)
    anneal = int(args.anneal_updates)
    gid = GOAL_INDEX[args.warmup_goal]
    if warm > 0 and update <= warm:
        return None, [gid]
    # Soft anneal window after hard cut (Gustafsson: keep local capture mass).
    if bias > 0.0:
        end = warm + anneal if anneal > 0 else args.updates
        if update <= end:
            bias = min(max(bias, 0.0), 1.0)
            rest = (1.0 - bias) / max(1, n - 1)
            probs = [rest] * n
            probs[gid] = bias
            return probs, None
    return None, None


def main():
    args = parse_args()
    constants = load_constants()
    if args.smoke:
        args.num_envs = min(args.num_envs, 32)
        args.rollout = 32
        args.updates = 2
        args.minibatch = 64
        args.episode_len = 64
        args.warmup_updates = 0
        args.uu_bias = 0.0
        args.anneal_updates = 0
    if args.device:
        device_name = args.device
    elif torch.cuda.is_available():
        device_name = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = (args.run_name or default_run_name(args)).strip().replace(" ", "-")
    logdir = args.logdir / f"{stamp}_{slug}"
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))
    rkw = reward_kwargs(args)
    print(
        f"device={device} envs={args.num_envs} rollout={args.rollout} "
        f"updates={args.updates} obs_dim={OBS_DIM} goals={GOAL_IDS} logdir={logdir}"
    )
    print(
        f"reward: align_w={args.align_w} energy_w={args.energy_w} spin_w={args.spin_w} center_w={args.center_w} center_hold_w={args.center_hold_w} "
        f"track_limit={args.track_limit} reward_clip={args.reward_clip} "
        f"warmup={args.warmup_updates}x{args.warmup_goal} uu_bias={args.uu_bias} "
        f"anneal={args.anneal_updates} near_goal_p={args.near_goal_p} "
        f"hang_start_p={args.hang_start_p} wrong_eq_p={args.wrong_eq_p} "
        f"goal_switch_p={args.goal_switch_p} "
        f"her_ratio={args.her_ratio}",
        flush=True,
    )
    print(f"tensorboard --logdir {args.logdir}", flush=True)

    model = ActorCritic(obs_dim=OBS_DIM, hidden=constants["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.checkpoint.is_file():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
        prev = ckpt.get("update") if isinstance(ckpt, dict) else None
        print(
            f"loaded checkpoint {args.checkpoint}"
            + (f" (saved at update={prev})" if prev is not None else ""),
            flush=True,
        )
    else:
        print(f"no checkpoint at {args.checkpoint}; cold start", flush=True)
    _probs, _allowed = goal_probs_for_update(args, 1)
    goals = sample_goals(args.num_envs, device, allowed=_allowed, probs=_probs)
    state = random_states(
        args.num_envs,
        device,
        constants,
        mild=True,
        goals=goals,
        near_goal_p=args.near_goal_p,
        hang_start_p=args.hang_start_p,
        wrong_eq_p=args.wrong_eq_p,
    )
    steps_left = torch.randint(1, args.episode_len + 1, (args.num_envs,), device=device)

    best_align = -1e9
    t0 = time.time()
    t_prev = time.perf_counter()
    for update in range(1, args.updates + 1):
        goal_probs, allowed = goal_probs_for_update(args, update)
        obs_buf = []
        raw_buf = []
        log_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []
        state_buf = []
        force_buf = []
        goal_buf = []

        for _ in range(args.rollout):
            obs = make_obs(state, goals, constants)
            with torch.no_grad():
                raw, log_prob, value = model.act(obs)
            force = tanh_action(raw, constants["forceLimit"]).squeeze(-1)
            state = apply_impulses(state, args.impulse_p)
            # Mid-episode goal flip (no state reset) — interactive picker edge case.
            if args.goal_switch_p > 0:
                flip = torch.rand(args.num_envs, device=device) < args.goal_switch_p
                if flip.any():
                    new_g = sample_goals(
                        args.num_envs, device, allowed=allowed, probs=goal_probs
                    )
                    # Avoid no-op flips when possible.
                    same = new_g == goals
                    if same.any():
                        new_g = torch.where(
                            same,
                            (new_g + 1 + torch.randint(0, 3, (args.num_envs,), device=device)) % 4,
                            new_g,
                        )
                    goals = torch.where(flip, new_g, goals)
            next_state = step(state, force, constants=constants)
            rew = goal_reward(next_state, force, goals, constants, **rkw)
            steps_left -= 1
            oob = next_state[:, 0].abs() > args.track_limit
            done = (steps_left <= 0) | oob
            # Buffer transition under the goal that produced the reward (pre-reset).
            goal_buf.append(goals.detach().clone())
            state_buf.append(next_state.detach())
            force_buf.append(force.detach())
            if done.any():
                new_goals = sample_goals(
                    args.num_envs, device, allowed=allowed, probs=goal_probs
                )
                reset_goals = torch.where(done, new_goals, goals)
                reset = random_states(
                    args.num_envs,
                    device,
                    constants,
                    mild=True,
                    goals=reset_goals,
                    near_goal_p=args.near_goal_p,
                    hang_start_p=args.hang_start_p,
                    wrong_eq_p=args.wrong_eq_p,
                )
                next_state = torch.where(done.unsqueeze(-1), reset, next_state)
                goals = reset_goals
                steps_left = torch.where(
                    done,
                    torch.full_like(steps_left, args.episode_len),
                    steps_left,
                )
            obs_buf.append(obs)
            raw_buf.append(raw.detach())
            log_buf.append(log_prob.detach())
            val_buf.append(value.detach())
            rew_buf.append(rew.detach())
            done_buf.append(done.float())
            state = next_state

        obs_t = torch.stack(obs_buf)
        raw_t = torch.stack(raw_buf)
        log_t = torch.stack(log_buf)
        val_t = torch.stack(val_buf)
        rew_t = torch.stack(rew_buf)
        done_t = torch.stack(done_buf)
        state_t = torch.stack(state_buf)
        force_t = torch.stack(force_buf)
        goal_t = torch.stack(goal_buf)

        obs_t, rew_t, goal_t, val_t = her_relabel_inplace(
            obs_t,
            rew_t,
            val_t,
            state_t,
            force_t,
            goal_t,
            done_t,
            model,
            constants,
            args.her_ratio,
            rkw,
        )

        with torch.no_grad():
            last_obs = make_obs(state, goals, constants)
            last_val = model.critic(last_obs).squeeze(-1)

        adv = torch.zeros_like(rew_t)
        last_adv = torch.zeros(args.num_envs, device=device)
        next_value = last_val
        for t in reversed(range(args.rollout)):
            mask = 1.0 - done_t[t]
            delta = rew_t[t] + args.gamma * next_value * mask - val_t[t]
            last_adv = delta + args.gamma * args.gae * mask * last_adv
            adv[t] = last_adv
            next_value = val_t[t]
        ret = adv + val_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_flat = obs_t.reshape(-1, OBS_DIM)
        raw_flat = raw_t.reshape(-1, 1)
        log_flat = log_t.reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)
        n = obs_flat.shape[0]
        mb = min(args.minibatch, n)

        last_policy = last_value = last_ent = last_loss = None
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, mb):
                idx = perm[start : start + mb]
                log_p, ent, value = model.evaluate(obs_flat[idx], raw_flat[idx])
                ratio = (log_p - log_flat[idx]).exp()
                surr1 = ratio * adv_flat[idx]
                surr2 = ratio.clamp(1 - args.clip, 1 + args.clip) * adv_flat[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (value - ret_flat[idx]).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss - args.ent * ent.mean()
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    # Skip poisoned minibatches (NaN reward / value) instead of
                    # writing NaNs into the policy weights.
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                last_policy = float(policy_loss.detach())
                last_value = float(value_loss.detach())
                last_ent = float(ent.mean().detach())
                last_loss = float(loss.detach())

        writer.add_scalar("train/rollout_reward", float(rew_t.mean()), update)
        writer.add_scalar("train/policy_loss", last_policy, update)
        writer.add_scalar("train/value_loss", last_value, update)
        writer.add_scalar("train/entropy", last_ent, update)
        writer.add_scalar("train/loss", last_loss, update)

        # Throughput / hardware efficiency (every update; GPU sample is cheap).
        t_now = time.perf_counter()
        dt = max(1e-6, t_now - t_prev)
        t_prev = t_now
        env_steps = float(args.num_envs * args.rollout)
        writer.add_scalar("perf/sec_per_update", dt, update)
        writer.add_scalar("perf/updates_per_sec", 1.0 / dt, update)
        writer.add_scalar("perf/env_steps_per_sec", env_steps / dt, update)
        writer.add_scalar("perf/samples_per_update", env_steps, update)
        writer.add_scalar("perf/num_envs", float(args.num_envs), update)
        writer.add_scalar("perf/rollout", float(args.rollout), update)
        if update == 1 or update % 5 == 0:
            gpu_util, gpu_used, gpu_total = _gpu_stats()
            writer.add_scalar("perf/gpu_util_percent", gpu_util, update)
            writer.add_scalar("perf/gpu_mem_used_mb", gpu_used, update)
            writer.add_scalar("perf/gpu_mem_total_mb", gpu_total, update)
            if torch.cuda.is_available():
                writer.add_scalar(
                    "perf/torch_cuda_allocated_mb",
                    torch.cuda.memory_allocated() / (1024 ** 2),
                    update,
                )
            writer.add_scalar("perf/cpu_percent", _cpu_percent_self(), update)
        writer.add_scalar(
            "train/warmup_active",
            float(allowed is not None or goal_probs is not None),
            update,
        )
        if goal_probs is not None:
            writer.add_scalar(
                f"train/goal_target_frac/{args.warmup_goal}",
                float(goal_probs[GOAL_INDEX[args.warmup_goal]]),
                update,
            )
        for gid, name in enumerate(GOAL_IDS):
            frac = float((goal_t[-1] == gid).float().mean())
            writer.add_scalar(f"train/goal_frac/{name}", frac, update)

        if update == 1 or update % 10 == 0 or args.smoke:
            mean_rew, mean_align, per_goal = rollout_eval(
                model,
                constants,
                device,
                rkw,
                n=32 if args.smoke else 64,
                steps=200 if args.smoke else 400,
                track_limit=args.track_limit,
            )
            elapsed = time.time() - t0
            parts = " ".join(f"{name}={per_goal[name]['align']:.2f}" for name in GOAL_IDS)
            print(
                f"update={update:04d} reward={mean_rew:.3f} align={mean_align:.3f} "
                f"[{parts}] sec={elapsed:.1f}",
                flush=True,
            )
            writer.add_scalar("eval/reward", mean_rew, update)
            writer.add_scalar("eval/align", mean_align, update)
            writer.add_scalar("eval/upright", per_goal["UU"]["align"], update)
            for name, metrics in per_goal.items():
                writer.add_scalar(f"eval/reward/{name}", metrics["reward"], update)
                writer.add_scalar(f"eval/align/{name}", metrics["align"], update)
                writer.add_scalar(f"eval/at_goal/{name}", metrics["at_goal"], update)
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "update": update, "obs_dim": OBS_DIM},
                args.checkpoint,
            )
            if mean_align >= best_align:
                best_align = mean_align
                save_policy(model, constants, args.out)

    save_policy(model, constants, args.out)
    writer.flush()
    writer.close()
    print(f"wrote {args.out} and {WEB_POLICY_PATH}")
    print(f"tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
