#!/usr/bin/env python3
"""Batched goal-conditioned PPO for cart + triple pendulum. Smoke on CPU, scale on GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from goals_triple import (
    GOAL_IDS,
    GOAL_INDEX,
    NUM_GOALS,
    OBS_DIM,
    OBS_LAYOUT,
    angle_align,
    at_goal,
    compute_reward,
    conditioned_obs,
    goal_angles,
    goal_reward,
    mean_cos_align,
    nearest_goal,
    sample_goals,
)
from physics_triple import _MAX_ANG_VEL, _MAX_CART_VEL, load_constants, normalize_obs, observe, step
from ppo import ActorCritic, tanh_action

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policies" / "policy-triple.json"
WEB_POLICY_PATH = None  # do not overwrite double web demo


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
    parser.add_argument(
        "--avg-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="ATRPO-lite (Zhang–Ross 2106.07329 Alg.2/App.G): ρ-center rewards + "
        "γ-free GAE (λ=--gae). Keep EPISODE_LEN≥1200. Pair with --ent 0.",
    )
    parser.add_argument(
        "--avc-nu",
        type=float,
        default=0.0,
        help="APO Average Value Constraint (2106.03442 §4.1): after GAE returns, "
        "subtract ν·bias from ret so E[V]≈0 under γ-free. 0=off; try 0.1–0.3 "
        "with --avg-reward. Logs train/avc_bias.",
    )
    parser.add_argument(
        "--avc-ema-alpha",
        type=float,
        default=0.0,
        help="APO Alg.1 fidelity (2106.03442): EMA α for η̂ (ρ) and V-bias. "
        "0=cheap one-shot mean(ret) AVC-lite; try 0.1 for EMA-η̂ + EMA-V. "
        "Only meaningful with --avg-reward / --avc-nu.",
    )
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--ent", type=float, default=0.01)
    parser.add_argument(
        "--rpo-alpha",
        type=float,
        default=0.0,
        help="CleanRL RPO: Uniform(-α,α) jitter on actor mean at PPO update (0=off). Try 0.01 first.",
    )
    parser.add_argument(
        "--log-std-floor",
        type=float,
        default=0.0,
        help="ERA soft entropy floor H₀ for 1-D Normal (0=off). Softplus/detached; try 0.5–0.8. Not a hard clamp.",
    )
    parser.add_argument(
        "--reset-log-std",
        type=float,
        default=None,
        help="After checkpoint load, fill_ global log_std to this value and rebuild Adam "
        "(SB3 #155 curriculum resume). Omit=off. Use 0.0 when resuming a collapsed-σ H1 into RPO/ERA.",
    )
    parser.add_argument(
        "--use-sde",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Raffin/SB3 gSDE: state-dependent exploration from actor penultimate features "
        "(Zoo Pendulum-style). Pair with --ent 0; keeps RPO+ERA. Default off.",
    )
    parser.add_argument(
        "--sde-sample-freq",
        type=int,
        default=4,
        help="Resample gSDE exploration noise every N PPO updates (Zoo default 4).",
    )
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
    parser.add_argument(
        "--oob-penalty",
        type=float,
        default=20.0,
        help="Terminal void-death cost when |x|>track_limit (after reward_clip)",
    )
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
        default="UUU",
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
        help="Probability a reset spawns near hanging (DDD) — swing-up / energy-pump edge",
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
        "--fold-pair-p",
        type=float,
        default=0.0,
        help=(
            "Bias wrong-eq starts and mid-episode flips toward outer-link fold "
            "(flip tip link only: XOR 1 on goal id)"
        ),
    )
    parser.add_argument(
        "--transition-only",
        action="store_true",
        help=(
            "Every reset: spawn near discrete eq A with goal B≠A (uniform over directed "
            "pairs). Mid-episode flips always change goal. Ignores hang/near-goal/wrong-eq mixes."
        ),
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
    parser.add_argument(
        "--reward-mode",
        default="product",
        choices=("product", "additive"),
        help="product=Lim-style [0,1] product on world angles; additive=legacy align/energy",
    )
    parser.add_argument(
        "--cart-barrier-coef",
        type=float,
        default=50.0,
        help="(x/track_limit)^8 penalty (fawraw anti rail-slide); 0 disables",
    )
    parser.add_argument(
        "--w-up",
        type=float,
        default=5.0,
        help="Angle weight for UP-targeted links (adaptive anti-DDD)",
    )
    parser.add_argument(
        "--w-down",
        type=float,
        default=1.0,
        help="Angle weight for DOWN-targeted links",
    )
    parser.add_argument(
        "--alpha-u", type=float, default=0.0, help="Baek floor on R_u (product mode)"
    )
    parser.add_argument(
        "--alpha-y", type=float, default=0.0, help="Baek floor on R_y (product mode)"
    )
    parser.add_argument(
        "--alpha-th", type=float, default=0.5, help="Baek floor on R_θ (product; 0.5=Baek)"
    )
    parser.add_argument(
        "--alpha-w", type=float, default=0.0, help="Baek floor on R_ω (product mode)"
    )
    parser.add_argument(
        "--fall-grace-steps",
        type=int,
        default=20,
        help="First N steps of an episode: no oob/angle-fall terminate",
    )
    parser.add_argument(
        "--start-grace-steps",
        type=int,
        default=0,
        help="Extra immune steps at episode start (stacked with fall-grace for oob)",
    )
    parser.add_argument(
        "--angle-fall",
        action="store_true",
        help="Terminate when any link exceeds per-target fall threshold (UP=0.6/DOWN=1.5)",
    )
    parser.add_argument(
        "--fall-thresh-up", type=float, default=0.6, help="Angle-fall rad for UP targets"
    )
    parser.add_argument(
        "--fall-thresh-down", type=float, default=1.5, help="Angle-fall rad for DOWN targets"
    )
    parser.add_argument(
        "--init-mode",
        default="mixed",
        choices=("mixed", "bottom", "near_target", "wide"),
        help="Reset prior: mixed=curriculum mix; bottom=hang; near_target; wide=Lim-ish ICs",
    )
    parser.add_argument(
        "--init-noise",
        type=float,
        default=0.3,
        help=(
            "Near-target / near-goal angle half-range (rad). Also scales cart (×2) and "
            "ω (×5) proportionally with only 1e-3 absolute floors (quiet-basin; was "
            "cart≥0.05 / ω≥0.1 which nulled small INIT_NOISE). M2 hold uses 0.05."
        ),
    )
    parser.add_argument(
        "--warmup-hang-start-p",
        type=float,
        default=0.0,
        help="hang_start_p during UUU warmup (default 0 — anti-DDD)",
    )
    parser.add_argument(
        "--vel-cost-coef",
        type=float,
        default=0.0,
        help="Extra additive vel^2 cost (additive mode; prefer 0.01-0.02)",
    )
    parser.add_argument(
        "--force-limit",
        type=float,
        default=None,
        help="Override constants forceLimit (N); also env FORCE_LIMIT. Hard tanh action cap.",
    )
    parser.add_argument(
        "--track-walls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Hard inelastic cart endstops at ±trackLimit (clamp x, xd=0). "
            "Default: constants trackWalls (JSON false = void). "
            "Also env TRACK_WALLS=1/0. Walls prevent oob death — use for P1a curriculum."
        ),
    )
    parser.add_argument(
        "--progress-w",
        type=float,
        default=None,
        help=(
            "Dense progress bonus weight on Δ mean cos-align toward goal (fawraw). "
            "Default 1.0 when --reward-mode product, else 0. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--flip-augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Baek VER left–right mirror augment: after GAE, duplicate PPO batch with "
            "cart x/xd, angles/rates, and force/raw flipped (default on for triple). "
            "Symmetry: planar reflect across vertical midline; θ*=0/π goals invariant."
        ),
    )
    parser.add_argument(
        "--eval-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also eval from near_target and hang (bottom) starts; log "
            "eval/near_target/* and eval/hang/* separately from random-IC eval "
            "(default on)."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "policies" / "checkpoint-triple.pt")
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
    parts.append("triple")
    tw = getattr(args, "track_walls", None)
    parts.append("walls" if tw else "nowalls")
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
    if getattr(args, "fold_pair_p", 0):
        parts.append(f"fold{args.fold_pair_p:g}".replace(".", ""))
    if getattr(args, "transition_only", False):
        parts.append("txonly")
    if getattr(args, "warmup_updates", 0):
        parts.append(f"wu{args.warmup_updates}{args.warmup_goal}")
    mode = getattr(args, "reward_mode", "product")
    if mode:
        parts.append(mode[:3])
    if getattr(args, "cart_barrier_coef", 0):
        parts.append(f"bar{args.cart_barrier_coef:g}".replace(".", ""))
    init = getattr(args, "init_mode", "mixed")
    if init and init != "mixed":
        parts.append(init)
    noise = getattr(args, "init_noise", 0.3)
    if noise is not None and abs(float(noise) - 0.3) > 1e-9:
        parts.append(f"in{float(noise):g}".replace(".", ""))
    return "-".join(parts)


def reward_kwargs(args):
    pw = args.progress_w
    if pw is None:
        pw = 1.0 if (args.reward_mode or "product").lower() == "product" else 0.0
    args.progress_w = float(pw)
    return dict(
        reward_mode=args.reward_mode,
        align_w=args.align_w,
        energy_w=args.energy_w,
        spin_w=args.spin_w,
        center_w=args.center_w,
        center_hold_w=args.center_hold_w,
        track_limit=args.track_limit,
        reward_clip=args.reward_clip if args.reward_mode == "additive" else 0.0,
        oob_penalty=args.oob_penalty,
        cart_barrier_coef=args.cart_barrier_coef,
        w_up=args.w_up,
        w_down=args.w_down,
        alpha_u=args.alpha_u,
        alpha_y=args.alpha_y,
        alpha_th=args.alpha_th,
        alpha_w=args.alpha_w,
        vel_cost_coef=args.vel_cost_coef,
        progress_w=float(pw),
    )


def fold_partner(goal_ids: torch.Tensor) -> torch.Tensor:
    """Outer-link (tip) fold: XOR 1 on goal id (DDD↔DDU, …, UUD↔UUU)."""
    return goal_ids ^ 1


def random_states(
    n,
    device,
    constants,
    mild=False,
    goals=None,
    near_goal_p=0.0,
    hang_start_p=0.0,
    wrong_eq_p=0.0,
    fold_pair_p=0.0,
    transition_only=False,
    init_mode="mixed",
    init_noise=0.3,
):
    """Reset distribution.

    If transition_only and goals are set: every env spawns near a discrete eq
    different from its goal (uniform directed A→B, A≠B). Otherwise priority is
    near-goal → hang → wrong-eq → uniform / mild.

    init_mode:
      mixed       — curriculum mix (default)
      bottom      — all near hanging DDD (swing-up from bottom)
      near_target — near assigned goal; hang_start_p overlays hang ICs for swing-up
      wide        — Lim-style wide random ICs (full angle, larger rates)

    init_noise: angle (and cart-x) half-range for near_target / near-goal;
                rates xd/ω FIXED ±0.01 (fawraw quiet basin — not scaled by noise).
    """
    mode = (init_mode or "mixed").lower()

    # Wide / Lim-ish base ranges (anti-DDD: cover full angle space)
    if mode == "wide":
        x = torch.empty(n, device=device).uniform_(-0.3, 0.3)
        xd = torch.empty(n, device=device).uniform_(-1.2, 1.2)
        th1 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
        th2 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
        th3 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
        th1d = torch.empty(n, device=device).uniform_(-10.0, 10.0)
        th2d = torch.empty(n, device=device).uniform_(-20.0, 20.0)
        th3d = torch.empty(n, device=device).uniform_(-30.0, 30.0)
        return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)

    if mode == "bottom":
        x = torch.empty(n, device=device).uniform_(-1.0, 1.0)
        xd = torch.empty(n, device=device).uniform_(-0.8, 0.8)
        th1 = math.pi + torch.empty(n, device=device).uniform_(-0.35, 0.35)
        th2 = math.pi + torch.empty(n, device=device).uniform_(-0.35, 0.35)
        th3 = math.pi + torch.empty(n, device=device).uniform_(-0.35, 0.35)
        th1d = torch.empty(n, device=device).uniform_(-1.0, 1.0)
        th2d = torch.empty(n, device=device).uniform_(-1.0, 1.0)
        th3d = torch.empty(n, device=device).uniform_(-1.0, 1.0)
        return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)

    if mode == "near_target" and goals is not None:
        # Hold prior near assigned goal; optional hang_start_p overlays swing-up ICs.
        # Quiet-basin (fawraw M2): angles/cart ~ ±init_noise; rates xd/ω FIXED ±0.01
        # (NOT init_noise×5 / ×2 — that blew the true quiet basin and made LQR fail).
        n_ang = max(float(init_noise), 1e-3)
        n_x = n_ang
        n_rate = 0.01
        angles = goal_angles(goals, device=device, dtype=torch.float32)
        x = torch.empty(n, device=device).uniform_(-n_x, n_x)
        xd = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        th1 = angles[:, 0] + torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        th2 = angles[:, 1] + torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        th3 = angles[:, 2] + torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        th1d = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        th2d = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        th3d = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        if hang_start_p > 0:
            hit = torch.rand(n, device=device) < hang_start_p
            if hit.any():
                n_hit = int(hit.sum())
                x[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
                xd[hit] = torch.empty(n_hit, device=device).uniform_(-0.8, 0.8)
                th1[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
                th2[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
                th3[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
                th1d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
                th2d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
                th3d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
        return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)

    # mixed (default): wider base than old ±8 rad/s — closer to Lim coverage
    x = torch.empty(n, device=device).uniform_(-2.0, 2.0)
    xd = torch.empty(n, device=device).uniform_(-3.0, 3.0)
    th1 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th2 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th3 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th1d = torch.empty(n, device=device).uniform_(-10.0, 10.0)
    th2d = torch.empty(n, device=device).uniform_(-12.0, 12.0)
    th3d = torch.empty(n, device=device).uniform_(-14.0, 14.0)
    if mild:
        k = max(1, n // 5)
        x[:k] = torch.empty(k, device=device).uniform_(-0.4, 0.4)
        xd[:k] = torch.empty(k, device=device).uniform_(-0.4, 0.4)
        th1[:k] = torch.empty(k, device=device).uniform_(-0.25, 0.25)
        th2[:k] = torch.empty(k, device=device).uniform_(-0.25, 0.25)
        th3[:k] = torch.empty(k, device=device).uniform_(-0.25, 0.25)
        th1d[:k] = torch.empty(k, device=device).uniform_(-0.5, 0.5)
        th2d[:k] = torch.empty(k, device=device).uniform_(-0.5, 0.5)
        th3d[:k] = torch.empty(k, device=device).uniform_(-0.5, 0.5)

    claimed = torch.zeros(n, dtype=torch.bool, device=device)

    # Transition-only: always start at discrete eq A with goal B≠A.
    if transition_only and goals is not None:
        start = torch.randint(0, NUM_GOALS, (n,), device=device)
        same = start == goals
        if same.any():
            start = torch.where(
                same,
                (start + 1 + torch.randint(0, NUM_GOALS - 1, (n,), device=device)) % NUM_GOALS,
                start,
            )
        angles = goal_angles(start, device=device, dtype=torch.float32)
        x = torch.empty(n, device=device).uniform_(-0.6, 0.6)
        xd = torch.empty(n, device=device).uniform_(-0.8, 0.8)
        th1 = angles[:, 0] + torch.empty(n, device=device).uniform_(-0.3, 0.3)
        th2 = angles[:, 1] + torch.empty(n, device=device).uniform_(-0.3, 0.3)
        th3 = angles[:, 2] + torch.empty(n, device=device).uniform_(-0.3, 0.3)
        th1d = torch.empty(n, device=device).uniform_(-1.2, 1.2)
        th2d = torch.empty(n, device=device).uniform_(-1.2, 1.2)
        th3d = torch.empty(n, device=device).uniform_(-1.2, 1.2)
        return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)

    # Local capture near the requested goal.
    if goals is not None and near_goal_p > 0:
        hit = (torch.rand(n, device=device) < near_goal_p) & ~claimed
        if hit.any():
            angles = goal_angles(goals, device=device, dtype=torch.float32)
            n_hit = int(hit.sum())
            # Quiet-basin: same as near_target (angles/cart ±noise; rates ±0.01).
            n_ang = max(float(init_noise), 1e-3)
            n_x = n_ang
            n_rate = 0.01
            x[hit] = torch.empty(n_hit, device=device).uniform_(-n_x, n_x)
            xd[hit] = torch.empty(n_hit, device=device).uniform_(-n_rate, n_rate)
            th1[hit] = angles[hit, 0] + torch.empty(n_hit, device=device).uniform_(-n_ang, n_ang)
            th2[hit] = angles[hit, 1] + torch.empty(n_hit, device=device).uniform_(-n_ang, n_ang)
            th3[hit] = angles[hit, 2] + torch.empty(n_hit, device=device).uniform_(-n_ang, n_ang)
            th1d[hit] = torch.empty(n_hit, device=device).uniform_(-n_rate, n_rate)
            th2d[hit] = torch.empty(n_hit, device=device).uniform_(-n_rate, n_rate)
            th3d[hit] = torch.empty(n_hit, device=device).uniform_(-n_rate, n_rate)
            claimed |= hit

    # Hang / near-DDD: forces energy pumping toward whatever goal is assigned.
    if hang_start_p > 0:
        hit = (torch.rand(n, device=device) < hang_start_p) & ~claimed
        if hit.any():
            n_hit = int(hit.sum())
            x[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            xd[hit] = torch.empty(n_hit, device=device).uniform_(-0.8, 0.8)
            # θ=0 upright; hang is ±π (DDD). Small noise so it's not a singular rest.
            th1[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
            th2[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
            th3[hit] = math.pi + torch.empty(n_hit, device=device).uniform_(-0.35, 0.35)
            th1d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            th2d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            th3d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            claimed |= hit

    # Start at a *different* discrete equilibrium than the goal (switch / recover edge).
    if goals is not None and wrong_eq_p > 0:
        hit = (torch.rand(n, device=device) < wrong_eq_p) & ~claimed
        if hit.any():
            n_hit = int(hit.sum())
            other = torch.randint(0, NUM_GOALS, (n,), device=device)
            # Reroll collisions with the assigned goal.
            same = other == goals
            if same.any():
                other = torch.where(
                    same,
                    (other + 1 + torch.randint(0, NUM_GOALS - 1, (n,), device=device)) % NUM_GOALS,
                    other,
                )
            # Bias toward tip-link fold pairs (XOR 1).
            if fold_pair_p > 0:
                use_fold = torch.rand(n, device=device) < fold_pair_p
                other = torch.where(use_fold, fold_partner(goals), other)
            angles = goal_angles(other, device=device, dtype=torch.float32)
            x[hit] = torch.empty(n_hit, device=device).uniform_(-0.6, 0.6)
            xd[hit] = torch.empty(n_hit, device=device).uniform_(-0.6, 0.6)
            th1[hit] = angles[hit, 0] + torch.empty(n_hit, device=device).uniform_(-0.25, 0.25)
            th2[hit] = angles[hit, 1] + torch.empty(n_hit, device=device).uniform_(-0.25, 0.25)
            th3[hit] = angles[hit, 2] + torch.empty(n_hit, device=device).uniform_(-0.25, 0.25)
            th1d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            th2d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)
            th3d[hit] = torch.empty(n_hit, device=device).uniform_(-1.0, 1.0)

    return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d), dim=-1)


def apply_impulses(state, p):
    n = state.shape[0]
    hit = torch.rand(n, device=state.device) < p
    if not hit.any():
        return state
    kick = state.clone()
    kick[:, 1] += torch.empty(n, device=state.device).uniform_(-4.0, 4.0)
    kick[:, 3] += torch.empty(n, device=state.device).uniform_(-6.0, 6.0)
    kick[:, 5] += torch.empty(n, device=state.device).uniform_(-6.0, 6.0)
    kick[:, 7] += torch.empty(n, device=state.device).uniform_(-6.0, 6.0)
    # Same soft caps as physics.step so a kick cannot leave unbounded velocities.
    kick[:, 1].clamp_(-_MAX_CART_VEL, _MAX_CART_VEL)
    kick[:, 3].clamp_(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    kick[:, 5].clamp_(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    kick[:, 7].clamp_(-_MAX_ANG_VEL, _MAX_ANG_VEL)
    return torch.where(hit.unsqueeze(-1), kick, state)



def hang_start_p_for_update(args, update: int) -> float:
    """Low/zero hang during UUU warmup; then the configured hang_start_p."""
    warm = int(args.warmup_updates)
    if warm > 0 and update <= warm:
        return float(getattr(args, "warmup_hang_start_p", 0.0))
    return float(args.hang_start_p)


def angle_fall_mask(state, goal_ids, thresh_up=0.6, thresh_down=1.5):
    """Per-link fall: UP targets use tight thresh, DOWN use loose (fawraw)."""
    angles = goal_angles(goal_ids, device=state.device, dtype=state.dtype)
    th = torch.stack((state[..., 2], state[..., 4], state[..., 6]), dim=-1)
    # smallest absolute angle error to target
    err = torch.atan2(torch.sin(th - angles), torch.cos(th - angles)).abs()
    up = torch.cos(angles) > 0.0
    thresh = torch.where(
        up,
        torch.full_like(err, float(thresh_up)),
        torch.full_like(err, float(thresh_down)),
    )
    return (err > thresh).any(dim=-1)


def make_obs(state, goal_ids, constants):
    return normalize_obs(conditioned_obs(observe(state), goal_ids), constants)


# Normalized obs channels that are odd under left–right planar reflection.
# Layout: x, xd, sinθ1, cosθ1, sinθ2, cosθ2, sinθ3, cosθ3, θ1d, θ2d, θ3d, + goal (even).
_FLIP_OBS_IDX = (0, 1, 2, 4, 6, 8, 9, 10)


def flip_lr_state(state: torch.Tensor) -> torch.Tensor:
    """Left–right reflect plant state: (x, xd, θ_i, θd_i) → (−x, −xd, −θ_i, −θd_i).

    Symmetry assumption (Baek VER / planar cart-pendulum): reflecting across the
    vertical midline leaves the Lagrangian invariant when force also flips.
    Equilibria with θ* ∈ {0, π} are invariant (goal sin/cos unchanged).
    """
    out = state.clone()
    out[..., 0] = -state[..., 0]
    out[..., 1] = -state[..., 1]
    out[..., 2] = -state[..., 2]
    out[..., 3] = -state[..., 3]
    out[..., 4] = -state[..., 4]
    out[..., 5] = -state[..., 5]
    out[..., 6] = -state[..., 6]
    out[..., 7] = -state[..., 7]
    return out


def flip_lr_obs(obs: torch.Tensor) -> torch.Tensor:
    """Flip odd channels of a normalized conditioned obs (goal channels unchanged)."""
    out = obs.clone()
    out[..., list(_FLIP_OBS_IDX)] = -out[..., list(_FLIP_OBS_IDX)]
    return out


def export_actor_triple(model, constants):
    """Local export so we do not pull double GOAL_IDS from ppo.export_actor."""
    actor = model.actor
    layers = []
    for module in actor:
        if isinstance(module, nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "weight": module.weight.detach().cpu().tolist(),
                    "bias": module.bias.detach().cpu().tolist(),
                }
            )
        elif isinstance(module, nn.Tanh):
            layers.append({"type": "tanh"})
    obs_dim = getattr(model, "obs_dim", OBS_DIM)
    return {
        "obs_dim": obs_dim,
        "obs_layout": OBS_LAYOUT,
        "goals": list(GOAL_IDS),
        "num_goals": NUM_GOALS,
        "n_links": 3,
        "hidden": constants["hidden"],
        "act_dim": 1,
        "force_limit": constants["forceLimit"],
        "physics": constants,
        "layers": layers,
        "note": (
            "Goal-conditioned UVFA policy for cart-triple-pendulum. "
            f"Obs = 11 state + 8 one-hot + 6 target sin/cos (OBS_DIM={obs_dim})."
        ),
    }


def save_policy(model, constants, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_actor_triple(model, constants)
    path.write_text(json.dumps(payload))


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
        conditioned_obs(observe(rel_states.reshape(-1, 8)), rel_goals.reshape(-1)),
        constants,
    ).reshape(t_h, n_h, OBS_DIM)
    flat_rew = compute_reward(
        rel_states.reshape(-1, 8),
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
def rollout_eval(
    model,
    constants,
    device,
    rkw,
    n=64,
    steps=400,
    track_limit=4.0,
    init_mode="random",
    hang_start_p=0.0,
    near_goal_p=0.0,
):
    """Evaluate each discrete goal; optional init_mode for curriculum meters.

    init_mode:
      random      — harsh wide random ICs (legacy eval/* meter)
      near_target — spawn near assigned goal (hold / catch basin)
      hang        — spawn near hanging DDD (swing-up from bottom)
    """
    model.eval()
    per_goal = {}
    all_rew = []
    all_align = []
    mode = (init_mode or "random").lower()
    for gid, name in enumerate(GOAL_IDS):
        goals = torch.full((n,), gid, device=device, dtype=torch.long)
        if mode == "near_target":
            state = random_states(
                n,
                device,
                constants,
                mild=False,
                goals=goals,
                near_goal_p=1.0 if near_goal_p <= 0 else near_goal_p,
                hang_start_p=0.0,
                init_mode="near_target",
            )
        elif mode in ("hang", "bottom"):
            state = random_states(
                n,
                device,
                constants,
                mild=False,
                goals=goals,
                hang_start_p=1.0 if hang_start_p <= 0 else hang_start_p,
                near_goal_p=0.0,
                init_mode="bottom",
            )
        else:
            state = random_states(n, device, constants, mild=False)
        total = torch.zeros(n, device=device)
        align_sum = torch.zeros(n, device=device)
        hits = torch.zeros(n, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        # Eval: no progress term (no prev); strip progress_w for clean scores.
        eval_rkw = {**rkw, "progress_w": 0.0, "prev_state": None}
        for _ in range(steps):
            obs = make_obs(state, goals, constants)
            force = tanh_action(model.deterministic(obs), constants["forceLimit"]).squeeze(-1)
            state = step(state, force, constants=constants)
            rew = compute_reward(state, force, goals, constants, **eval_rkw)
            # Stop accruing after track fail so eval reward stays interpretable.
            total += torch.where(alive, rew, torch.zeros_like(rew))
            align = mean_cos_align(state, goals)
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
    """Curriculum mix: hard UUU warmup, then optional soft UUU bias, then uniform.

    Returns (probs_tensor_or_None, allowed_list_or_None). When probs is set,
    sample_goals uses multinomial; when only allowed is set, uniform over that
    subset; both None => uniform over all eight goals.
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
    # Optional force override: CLI --force-limit wins over env FORCE_LIMIT over JSON default.
    fl = args.force_limit
    if fl is None:
        env_fl = os.environ.get("FORCE_LIMIT", "").strip()
        if env_fl:
            fl = float(env_fl)
    if fl is not None:
        constants = dict(constants)
        constants["forceLimit"] = float(fl)
        args.force_limit = float(fl)
    # Optional track walls: CLI wins over env TRACK_WALLS over JSON trackWalls.
    tw = args.track_walls
    if tw is None:
        env_tw = os.environ.get("TRACK_WALLS", "").strip().lower()
        if env_tw in ("1", "true", "yes", "on"):
            tw = True
        elif env_tw in ("0", "false", "no", "off"):
            tw = False
        else:
            tw = bool(constants.get("trackWalls", False))
    constants = dict(constants)
    constants["trackWalls"] = bool(tw)
    # Keep physics wall rail aligned with training oob threshold.
    constants["trackLimit"] = float(args.track_limit)
    args.track_walls = bool(tw)
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
        f"reward: mode={args.reward_mode} align_w={args.align_w} energy_w={args.energy_w} spin_w={args.spin_w} "
        f"center_w={args.center_w} center_hold_w={args.center_hold_w} "
        f"barrier={args.cart_barrier_coef} w_up={args.w_up} w_down={args.w_down} "
        f"forceLimit={constants['forceLimit']} "
        f"progress_w={args.progress_w} flip_augment={args.flip_augment} "
        f"eval_curriculum={args.eval_curriculum} "
        f"alpha_th={args.alpha_th} track_limit={args.track_limit} track_walls={args.track_walls} "
        f"reward_clip={args.reward_clip} oob_penalty={args.oob_penalty} "
        f"warmup={args.warmup_updates}x{args.warmup_goal}(hang={args.warmup_hang_start_p}) "
        f"uu_bias={args.uu_bias} anneal={args.anneal_updates} near_goal_p={args.near_goal_p} "
        f"hang_start_p={args.hang_start_p} wrong_eq_p={args.wrong_eq_p} "
        f"goal_switch_p={args.goal_switch_p} fold_pair_p={args.fold_pair_p} "
        f"init_mode={args.init_mode} init_noise={args.init_noise} fall_grace={args.fall_grace_steps} "
        f"start_grace={args.start_grace_steps} angle_fall={args.angle_fall} "
        f"transition_only={args.transition_only} her_ratio={args.her_ratio}",
        flush=True,
    )
    print(f"tensorboard --logdir {args.logdir}", flush=True)

    model = ActorCritic(
        obs_dim=OBS_DIM,
        hidden=constants["hidden"],
        use_sde=bool(getattr(args, "use_sde", False)),
    ).to(device)
    # ERA soft floor applies on rollout+update; RPO α is evaluate-only (passed to evaluate).
    # gSDE (if on) uses actor penultimate features; resample noise every sde_sample_freq updates.
    _floor = float(getattr(args, "log_std_floor", 0.0) or 0.0)
    model.log_std_floor = _floor if _floor > 0 else None
    _sde_freq = int(getattr(args, "sde_sample_freq", 4) or 4)
    print(
        f"explore: ent={args.ent} rpo_alpha={getattr(args, 'rpo_alpha', 0.0)} "
        f"log_std_floor={model.log_std_floor} use_sde={model.use_sde} "
        f"sde_sample_freq={_sde_freq}",
        flush=True,
    )
    if args.checkpoint.is_file():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # gSDE adds sde_weights; cold-start arm removes ckpt, but allow partial load.
        model.load_state_dict(state_dict, strict=not bool(getattr(args, "use_sde", False)))
        prev = ckpt.get("update") if isinstance(ckpt, dict) else None
        print(
            f"loaded checkpoint {args.checkpoint}"
            + (f" (saved at update={prev})" if prev is not None else ""),
            flush=True,
        )
        # SB3 #155: fill_ in-place (do NOT replace nn.Parameter — that orphans Adam state).
        if getattr(args, "reset_log_std", None) is not None:
            with torch.no_grad():
                model.log_std.fill_(float(args.reset_log_std))
            print(
                f"reset log_std → {float(args.reset_log_std)} "
                f"(in-place fill_; Adam rebuilt below)",
                flush=True,
            )
    else:
        print(f"no checkpoint at {args.checkpoint}; cold start", flush=True)
    # Adam after load/reset so momentum matches current log_std (critical after σ collapse).
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    _probs, _allowed = goal_probs_for_update(args, 1)
    goals = sample_goals(args.num_envs, device, allowed=_allowed, probs=_probs)
    state = random_states(
        args.num_envs,
        device,
        constants,
        mild=True,
        goals=goals,
        near_goal_p=args.near_goal_p,
        hang_start_p=hang_start_p_for_update(args, 1),
        wrong_eq_p=args.wrong_eq_p,
        fold_pair_p=args.fold_pair_p,
        transition_only=args.transition_only,
        init_mode=args.init_mode,
        init_noise=args.init_noise,
    )
    steps_left = torch.randint(1, args.episode_len + 1, (args.num_envs,), device=device)
    # Episode age (steps since last reset) for fall/oob grace
    ep_age = torch.zeros(args.num_envs, device=device, dtype=torch.long)

    best_align = -1e9
    t0 = time.time()
    t_prev = time.perf_counter()
    # APO Alg.1 EMA state (persists across updates when --avc-ema-alpha > 0).
    ema_rho = None  # η̂
    ema_v = None  # critic-mean bias b
    for update in range(1, args.updates + 1):
        # Zoo/SB3 gSDE: resample exploration noise every sde_sample_freq updates.
        if model.use_sde and (_sde_freq <= 1 or (update - 1) % _sde_freq == 0):
            model.sample_sde_noise()
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
        oob_hits = 0

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
                    if args.transition_only:
                        # Always change goal: uniform over the other seven eqs.
                        new_g = (
                            goals
                            + 1
                            + torch.randint(0, NUM_GOALS - 1, (args.num_envs,), device=device)
                        ) % NUM_GOALS
                    else:
                        new_g = sample_goals(
                            args.num_envs, device, allowed=allowed, probs=goal_probs
                        )
                        # Avoid no-op flips when possible.
                        same = new_g == goals
                        if same.any():
                            new_g = torch.where(
                                same,
                                (new_g + 1 + torch.randint(0, NUM_GOALS - 1, (args.num_envs,), device=device)) % NUM_GOALS,
                                new_g,
                            )
                        # Bias flips toward tip-link fold (XOR 1).
                        if args.fold_pair_p > 0:
                            use_fold = torch.rand(args.num_envs, device=device) < args.fold_pair_p
                            new_g = torch.where(use_fold, fold_partner(goals), new_g)
                    goals = torch.where(flip, new_g, goals)
            next_state = step(state, force, constants=constants)
            rew = compute_reward(
                next_state, force, goals, constants, prev_state=state, **rkw
            )
            steps_left -= 1
            ep_age = ep_age + 1
            grace_n = int(args.fall_grace_steps) + int(args.start_grace_steps)
            in_grace = ep_age <= grace_n
            # With hard walls, cart is clamped at trackLimit so void oob should
            # not fire; keep a safety check only if somehow past the bumper.
            if args.track_walls:
                oob_raw = next_state[:, 0].abs() > (args.track_limit + 1e-4)
            else:
                oob_raw = next_state[:, 0].abs() > args.track_limit
            oob = oob_raw & (~in_grace)
            fell = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            if args.angle_fall:
                fell_raw = angle_fall_mask(
                    next_state, goals, args.fall_thresh_up, args.fall_thresh_down
                )
                fell = fell_raw & (~in_grace)
            done = (steps_left <= 0) | oob | fell
            oob_hits += int(oob.sum().item())
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
                    hang_start_p=hang_start_p_for_update(args, update),
                    wrong_eq_p=args.wrong_eq_p,
                    fold_pair_p=args.fold_pair_p,
                    transition_only=args.transition_only,
                    init_mode=args.init_mode,
                    init_noise=args.init_noise,
                )
                next_state = torch.where(done.unsqueeze(-1), reset, next_state)
                goals = reset_goals
                steps_left = torch.where(
                    done,
                    torch.full_like(steps_left, args.episode_len),
                    steps_left,
                )
                ep_age = torch.where(done, torch.zeros_like(ep_age), ep_age)
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
        # ATRPO-lite: subtract batch-mean reward ρ̂, drop γ from TD/GAE (λ only).
        # With --avc-ema-alpha>0, ρ̂ is EMA-smoothed η̂ (APO Alg.1).
        rho_batch = rew_t.mean() if args.avg_reward else rew_t.new_zeros(())
        avc_ema_alpha = float(getattr(args, "avc_ema_alpha", 0.0) or 0.0)
        if args.avg_reward and avc_ema_alpha > 0.0:
            if ema_rho is None:
                ema_rho = rho_batch.detach()
            else:
                ema_rho = (1.0 - avc_ema_alpha) * ema_rho + avc_ema_alpha * rho_batch.detach()
            rho_hat = ema_rho
        else:
            rho_hat = rho_batch
        for t in reversed(range(args.rollout)):
            mask = 1.0 - done_t[t]
            if args.avg_reward:
                delta = (rew_t[t] - rho_hat) + next_value * mask - val_t[t]
                last_adv = delta + args.gae * mask * last_adv
            else:
                delta = rew_t[t] + args.gamma * next_value * mask - val_t[t]
                last_adv = delta + args.gamma * args.gae * mask * last_adv
            adv[t] = last_adv
            next_value = val_t[t]
        ret = adv + val_t
        # APO AVC (2106.03442): pin differential-value offset via ν·bias.
        # AVC-lite (α=0): bias = mean(ret). EMA-AVC (α>0): bias = EMA of mean(V_φ).
        avc_nu = float(getattr(args, "avc_nu", 0.0) or 0.0)
        avc_bias = ret.new_zeros(())
        critic_mean = val_t.mean().detach()
        if avc_nu > 0.0:
            if avc_ema_alpha > 0.0:
                if ema_v is None:
                    ema_v = critic_mean
                else:
                    ema_v = (1.0 - avc_ema_alpha) * ema_v + avc_ema_alpha * critic_mean
                avc_bias = ema_v
            else:
                avc_bias = ret.mean().detach()
            ret = ret - avc_nu * avc_bias
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Baek VER / left–right flip augment (on-policy): duplicate PPO batch.
        # Rewards/advantages unchanged under planar symmetry; recompute logπ(-a|s').
        if args.flip_augment:
            obs_f = flip_lr_obs(obs_t)
            raw_f = -raw_t
            with torch.no_grad():
                t_len, n_envs = obs_f.shape[0], obs_f.shape[1]
                log_f, _, _ = model.evaluate(
                    obs_f.reshape(-1, OBS_DIM), raw_f.reshape(-1, 1)
                )
                log_f = log_f.reshape(t_len, n_envs)
            obs_t = torch.cat((obs_t, obs_f), dim=1)
            raw_t = torch.cat((raw_t, raw_f), dim=1)
            log_t = torch.cat((log_t, log_f), dim=1)
            adv = torch.cat((adv, adv), dim=1)
            ret = torch.cat((ret, ret), dim=1)

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
                _floor = getattr(args, "log_std_floor", 0.0) or 0.0
                log_p, ent, value = model.evaluate(
                    obs_flat[idx],
                    raw_flat[idx],
                    rpo_alpha=float(getattr(args, "rpo_alpha", 0.0) or 0.0),
                    log_std_floor=_floor if _floor > 0 else None,
                )
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
        # If every minibatch was skipped (non-finite loss), last_* stay None —
        # do not crash TensorBoard logging; mark the skip so we can see it.
        if last_policy is not None:
            writer.add_scalar("train/policy_loss", last_policy, update)
            writer.add_scalar("train/value_loss", last_value, update)
            writer.add_scalar("train/entropy", last_ent, update)
            if args.avg_reward:
                writer.add_scalar("train/avg_reward_rho", float(rho_hat.detach()), update)
                writer.add_scalar("train/avg_reward_rho_batch", float(rho_batch.detach()), update)
            if float(getattr(args, "avc_nu", 0.0) or 0.0) > 0.0:
                writer.add_scalar("train/avc_bias", float(avc_bias.detach()), update)
                writer.add_scalar("train/avc_nu", float(args.avc_nu), update)
                writer.add_scalar("train/critic_mean", float(critic_mean.detach()), update)
                if float(getattr(args, "avc_ema_alpha", 0.0) or 0.0) > 0.0:
                    writer.add_scalar("train/avc_ema_alpha", float(args.avc_ema_alpha), update)
            writer.add_scalar("train/loss", last_loss, update)
        else:
            writer.add_scalar("train/skipped_all_minibatches", 1.0, update)

        # Throughput / hardware efficiency (every update; GPU sample is cheap).
        t_now = time.perf_counter()
        dt = max(1e-6, t_now - t_prev)
        t_prev = t_now
        env_steps = float(args.num_envs * args.rollout)
        writer.add_scalar(
            "train/oob_rate",
            float(oob_hits) / float(args.num_envs * args.rollout),
            update,
        )
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
            n_eval = 16 if args.smoke else 64
            n_steps = 80 if args.smoke else 400
            mean_rew, mean_align, per_goal = rollout_eval(
                model,
                constants,
                device,
                rkw,
                n=n_eval,
                steps=n_steps,
                track_limit=args.track_limit,
                init_mode="random",
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
            writer.add_scalar("eval/upright", per_goal["UUU"]["align"], update)
            for name, metrics in per_goal.items():
                writer.add_scalar(f"eval/reward/{name}", metrics["reward"], update)
                writer.add_scalar(f"eval/align/{name}", metrics["align"], update)
                writer.add_scalar(f"eval/at_goal/{name}", metrics["at_goal"], update)

            if args.eval_curriculum:
                # Hold-basin meter (diagnoses capture; random-IC understates this).
                _, _, nt = rollout_eval(
                    model,
                    constants,
                    device,
                    rkw,
                    n=n_eval,
                    steps=n_steps,
                    track_limit=args.track_limit,
                    init_mode="near_target",
                )
                for name, metrics in nt.items():
                    writer.add_scalar(
                        f"eval/near_target/reward/{name}", metrics["reward"], update
                    )
                    writer.add_scalar(
                        f"eval/near_target/align/{name}", metrics["align"], update
                    )
                    writer.add_scalar(
                        f"eval/near_target/at_goal/{name}", metrics["at_goal"], update
                    )
                writer.add_scalar(
                    "eval/near_target/align",
                    sum(nt[g]["align"] for g in GOAL_IDS) / NUM_GOALS,
                    update,
                )
                # Swing-from-hang meter.
                _, _, hg = rollout_eval(
                    model,
                    constants,
                    device,
                    rkw,
                    n=n_eval,
                    steps=n_steps,
                    track_limit=args.track_limit,
                    init_mode="hang",
                )
                for name, metrics in hg.items():
                    writer.add_scalar(
                        f"eval/hang/reward/{name}", metrics["reward"], update
                    )
                    writer.add_scalar(
                        f"eval/hang/align/{name}", metrics["align"], update
                    )
                    writer.add_scalar(
                        f"eval/hang/at_goal/{name}", metrics["at_goal"], update
                    )
                writer.add_scalar(
                    "eval/hang/align",
                    sum(hg[g]["align"] for g in GOAL_IDS) / NUM_GOALS,
                    update,
                )
                print(
                    f"  eval/near_target/at_goal/UUU={nt['UUU']['at_goal']:.4f} "
                    f"align/UUU={nt['UUU']['align']:.3f} | "
                    f"eval/hang/at_goal/UUU={hg['UUU']['at_goal']:.4f} "
                    f"align/UUU={hg['UUU']['align']:.3f}",
                    flush=True,
                )

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
    print(f"wrote {args.out}")
    print(f"tensorboard --logdir {args.logdir}")


if __name__ == "__main__":
    main()
