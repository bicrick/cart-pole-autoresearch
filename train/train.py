#!/usr/bin/env python3
"""Batched goal-conditioned PPO for cart + double pendulum. Smoke on CPU, scale on GPU."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from goals import (
    GOAL_IDS,
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
from physics import load_constants, normalize_obs, observe, step
from ppo import ActorCritic, export_actor, tanh_action

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policies" / "policy.json"
WEB_POLICY_PATH = ROOT / "web" / "public" / "policy.json"


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
    parser.add_argument("--episode-len", type=int, default=1200)
    parser.add_argument("--impulse-p", type=float, default=0.01)
    parser.add_argument(
        "--her-ratio",
        type=float,
        default=0.8,
        help="Fraction of failed episodes to relabel with nearest achieved equilibrium",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "policies" / "checkpoint.pt")
    parser.add_argument("--out", type=Path, default=POLICY_PATH)
    parser.add_argument("--logdir", type=Path, default=ROOT / "runs")
    return parser.parse_args()


def random_states(n, device, constants, mild=False):
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
    obs_t, rew_t, val_t, state_t, force_t, goal_t, done_t, model, constants, her_ratio
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
def rollout_eval(model, constants, device, n=64, steps=400):
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
        for _ in range(steps):
            obs = make_obs(state, goals, constants)
            force = tanh_action(model.deterministic(obs), constants["forceLimit"]).squeeze(-1)
            state = step(state, force, constants=constants)
            total += goal_reward(state, force, goals, constants)
            angles = goal_angles(goals, device=device, dtype=state.dtype)
            align = 0.5 * (
                angle_align(state[:, 2], angles[:, 0]) + angle_align(state[:, 4], angles[:, 1])
            )
            align_sum += align
            hits += at_goal(state, goals).float()
        mean_rew = float(total.mean())
        mean_align = float(align_sum.mean() / steps)
        mean_hit = float(hits.mean() / steps)
        per_goal[name] = {"reward": mean_rew, "align": mean_align, "at_goal": mean_hit}
        all_rew.append(mean_rew)
        all_align.append(mean_align)
    model.train()
    return sum(all_rew) / NUM_GOALS, sum(all_align) / NUM_GOALS, per_goal


def main():
    args = parse_args()
    constants = load_constants()
    if args.smoke:
        args.num_envs = min(args.num_envs, 32)
        args.rollout = 32
        args.updates = 2
        args.minibatch = 64
        args.episode_len = 64
    if args.device:
        device_name = args.device
    elif torch.cuda.is_available():
        device_name = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device_name = "mps"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    logdir = args.logdir / time.strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logdir))
    print(
        f"device={device} envs={args.num_envs} rollout={args.rollout} "
        f"updates={args.updates} obs_dim={OBS_DIM} goals={GOAL_IDS} logdir={logdir}"
    )
    print(f"tensorboard --logdir {args.logdir}", flush=True)

    model = ActorCritic(obs_dim=OBS_DIM, hidden=constants["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = random_states(args.num_envs, device, constants, mild=True)
    goals = sample_goals(args.num_envs, device)
    steps_left = torch.randint(1, args.episode_len + 1, (args.num_envs,), device=device)

    best_align = -1e9
    t0 = time.time()
    for update in range(1, args.updates + 1):
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
            next_state = step(state, force, constants=constants)
            rew = goal_reward(next_state, force, goals, constants)
            steps_left -= 1
            done = steps_left <= 0
            # Buffer transition under the goal that produced the reward (pre-reset).
            goal_buf.append(goals.detach().clone())
            state_buf.append(next_state.detach())
            force_buf.append(force.detach())
            if done.any():
                reset = random_states(args.num_envs, device, constants, mild=True)
                next_state = torch.where(done.unsqueeze(-1), reset, next_state)
                new_goals = sample_goals(args.num_envs, device)
                goals = torch.where(done, new_goals, goals)
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
        for gid, name in enumerate(GOAL_IDS):
            frac = float((goal_t[-1] == gid).float().mean())
            writer.add_scalar(f"train/goal_frac/{name}", frac, update)

        if update == 1 or update % 10 == 0 or args.smoke:
            mean_rew, mean_align, per_goal = rollout_eval(
                model,
                constants,
                device,
                n=32 if args.smoke else 64,
                steps=200 if args.smoke else 400,
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
