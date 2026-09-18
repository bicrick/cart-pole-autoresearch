#!/usr/bin/env python3
"""Batched PPO for cart + double pendulum. Smoke on CPU, scale on GPU."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from physics import load_constants, normalize_obs, observe, reward, step
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
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "policies" / "checkpoint.pt")
    parser.add_argument("--out", type=Path, default=POLICY_PATH)
    return parser.parse_args()


def random_states(n, device, constants, mild=False):
    x = torch.empty(n, device=device).uniform_(-2.0, 2.0)
    xd = torch.empty(n, device=device).uniform_(-3.0, 3.0)
    th1 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th2 = torch.empty(n, device=device).uniform_(-math.pi, math.pi)
    th1d = torch.empty(n, device=device).uniform_(-8.0, 8.0)
    th2d = torch.empty(n, device=device).uniform_(-8.0, 8.0)
    if mild:
        # Keep most mass on the circle; sprinkle a few near-upright catch states.
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


def save_policy(model, constants, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_actor(model, constants)
    path.write_text(json.dumps(payload))
    WEB_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    WEB_POLICY_PATH.write_text(json.dumps(payload))


@torch.no_grad()
def rollout_eval(model, constants, device, n=256):
    model.eval()
    state = random_states(n, device, constants, mild=False)
    total = torch.zeros(n, device=device)
    upright = torch.zeros(n, device=device)
    for _ in range(400):
        obs = normalize_obs(observe(state), constants)
        force = tanh_action(model.deterministic(obs), constants["forceLimit"]).squeeze(-1)
        state = step(state, force, constants=constants)
        total += reward(state, force, constants)
        upright += 0.5 * (torch.cos(state[:, 2]) + torch.cos(state[:, 4]))
    model.train()
    return float(total.mean()), float(upright.mean() / 400)


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
    print(f"device={device} envs={args.num_envs} rollout={args.rollout} updates={args.updates}")

    model = ActorCritic(hidden=constants["hidden"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    state = random_states(args.num_envs, device, constants, mild=True)
    steps_left = torch.randint(1, args.episode_len + 1, (args.num_envs,), device=device)

    best_upright = -1e9
    t0 = time.time()
    for update in range(1, args.updates + 1):
        obs_buf = []
        raw_buf = []
        log_buf = []
        val_buf = []
        rew_buf = []
        done_buf = []

        for _ in range(args.rollout):
            obs = normalize_obs(observe(state), constants)
            with torch.no_grad():
                raw, log_prob, value = model.act(obs)
            force = tanh_action(raw, constants["forceLimit"]).squeeze(-1)
            state = apply_impulses(state, args.impulse_p)
            next_state = step(state, force, constants=constants)
            rew = reward(next_state, force, constants)
            steps_left -= 1
            done = steps_left <= 0
            if done.any():
                reset = random_states(args.num_envs, device, constants, mild=True)
                next_state = torch.where(done.unsqueeze(-1), reset, next_state)
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

        with torch.no_grad():
            last_obs = normalize_obs(observe(state), constants)
            last_val = model.critic(last_obs).squeeze(-1)

        obs_t = torch.stack(obs_buf)
        raw_t = torch.stack(raw_buf)
        log_t = torch.stack(log_buf)
        val_t = torch.stack(val_buf)
        rew_t = torch.stack(rew_buf)
        done_t = torch.stack(done_buf)

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

        obs_flat = obs_t.reshape(-1, 8)
        raw_flat = raw_t.reshape(-1, 1)
        log_flat = log_t.reshape(-1)
        adv_flat = adv.reshape(-1)
        ret_flat = ret.reshape(-1)
        n = obs_flat.shape[0]
        mb = min(args.minibatch, n)

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
                loss.backward()
                nn_clip = 0.5
                torch.nn.utils.clip_grad_norm_(model.parameters(), nn_clip)
                opt.step()

        if update == 1 or update % 10 == 0 or args.smoke:
            mean_rew, upright = rollout_eval(model, constants, device, n=64 if args.smoke else 256)
            elapsed = time.time() - t0
            print(
                f"update={update:04d} reward={mean_rew:.3f} upright={upright:.3f} "
                f"sec={elapsed:.1f}",
                flush=True,
            )
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "update": update}, args.checkpoint)
            if upright >= best_upright:
                best_upright = upright
                save_policy(model, constants, args.out)

    save_policy(model, constants, args.out)
    print(f"wrote {args.out} and {WEB_POLICY_PATH}")


if __name__ == "__main__":
    main()
