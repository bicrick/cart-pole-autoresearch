"""Put LQR braking episodes into a TQC replay buffer.

The gated actor loss only sees states the behavior policy visits. A pump
leaves the catchable set in one step, so the loss never fits the rest of the
brake. These episodes are the trajectory LQR actually settles.
"""

from __future__ import annotations

import numpy as np
import torch

from envs.triple_gym import TriplePendulumUUUEnv
from lqr_uuu import design_lqr_uuu


def _arrival_state(omega: float) -> torch.Tensor:
    s = torch.zeros(8)
    s[2] = 0.05
    s[4] = 0.05
    s[6] = 0.05
    s[3] = float(omega)
    s[5] = float(omega)
    s[7] = float(omega)
    return s


def seed_lqr_arrival(model, args) -> dict:
    """Add `lqr_seed_episodes` LQR arrivals on every vec-env slot.

    Spawn matches the swing arrival mix: all links at 0.05 rad, shared rate
    uniform in [0.05, arrival_omega]. Returns a small summary for the log.
    """
    n_eps = int(getattr(args, "lqr_seed_episodes", 0) or 0)
    if n_eps <= 0:
        return {"episodes": 0, "transitions": 0}
    n_envs = int(model.n_envs)
    lqr = design_lqr_uuu(force_limit=float(args.force_limit))
    envs = [
        TriplePendulumUUUEnv(
            force_limit=float(args.force_limit),
            max_steps=int(args.max_steps),
            track_walls=True,
            init_mode="bottom",
            init_noise=float(args.init_noise),
            angle_fall=False,
            goal=str(args.goal),
            progress_w=float(args.progress_w),
            cart_barrier_coef=float(args.cart_barrier_coef),
            reward_mode=str(args.reward_mode),
            vel_cost_coef=float(args.vel_cost_coef),
            cart_cost_coef=float(args.cart_cost_coef),
            transition_bonus=float(args.transition_bonus),
            transition_tol=float(args.transition_tol),
            transition_steps=int(args.transition_steps),
            soft_landing_rad=float(args.soft_landing_rad),
            seed=int(args.seed) + i,
        )
        for i in range(n_envs)
    ]
    rng = np.random.default_rng(int(args.seed) + 17)
    w_max = float(args.arrival_omega)
    returns = []
    added = 0
    for ep in range(n_eps):
        obs = []
        for i, env in enumerate(envs):
            env.reset(seed=int(args.seed) + ep * n_envs + i)
            mag = float(rng.uniform(0.05, max(w_max, 0.05)))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            s = _arrival_state(sign * mag)
            env._state = s
            env._prev_state = s.clone()
            obs.append(env._obs_np())
        obs_b = np.stack(obs).astype(np.float32)
        ep_ret = np.zeros(n_envs, dtype=np.float64)
        for _t in range(int(args.max_steps)):
            acts = np.stack(
                [
                    lqr.action_normed(env._state.detach().cpu().numpy())
                    for env in envs
                ]
            ).astype(np.float32)
            next_obs = []
            rewards = np.zeros(n_envs, dtype=np.float32)
            dones = np.zeros(n_envs, dtype=np.float32)
            infos = []
            for i, env in enumerate(envs):
                o2, r, term, trunc, _info = env.step(acts[i])
                next_obs.append(o2)
                rewards[i] = np.float32(r)
                ep_ret[i] += float(r)
                done = bool(term or trunc)
                dones[i] = np.float32(done)
                infos.append({"TimeLimit.truncated": bool(trunc and not term)})
            model.replay_buffer.add(obs_b, np.stack(next_obs).astype(np.float32), acts, rewards, dones, infos)
            added += n_envs
            obs_b = np.stack(next_obs).astype(np.float32)
            if bool(dones.min()):
                break
        returns.append(ep_ret)
    summary = {
        "episodes": n_eps * n_envs,
        "transitions": added,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
    }
    print(
        f"[swing] LQR brake seed episodes={summary['episodes']} "
        f"transitions={summary['transitions']} mean_return={summary['mean_return']:.1f}",
        flush=True,
    )
    return summary
