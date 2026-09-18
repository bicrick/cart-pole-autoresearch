"""Tiny actor-critic PPO for the batched cart-double-pendulum."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from goals import GOAL_IDS, NUM_GOALS, OBS_DIM, OBS_LAYOUT


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=128, act_dim=1):
        super().__init__()
        self.obs_dim = obs_dim
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self._init()

    def _init(self):
        last = self.actor[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

    def dist(self, obs):
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs):
        dist = self.dist(obs)
        raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return raw, log_prob, value

    def evaluate(self, obs, raw_action):
        dist = self.dist(obs)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def deterministic(self, obs):
        return self.actor(obs)


def tanh_action(raw, force_limit):
    return torch.tanh(raw) * force_limit


def export_actor(model: ActorCritic, constants: dict) -> dict:
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
        "hidden": constants["hidden"],
        "act_dim": 1,
        "force_limit": constants["forceLimit"],
        "physics": constants,
        "layers": layers,
        "note": (
            "Goal-conditioned UVFA policy. Browser inference is plain JS MLP "
            "(no ONNX / TF.js). Obs = 8 state features + 4 one-hot goal + 4 target sin/cos."
        ),
    }
