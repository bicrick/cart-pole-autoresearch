"""Tiny actor-critic PPO for the batched cart-double-pendulum."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=8, hidden=128, act_dim=1):
        super().__init__()
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
    return {
        "obs_dim": 8,
        "hidden": constants["hidden"],
        "act_dim": 1,
        "force_limit": constants["forceLimit"],
        "physics": constants,
        "layers": layers,
    }
