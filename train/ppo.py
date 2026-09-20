"""Tiny actor-critic PPO for the batched cart-double-pendulum."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from goals import GOAL_IDS, NUM_GOALS, OBS_DIM, OBS_LAYOUT


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden=128, act_dim=1, use_sde: bool = False):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.act_dim = act_dim
        self.use_sde = bool(use_sde)
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
        # Optional ERA soft floor (H₀); set from trainer. RPO α is evaluate-only.
        self.log_std_floor = None  # type: float | None
        # gSDE (Raffin / SB3 Zoo): state-dependent exploration from actor penultimate
        # features (not raw obs). Learnable sde_weights (H×A) × resampled ε buffer.
        # Works together with RPO mean jitter (evaluate) and ERA soft log_std floor.
        if self.use_sde:
            self.sde_weights = nn.Parameter(torch.ones(hidden, act_dim))
            self.register_buffer("_sde_noise", torch.zeros(hidden, act_dim))
            self.sample_sde_noise()
        self._init()

    def _init(self):
        last = self.actor[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

    def sample_sde_noise(self):
        """Resample gSDE exploration noise ε ~ N(0, I). Call every sde_sample_freq updates."""
        if not getattr(self, "use_sde", False):
            return
        with torch.no_grad():
            self._sde_noise.normal_(0.0, 1.0)

    def _actor_features(self, obs):
        """Actor penultimate features (all layers except final mean Linear).

        Documented gSDE feature source: SB3-style latent_sde from policy trunk,
        not raw observations. Tanh-bounded hidden keeps noise scale sane.
        """
        x = obs
        for mod in list(self.actor.children())[:-1]:
            x = mod(x)
        # Scale like SB3 latent norm so feat @ explor ~ O(1) before * std.
        return x / math.sqrt(float(self.hidden))

    def _actor_mean(self, obs):
        # Keep Normal(loc, scale) finite on MPS/CUDA when physics NaNs leak into obs.
        mean = self.actor(obs)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        return mean.clamp(-10.0, 10.0)

    def _std_from_log(self, mean, log_std_floor: float | None = None):
        log_std = self.log_std.clamp(-5.0, 2.0)
        if log_std_floor is None:
            log_std_floor = self.log_std_floor
        if log_std_floor is not None and log_std_floor > 0:
            # 1-D Normal entropy H = log_std + 0.5*(1+ln(2π)) ≈ log_std + 1.4189
            # Soft floor at H≥H₀ ⇔ log_std ≥ H₀ - 1.4189 without pinning grad to 0.
            log_std_min = float(log_std_floor) - 0.5 * (1.0 + math.log(2.0 * math.pi))
            gap = torch.nn.functional.softplus(log_std_min - log_std)
            log_std = log_std + gap.detach()
        std = log_std.exp().expand_as(mean)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-6)
        return std

    def dist(self, obs, rpo_alpha: float = 0.0, log_std_floor: float | None = None):
        """Gaussian policy. Optional CleanRL RPO mean jitter (update-only) and
        ERA-style soft log_std floor (H₀ as min entropy for 1-D Normal).
        Floor uses softplus/detached hinge — never hard-clamp log_std (zeros grad).
        When use_sde: same Normal(mean, std) for entropy/log_prob; sampling in act()
        uses state-dependent noise from penultimate features (gSDE).
        """
        mean = self._actor_mean(obs)
        if rpo_alpha and rpo_alpha > 0:
            # CleanRL RPO: μ ← μ + U(-α, α) at *evaluate* time only (arXiv:2212.07536).
            mean = mean + torch.empty_like(mean).uniform_(-rpo_alpha, rpo_alpha)
        std = self._std_from_log(mean, log_std_floor=log_std_floor)
        return Normal(mean, std)

    def act(self, obs):
        dist = self.dist(obs)
        if self.use_sde:
            # gSDE sample: a = μ + (φ(s)/√H · (W ⊙ ε)) ⊙ σ
            # ε resampled on schedule; W=sde_weights learnable; σ from log_std (+ERA).
            feat = self._actor_features(obs)
            explor_mat = self.sde_weights * self._sde_noise
            sde_eps = feat @ explor_mat
            raw = dist.mean + sde_eps * dist.stddev
        else:
            raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return raw, log_prob, value

    def evaluate(self, obs, raw_action, rpo_alpha: float = 0.0, log_std_floor: float | None = None):
        dist = self.dist(obs, rpo_alpha=rpo_alpha, log_std_floor=log_std_floor)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value

    def deterministic(self, obs):
        return self._actor_mean(obs)


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
