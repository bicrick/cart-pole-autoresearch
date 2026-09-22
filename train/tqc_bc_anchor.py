"""TQC with a TD3+BC-style anchor to the cloned actor.

One extra term on the actor loss after unfreeze:

    actor_loss += bc_reg_coef * |Q|.mean() * MSE(tanh μ, tanh μ_BC)

Keeps the hold from walking off a working BC/LQR policy when Q is still noisy.
"""

from __future__ import annotations

import copy

import numpy as np
import torch as th
import torch.nn.functional as F
from sb3_contrib.common.utils import quantile_huber_loss
from sb3_contrib.tqc.tqc import TQC
from stable_baselines3.common.utils import polyak_update


class TQCWithBCAnchor(TQC):
    def __init__(self, *args, bc_reg_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_reg_coef = float(bc_reg_coef)
        self._bc_anchor = None
        self._last_bc_loss = 0.0

    def snapshot_actor_anchor(self) -> None:
        self._bc_anchor = copy.deepcopy(self.actor)
        self._bc_anchor.eval()
        for p in self._bc_anchor.parameters():
            p.requires_grad = False
        print(
            f"[hold] BC actor anchor snapped (bc_reg_coef={self.bc_reg_coef:g})",
            flush=True,
        )

    def _bc_mse(self, obs: th.Tensor) -> th.Tensor:
        if self._bc_anchor is None or self.bc_reg_coef <= 0:
            return obs.new_zeros(())
        mean_pi, _, _ = self.actor.get_action_dist_params(obs)
        with th.no_grad():
            mean_bc, _, _ = self._bc_anchor.get_action_dist_params(obs)
        return F.mse_loss(th.tanh(mean_pi), th.tanh(mean_bc))

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                n_target_quantiles = (
                    self.critic.quantiles_total
                    - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                )
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target_quantiles]
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_quantiles
                target_quantiles.unsqueeze_(dim=1)

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - qf_pi).mean()
            bc_loss = self._bc_mse(replay_data.observations)
            if self.bc_reg_coef > 0 and bc_loss.ndim == 0:
                scale = qf_pi.detach().abs().mean().clamp(min=1.0)
                actor_loss = actor_loss + (self.bc_reg_coef * scale) * bc_loss
                bc_losses.append(float(bc_loss.detach()))
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if bc_losses:
            self.logger.record("train/bc_anchor_mse", float(np.mean(bc_losses)))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> list:
        return super()._excluded_save_params() + ["_bc_anchor"]
