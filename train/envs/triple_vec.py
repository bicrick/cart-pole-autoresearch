"""Batched SB3 VecEnv for the cart-triple UUU plant.

Steps N copies in one physics_triple.step so TQC can feed Apple Silicon
without a Python loop per env. Observation / action contract matches
TriplePendulumUUUEnv (11-D obs, action in [-1, 1]).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from envs.triple_gym import (
    FALL_THRESH_DOWN,
    FALL_THRESH_UP,
    SURVIVAL_FRAC,
    angle_fall_mask,
    sample_states,
)
from hard_ics import load_bank, mix_hard_ics
from goals_triple import (
    GOAL_IDS,
    at_goal,
    compute_reward,
    mean_cos_align,
    parse_goal,
    wrapped_angle_error,
)
from physics_triple import load_constants, observe, step


def _obs_space() -> spaces.Box:
    high = np.array(
        [4.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0],
        dtype=np.float32,
    )
    return spaces.Box(low=-high, high=high, dtype=np.float32)


def _act_space() -> spaces.Box:
    return spaces.Box(
        low=np.array([-1.0], dtype=np.float32),
        high=np.array([1.0], dtype=np.float32),
        dtype=np.float32,
    )


class TripleUUUVecEnv(VecEnv):
    """Vectorized UUU hold env. Physics on `device` (cpu or mps)."""

    def __init__(
        self,
        n_envs: int = 32,
        force_limit: float = 40.0,
        max_steps: int = 1000,
        track_limit: float = 2.4,
        track_walls: bool = True,
        init_mode: str = "near_target",
        init_noise: float = 0.05,
        init_noise_min: float = 0.0,
        hang_frac: float = 0.0,
        wide_frac: float = 0.0,
        progress_w: float = 0.0,
        cart_barrier_coef: float = 10.0,
        alpha_th: float = 0.5,
        w_up: float = 5.0,
        w_down: float = 1.0,
        sparse_bonus: float = 1.0,
        energy_w: float = 0.0,
        oob_penalty: float = 20.0,
        reward_mode: str = "product",
        vel_cost_coef: float = 0.02,
        cart_cost_coef: float = 0.2,
        transition_bonus: float = 0.0,
        transition_tol: float = 0.3,
        transition_steps: int = 100,
        fall_thresh_up: float = FALL_THRESH_UP,
        fall_thresh_down: float = FALL_THRESH_DOWN,
        survival_frac: float = SURVIVAL_FRAC,
        angle_fall: bool = True,
        device: str = "cpu",
        seed: Optional[int] = None,
        hard_ic_path: str = "",
        hard_ic_frac: float = 0.0,
        goal: str | int = "UUU",
    ):
        n_envs = max(1, int(n_envs))
        super().__init__(n_envs, _obs_space(), _act_space())
        self.constants = dict(load_constants())
        self.constants["forceLimit"] = float(force_limit)
        self.constants["trackLimit"] = float(track_limit)
        self.constants["trackWalls"] = bool(track_walls)
        self.force_limit = float(force_limit)
        self.max_steps = int(max_steps)
        self.track_limit = float(track_limit)
        self.track_walls = bool(track_walls)
        self.init_mode = init_mode
        self.init_noise = float(init_noise)
        self.init_noise_min = float(init_noise_min)
        self.hang_frac = float(hang_frac)
        self.wide_frac = float(wide_frac)
        self.fall_thresh_up = float(fall_thresh_up)
        self.fall_thresh_down = float(fall_thresh_down)
        self.survival_frac = float(survival_frac)
        self.angle_fall = bool(angle_fall)
        self.goal_id = parse_goal(goal)
        self.goal_name = GOAL_IDS[self.goal_id]
        self.device = torch.device(device)
        self._reward_kwargs = dict(
            track_limit=self.track_limit,
            oob_penalty=float(oob_penalty),
            progress_w=float(progress_w),
            cart_barrier_coef=float(cart_barrier_coef),
            alpha_th=float(alpha_th),
            w_up=float(w_up),
            w_down=float(w_down),
            energy_w=float(energy_w),
            sparse_bonus=float(sparse_bonus),
            vel_cost_coef=float(vel_cost_coef),
            cart_cost_coef=float(cart_cost_coef),
        )
        self.reward_mode = (reward_mode or "product").lower()
        self.transition_bonus = float(transition_bonus)
        self.transition_tol = float(transition_tol)
        self.transition_steps = int(transition_steps)
        self._in_tol = torch.zeros(n_envs, dtype=torch.int32, device=self.device)
        self._bonus_paid = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        self._goal = torch.full(
            (n_envs,), self.goal_id, device=self.device, dtype=torch.long
        )
        self._state = torch.zeros(n_envs, 8, device=self.device)
        self._prev = torch.zeros(n_envs, 8, device=self.device)
        self._steps = torch.zeros(n_envs, dtype=torch.int32, device=self.device)
        self._ep_ret = np.zeros(n_envs, dtype=np.float64)
        self._pending_actions: Optional[np.ndarray] = None
        self.hard_ic_frac = float(hard_ic_frac)
        self._hard_bank = (
            load_bank(hard_ic_path, self.device) if hard_ic_path else None
        )
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

    def _obs_np(self) -> np.ndarray:
        return observe(self._state).detach().cpu().numpy().astype(np.float32)

    def _resample(self, mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        fresh = sample_states(
            int(idx.numel()),
            self.init_mode,
            self.hang_frac,
            self.wide_frac,
            self.init_noise,
            self.device,
            noise_min=self.init_noise_min,
            goal_id=self.goal_id,
        )
        if self._hard_bank is not None:
            fresh = mix_hard_ics(fresh, self._hard_bank, self.hard_ic_frac)
        self._state[idx] = fresh
        self._prev[idx] = fresh
        self._steps[idx] = 0
        self._in_tol[idx] = 0
        self._bonus_paid[idx] = False
        self._ep_ret[idx.detach().cpu().numpy()] = 0.0

    def reset(self) -> np.ndarray:
        self._resample(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self._obs_np()

    def step_async(self, actions: np.ndarray) -> None:
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        assert self._pending_actions is not None
        raw = np.clip(self._pending_actions.reshape(self.num_envs, -1)[:, 0], -1.0, 1.0)
        force_np = raw * self.force_limit
        force = torch.as_tensor(force_np, device=self.device, dtype=torch.float32)
        prev = self._state
        nxt = step(prev, force, constants=self.constants)
        self._prev = prev
        self._state = nxt
        self._steps += 1

        rew_t = compute_reward(
            nxt,
            force,
            self._goal,
            self.constants,
            reward_mode=self.reward_mode,
            prev_state=prev,
            **self._reward_kwargs,
        )
        if self.reward_mode in ("m4", "m4_swing") and self.transition_bonus > 0:
            err = wrapped_angle_error(nxt, self._goal)
            in_tol = (err.abs() < self.transition_tol).all(dim=-1)
            self._in_tol = torch.where(
                in_tol, self._in_tol + 1, torch.zeros_like(self._in_tol)
            )
            pay = (self._in_tol >= self.transition_steps) & (~self._bonus_paid)
            rew_t = rew_t + float(self.transition_bonus) * pay.to(rew_t.dtype)
            self._bonus_paid = self._bonus_paid | pay
        rewards = rew_t.detach().cpu().numpy().astype(np.float64)
        self._ep_ret += rewards

        x = nxt[:, 0]
        if self.track_walls:
            oob = x.abs() > (self.track_limit + 1e-4)
        else:
            oob = x.abs() > self.track_limit
        fell = (
            angle_fall_mask(
                nxt, self._goal, self.fall_thresh_up, self.fall_thresh_down
            )
            if self.angle_fall
            else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        )
        truncated = self._steps >= self.max_steps
        terminated = oob | fell
        done = terminated | truncated
        align = mean_cos_align(nxt, self._goal)
        goal_ok = at_goal(nxt, self._goal)
        survive_len = int(self.survival_frac * self.max_steps)
        survived = (~oob) & (~fell) & (self._steps >= survive_len)

        infos: list[dict[str, Any]] = []
        done_np = done.detach().cpu().numpy()
        term_np = terminated.detach().cpu().numpy()
        trunc_np = truncated.detach().cpu().numpy()
        align_np = align.detach().cpu().numpy()
        goal_np = goal_ok.detach().cpu().numpy()
        surv_np = survived.detach().cpu().numpy()
        steps_np = self._steps.detach().cpu().numpy()
        x_np = x.detach().cpu().numpy()
        for i in range(self.num_envs):
            info = {
                "align": float(align_np[i]),
                "at_goal": bool(goal_np[i]),
                "oob": bool(oob[i].item()),
                "fell": bool(fell[i].item()),
                "force": float(force_np[i]),
                "x": float(x_np[i]),
                "ep_len": int(steps_np[i]),
                "survival_success": False,
                "is_success": False,
                "TimeLimit.truncated": bool(trunc_np[i] and not term_np[i]),
            }
            if done_np[i]:
                survived_i = bool(surv_np[i])
                info["survival_success"] = survived_i
                info["is_success"] = survived_i
                info["at_goal_final"] = bool(goal_np[i])
                info["terminal_observation"] = self._obs_np()[i]
                info["episode"] = {
                    "r": float(self._ep_ret[i]),
                    "l": int(steps_np[i]),
                    "t": 0.0,
                }
            infos.append(info)

        if bool(done.any()):
            self._resample(done)

        self._pending_actions = None
        return self._obs_np(), rewards.astype(np.float32), done_np, infos

    def close(self) -> None:
        return None

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        idx = self._cast_idx(indices)
        val = getattr(self, attr_name, None)
        return [val for _ in idx]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        if hasattr(self, attr_name):
            setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        idx = self._cast_idx(indices)
        return [False for _ in idx]

    def _cast_idx(self, indices) -> Sequence[int]:
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return list(indices)


def make_triple_uuu_vec(**kwargs) -> TripleUUUVecEnv:
    return TripleUUUVecEnv(**kwargs)
