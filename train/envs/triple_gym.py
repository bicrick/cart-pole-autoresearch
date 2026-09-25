"""Gymnasium wrapper: cart-triple plant, one Lim specialist (fawraw M2 path).

Observation: 11-D state features (no goal one-hot; goal fixed per env).
Action: Box(-1, 1) scaled by forceLimit (N).
Reward: Lim/Baek product_reward toward that EP.

Default goal is UUU (Lim EP7). Pass goal='DDD' for the next specialist.
Do not add a one-hot / random-target UVFA (fawraw M3 — retired here).

Env contract (mirror fawraw TriplePendulumEnv near_target / M2 hold):
  - Quiet-basin ICs: angles ~ target ±init_noise; cart x ~ ±init_noise;
    rates (xd, ω) fixed ±0.01 — NOT init_noise×5 / ×2.
  - Fall-kill: UP-target |θ_err| > 0.6, DOWN-target |θ_err| > 1.5.
  - progress_w default 0 (hold recipe; no swing progress term).
  - Primary M2 success = survival (ep_len >= 0.8 * max_steps without fall/oob);
    also log final at_goal.
Walls default ON for our plant (inelastic endstops).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

_TRAIN = Path(__file__).resolve().parents[1]
if str(_TRAIN) not in sys.path:
    sys.path.insert(0, str(_TRAIN))

from goals_triple import (  # noqa: E402
    GOAL_ANGLES,
    GOAL_IDS,
    GOAL_INDEX,
    at_goal,
    compute_reward,
    goal_angles,
    mean_cos_align,
    parse_goal,
    wrapped_angle_error,
)
from hard_ics import load_bank, mix_hard_ics  # noqa: E402
from physics_triple import load_constants, observe, step  # noqa: E402

import gymnasium as gym
from gymnasium import spaces

STATE_DIM = 11
UUU_ID = int(GOAL_INDEX["UUU"])
DDD_ID = int(GOAL_INDEX["DDD"])

# fawraw FALL_THRESHOLD_UP_RAD / DOWN.
FALL_THRESH_UP = 0.6
FALL_THRESH_DOWN = 1.5
# Fixed quiet rates (fawraw: qvel[:] = uniform(-0.01, 0.01)).
QUIET_RATE = 0.01
# Survival success fraction of max_steps (primary M2 hold metric).
SURVIVAL_FRAC = 0.8


def _sample_state(
    init_mode: str,
    hang_frac: float,
    wide_frac: float,
    noise: float,
    device: torch.device,
    goal_id: int = UUU_ID,
) -> torch.Tensor:
    """Single-env IC for one Lim specialist.

    Default mass is near-target quiet basin (fawraw M2):
      θ ~ target ±init_noise, x ~ U(±init_noise), xd/ω ~ U(±0.01) fixed.
    """
    mode = (init_mode or "near_target").lower()
    u = float(torch.rand((), device=device))
    if mode == "bottom":
        # fawraw M4 bottom: absolute DDD ± init_noise, every rate ±0.01.
        # Not the wide hang box. Angle-fall is off for this mode (see trainer).
        n = max(float(noise), 1e-3)
        n_rate = float(QUIET_RATE)
        x = torch.empty((), device=device).uniform_(-n, n)
        xd = torch.empty((), device=device).uniform_(-n_rate, n_rate)
        th1 = math.pi + torch.empty((), device=device).uniform_(-n, n)
        th2 = math.pi + torch.empty((), device=device).uniform_(-n, n)
        th3 = math.pi + torch.empty((), device=device).uniform_(-n, n)
        th1d = torch.empty((), device=device).uniform_(-n_rate, n_rate)
        th2d = torch.empty((), device=device).uniform_(-n_rate, n_rate)
        th3d = torch.empty((), device=device).uniform_(-n_rate, n_rate)
    elif u < hang_frac or mode == "hang":
        x = torch.empty((), device=device).uniform_(-1.0, 1.0)
        xd = torch.empty((), device=device).uniform_(-0.8, 0.8)
        th1 = math.pi + torch.empty((), device=device).uniform_(-0.35, 0.35)
        th2 = math.pi + torch.empty((), device=device).uniform_(-0.35, 0.35)
        th3 = math.pi + torch.empty((), device=device).uniform_(-0.35, 0.35)
        th1d = torch.empty((), device=device).uniform_(-1.0, 1.0)
        th2d = torch.empty((), device=device).uniform_(-1.0, 1.0)
        th3d = torch.empty((), device=device).uniform_(-1.0, 1.0)
    elif u < hang_frac + wide_frac or mode == "wide":
        # Lim KIEE 2025 IC ranges
        x = torch.empty((), device=device).uniform_(-0.3, 0.3)
        xd = torch.empty((), device=device).uniform_(-1.2, 1.2)
        th1 = torch.empty((), device=device).uniform_(-math.pi, math.pi)
        th2 = torch.empty((), device=device).uniform_(-math.pi, math.pi)
        th3 = torch.empty((), device=device).uniform_(-math.pi, math.pi)
        th1d = torch.empty((), device=device).uniform_(-10.0, 10.0)
        th2d = torch.empty((), device=device).uniform_(-20.0, 20.0)
        th3d = torch.empty((), device=device).uniform_(-30.0, 30.0)
    else:
        # Quiet-basin (fawraw M2 near_target): angles/cart scale with init_noise;
        # rates FIXED ±QUIET_RATE (was wrongly init_noise×5 / ×2 — blew the basin).
        n_ang = max(float(noise), 1e-3)
        n_x = n_ang  # fawraw: qpos[0] = uniform(-n, n)
        n_rate = float(QUIET_RATE)
        tgt = GOAL_ANGLES[int(goal_id)].to(device=device)
        x = torch.empty((), device=device).uniform_(-n_x, n_x)
        xd = torch.empty((), device=device).uniform_(-n_rate, n_rate)
        th1 = tgt[0] + torch.empty((), device=device).uniform_(-n_ang, n_ang)
        th2 = tgt[1] + torch.empty((), device=device).uniform_(-n_ang, n_ang)
        th3 = tgt[2] + torch.empty((), device=device).uniform_(-n_ang, n_ang)
        th1d = torch.empty((), device=device).uniform_(-n_rate, n_rate)
        th2d = torch.empty((), device=device).uniform_(-n_rate, n_rate)
        th3d = torch.empty((), device=device).uniform_(-n_rate, n_rate)
    return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d))


def angle_fall_mask(
    state: torch.Tensor,
    goal_id: int | torch.Tensor = UUU_ID,
    thresh_up: float = FALL_THRESH_UP,
    thresh_down: float = FALL_THRESH_DOWN,
) -> torch.Tensor:
    """Per-row True if any link |θ − target| exceeds the UP/DOWN thresh."""
    th = state[..., [2, 4, 6]]
    if torch.is_tensor(goal_id) and goal_id.numel() > 1:
        tgt = goal_angles(goal_id, device=th.device, dtype=th.dtype)
    else:
        gid = int(goal_id.reshape(-1)[0].item()) if torch.is_tensor(goal_id) else int(goal_id)
        tgt = GOAL_ANGLES[gid].to(device=th.device, dtype=th.dtype)
        while tgt.ndim < th.ndim:
            tgt = tgt.unsqueeze(0)
        tgt = tgt.expand_as(th)
    err = torch.atan2(torch.sin(th - tgt), torch.cos(th - tgt)).abs()
    is_up = tgt.abs() < 0.5
    thresh = torch.where(
        is_up,
        torch.full_like(err, float(thresh_up)),
        torch.full_like(err, float(thresh_down)),
    )
    return (err > thresh).any(dim=-1)


def uuu_angle_fall_mask(state: torch.Tensor, thresh: float = FALL_THRESH_UP) -> torch.Tensor:
    """Per-row True if any UUU (UP) link |θ| exceeds thresh (wrapped)."""
    return angle_fall_mask(state, UUU_ID, thresh_up=thresh, thresh_down=FALL_THRESH_DOWN)


def _uuu_angle_fall(state: torch.Tensor, thresh: float = FALL_THRESH_UP) -> bool:
    """True if any UUU (UP) link |θ| exceeds thresh (wrapped)."""
    return bool(uuu_angle_fall_mask(state, thresh).any().item())


def sample_states(
    n: int,
    init_mode: str,
    hang_frac: float,
    wide_frac: float,
    noise: float,
    device: torch.device,
    noise_min: float = 0.0,
    goal_id: int = UUU_ID,
    arrival_frac: float = 0.0,
    arrival_omega: float = 0.0,
) -> torch.Tensor:
    """Batched ICs. Default mass is fawraw M2 quiet basin (rates fixed ±0.01)."""
    n = int(n)
    mode = (init_mode or "near_target").lower()
    u = torch.rand(n, device=device)
    if mode == "bottom":
        # fawraw M4: DDD ± init_noise, rates ±0.01. See _sample_state.
        n_ang = max(float(noise), 1e-3)
        n_rate = float(QUIET_RATE)
        state = torch.empty(n, 8, device=device)
        state[:, 0] = torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        state[:, 1] = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        state[:, 2] = math.pi + torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        state[:, 3] = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        state[:, 4] = math.pi + torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        state[:, 5] = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        state[:, 6] = math.pi + torch.empty(n, device=device).uniform_(-n_ang, n_ang)
        state[:, 7] = torch.empty(n, device=device).uniform_(-n_rate, n_rate)
        # Half the swing episodes (when set) start near the target with a
        # shared link rate, so the net practices bleeding the flyby. Bottom
        # rows stay fawraw M4. Holds leave arrival_frac at 0.
        frac = float(arrival_frac)
        w_max = float(arrival_omega)
        if frac > 0.0 and w_max > 0.0:
            take = torch.rand(n, device=device) < frac
            k = int(take.sum().item())
            if k:
                tgt = GOAL_ANGLES[int(goal_id)].to(device=device)
                n_pos = 0.05
                mag = torch.empty(k, device=device).uniform_(0.05, w_max)
                sign = torch.where(
                    torch.rand(k, device=device) < 0.5,
                    torch.ones(k, device=device),
                    -torch.ones(k, device=device),
                )
                om = mag * sign
                state[take, 0] = torch.empty(k, device=device).uniform_(-n_pos, n_pos)
                state[take, 1] = torch.empty(k, device=device).uniform_(-0.05, 0.05)
                state[take, 2] = tgt[0] + torch.empty(k, device=device).uniform_(-n_pos, n_pos)
                state[take, 4] = tgt[1] + torch.empty(k, device=device).uniform_(-n_pos, n_pos)
                state[take, 6] = tgt[2] + torch.empty(k, device=device).uniform_(-n_pos, n_pos)
                state[take, 3] = om
                state[take, 5] = om
                state[take, 7] = om
        return state
    hang = (u < hang_frac) | (mode == "hang")
    wide = (~hang) & ((u < hang_frac + wide_frac) | (mode == "wide"))
    near = (~hang) & (~wide)

    state = torch.empty(n, 8, device=device)
    if hang.any():
        k = int(hang.sum().item())
        state[hang, 0] = torch.empty(k, device=device).uniform_(-1.0, 1.0)
        state[hang, 1] = torch.empty(k, device=device).uniform_(-0.8, 0.8)
        state[hang, 2] = math.pi + torch.empty(k, device=device).uniform_(-0.35, 0.35)
        state[hang, 3] = torch.empty(k, device=device).uniform_(-1.0, 1.0)
        state[hang, 4] = math.pi + torch.empty(k, device=device).uniform_(-0.35, 0.35)
        state[hang, 5] = torch.empty(k, device=device).uniform_(-1.0, 1.0)
        state[hang, 6] = math.pi + torch.empty(k, device=device).uniform_(-0.35, 0.35)
        state[hang, 7] = torch.empty(k, device=device).uniform_(-1.0, 1.0)
    if wide.any():
        k = int(wide.sum().item())
        state[wide, 0] = torch.empty(k, device=device).uniform_(-0.3, 0.3)
        state[wide, 1] = torch.empty(k, device=device).uniform_(-1.2, 1.2)
        state[wide, 2] = torch.empty(k, device=device).uniform_(-math.pi, math.pi)
        state[wide, 3] = torch.empty(k, device=device).uniform_(-10.0, 10.0)
        state[wide, 4] = torch.empty(k, device=device).uniform_(-math.pi, math.pi)
        state[wide, 5] = torch.empty(k, device=device).uniform_(-20.0, 20.0)
        state[wide, 6] = torch.empty(k, device=device).uniform_(-math.pi, math.pi)
        state[wide, 7] = torch.empty(k, device=device).uniform_(-30.0, 30.0)
    if near.any():
        k = int(near.sum().item())
        hi = max(float(noise), 1e-3)
        lo = float(noise_min)
        if 0.0 < lo < hi:
            n_ang = torch.empty(k, device=device).uniform_(lo, hi)
        else:
            n_ang = torch.full((k,), hi, device=device)
        n_rate = float(QUIET_RATE)
        u = torch.rand(k, 4, device=device) * 2.0 - 1.0
        tgt = GOAL_ANGLES[int(goal_id)].to(device=device)
        state[near, 0] = u[:, 0] * n_ang
        state[near, 1] = torch.empty(k, device=device).uniform_(-n_rate, n_rate)
        state[near, 2] = tgt[0] + u[:, 1] * n_ang
        state[near, 3] = torch.empty(k, device=device).uniform_(-n_rate, n_rate)
        state[near, 4] = tgt[1] + u[:, 2] * n_ang
        state[near, 5] = torch.empty(k, device=device).uniform_(-n_rate, n_rate)
        state[near, 6] = tgt[2] + u[:, 3] * n_ang
        state[near, 7] = torch.empty(k, device=device).uniform_(-n_rate, n_rate)
    return state


class TriplePendulumUUUEnv(gym.Env):
    """Single-env Gymnasium adapter for one Lim TQC specialist (default UUU)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
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
        ry_scale: float | None = None,
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
        soft_landing_rad: float = 0.0,
        vel_near_gain: float = 1.0,
        vel_near_rad: float = 0.3,
        excess_energy_coef: float = 0.0,
        arrival_frac: float = 0.0,
        arrival_omega: float = 0.0,
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
        super().__init__()
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
        self.hard_ic_frac = float(hard_ic_frac)
        self._hard_bank = (
            load_bank(hard_ic_path, self.device) if hard_ic_path else None
        )
        self._reward_kwargs = dict(
            track_limit=self.track_limit,
            oob_penalty=float(oob_penalty),
            progress_w=float(progress_w),
            ry_scale=None if ry_scale is None else float(ry_scale),
            cart_barrier_coef=float(cart_barrier_coef),
            alpha_th=float(alpha_th),
            w_up=float(w_up),
            w_down=float(w_down),
            energy_w=float(energy_w),
            sparse_bonus=float(sparse_bonus),
            vel_cost_coef=float(vel_cost_coef),
            cart_cost_coef=float(cart_cost_coef),
            soft_landing_rad=float(soft_landing_rad),
            vel_near_gain=float(vel_near_gain),
            vel_near_rad=float(vel_near_rad),
            excess_energy_coef=float(excess_energy_coef),
        )
        self.reward_mode = (reward_mode or "product").lower()
        self.transition_bonus = float(transition_bonus)
        self.transition_tol = float(transition_tol)
        self.transition_steps = int(transition_steps)
        self.arrival_frac = float(arrival_frac)
        self.arrival_omega = float(arrival_omega)
        self._in_tol = 0
        self._bonus_paid = False

        high = np.array(
            [4.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._state: Optional[torch.Tensor] = None
        self._prev_state: Optional[torch.Tensor] = None
        self._step_count = 0
        self._goal = torch.tensor(self.goal_id, device=self.device, dtype=torch.long)
        if seed is not None:
            self.reset(seed=seed)

    def _obs_np(self) -> np.ndarray:
        assert self._state is not None
        obs = observe(self._state.unsqueeze(0)).squeeze(0)
        return obs.detach().cpu().numpy().astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        opts = options or {}
        mode = opts.get("init_mode", self.init_mode)
        self._state = sample_states(
            1,
            mode,
            hang_frac=float(opts.get("hang_frac", self.hang_frac)),
            wide_frac=float(opts.get("wide_frac", self.wide_frac)),
            noise=float(opts.get("init_noise", self.init_noise)),
            device=self.device,
            noise_min=float(opts.get("init_noise_min", self.init_noise_min)),
            goal_id=self.goal_id,
            arrival_frac=self.arrival_frac,
            arrival_omega=self.arrival_omega,
        )
        if self._hard_bank is not None:
            self._state = mix_hard_ics(self._state, self._hard_bank, self.hard_ic_frac)
        self._state = self._state.squeeze(0)
        self._prev_state = self._state.clone()
        self._step_count = 0
        self._in_tol = 0
        self._bonus_paid = False
        info = {
            "align": float(
                mean_cos_align(self._state.unsqueeze(0), self._goal.unsqueeze(0)).item()
            ),
            "at_goal": bool(
                at_goal(self._state.unsqueeze(0), self._goal.unsqueeze(0)).item()
            ),
            "survival_success": False,
            "is_success": False,
        }
        return self._obs_np(), info

    def step(self, action: Any):
        assert self._state is not None
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        raw = float(np.clip(a[0], -1.0, 1.0))
        force_val = raw * self.force_limit
        force = torch.tensor([force_val], device=self.device, dtype=torch.float32)

        prev = self._state
        nxt = step(prev.unsqueeze(0), force, constants=self.constants).squeeze(0)
        self._prev_state = prev
        self._state = nxt
        self._step_count += 1

        rew_t = compute_reward(
            nxt.unsqueeze(0),
            force,
            self._goal.unsqueeze(0),
            self.constants,
            reward_mode=self.reward_mode,
            prev_state=prev.unsqueeze(0),
            **self._reward_kwargs,
        )
        reward = float(rew_t.item())
        if self.reward_mode in ("m4", "m4_swing") and self.transition_bonus > 0:
            err = wrapped_angle_error(nxt.unsqueeze(0), self._goal.unsqueeze(0))
            if bool((err.abs() < self.transition_tol).all().item()):
                self._in_tol += 1
            else:
                self._in_tol = 0
            if self._in_tol >= self.transition_steps and not self._bonus_paid:
                reward += self.transition_bonus
                self._bonus_paid = True

        x = float(nxt[0].item())
        # Walls clamp at trackLimit; only treat as void if somehow past bumper.
        if self.track_walls:
            oob = abs(x) > (self.track_limit + 1e-4)
        else:
            oob = abs(x) > self.track_limit
        fell = bool(
            self.angle_fall
            and angle_fall_mask(
                nxt.unsqueeze(0),
                self.goal_id,
                self.fall_thresh_up,
                self.fall_thresh_down,
            ).any().item()
        )
        align = float(
            mean_cos_align(nxt.unsqueeze(0), self._goal.unsqueeze(0)).item()
        )
        goal_ok = bool(at_goal(nxt.unsqueeze(0), self._goal.unsqueeze(0)).item())
        truncated = self._step_count >= self.max_steps
        terminated = bool(oob or fell)
        # Primary M2 success = survival hold (fawraw-style); also report at_goal.
        survival_success = (not terminated) and (
            self._step_count >= int(self.survival_frac * self.max_steps)
        )
        # On early fall/oob, survival is False; on truncate at max_steps, True.
        if truncated and not terminated:
            survival_success = True
        info = {
            "align": align,
            "at_goal": goal_ok,
            "oob": oob,
            "fell": fell,
            "force": force_val,
            "x": x,
            "ep_len": self._step_count,
            "survival_success": bool(survival_success and (truncated or terminated)),
            # EvalCallback / AtGoalRateCallback primary: survival for M2 hold.
            "is_success": bool(
                survival_success if (truncated or terminated) else False
            ),
        }
        # Only mark success flags meaningful at episode end.
        if not (truncated or terminated):
            info["is_success"] = False
            info["survival_success"] = False
        else:
            # At episode end: survival if lasted long enough without fall/oob.
            survived = (not oob) and (not fell) and (
                self._step_count >= int(self.survival_frac * self.max_steps)
            )
            info["survival_success"] = survived
            info["is_success"] = survived  # PRIMARY for M2
            info["at_goal_final"] = goal_ok
        return self._obs_np(), reward, terminated, truncated, info


def make_triple_uuu_env(**kwargs) -> TriplePendulumUUUEnv:
    return TriplePendulumUUUEnv(**kwargs)
