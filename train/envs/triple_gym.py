"""Gymnasium wrapper: cart-triple plant, UUU specialist (Lim / fawraw M2 path).

Observation: 11-D state features (no goal one-hot; goal fixed UUU).
Action: Box(-1, 1) scaled by forceLimit (N).
Reward: Lim/Baek product_reward toward UUU.
ICs: quiet-basin near_target — init_noise scales θ, ω, x/xd (floors 1e-3 only).
Walls default ON for our plant (Lim/fawraw use rail limits; we use inelastic walls).
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

from goals_triple import GOAL_INDEX, at_goal, compute_reward, mean_cos_align  # noqa: E402
from physics_triple import load_constants, observe, step  # noqa: E402

import gymnasium as gym
from gymnasium import spaces

STATE_DIM = 11
UUU_ID = int(GOAL_INDEX["UUU"])


def _sample_state(
    init_mode: str,
    hang_frac: float,
    wide_frac: float,
    noise: float,
    device: torch.device,
) -> torch.Tensor:
    """Single-env IC for UUU specialist.

    Default mass is near-upright (hold basin). With hang_frac spawn near DDD;
    with wide_frac use Lim-style wide random ICs.
    """
    mode = (init_mode or "near_target").lower()
    u = float(torch.rand((), device=device))
    if u < hang_frac or mode in ("bottom", "hang"):
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
        # Quiet-basin (fawraw M2): noise scales angle AND ω AND x/xd proportionally.
        # Legacy fixed ±0.5 cart / ±0.8 ω ignored init_noise — made "tighten" a no-op.
        # Tiny absolute floors 1e-3 only (see train_triple.random_states near_target).
        n_ang = max(float(noise), 1e-3)
        n_x = min(0.5, max(1e-3, n_ang * 2.0))
        n_xd = min(0.5, max(1e-3, n_ang * 2.0))
        n_w = min(0.8, max(1e-3, n_ang * 5.0))
        x = torch.empty((), device=device).uniform_(-n_x, n_x)
        xd = torch.empty((), device=device).uniform_(-n_xd, n_xd)
        th1 = torch.empty((), device=device).uniform_(-n_ang, n_ang)
        th2 = torch.empty((), device=device).uniform_(-n_ang, n_ang)
        th3 = torch.empty((), device=device).uniform_(-n_ang, n_ang)
        th1d = torch.empty((), device=device).uniform_(-n_w, n_w)
        th2d = torch.empty((), device=device).uniform_(-n_w, n_w)
        th3d = torch.empty((), device=device).uniform_(-n_w, n_w)
    return torch.stack((x, xd, th1, th1d, th2, th2d, th3, th3d))


class TriplePendulumUUUEnv(gym.Env):
    """Single-env Gymnasium adapter for UUU-only TQC (Lim EP7 specialist)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        force_limit: float = 40.0,
        max_steps: int = 1000,
        track_limit: float = 2.4,
        track_walls: bool = True,
        init_mode: str = "near_target",
        init_noise: float = 0.05,
        hang_frac: float = 0.0,
        wide_frac: float = 0.0,
        progress_w: float = 1.0,
        cart_barrier_coef: float = 10.0,
        alpha_th: float = 0.5,
        w_up: float = 5.0,
        w_down: float = 1.0,
        energy_w: float = 0.0,
        oob_penalty: float = 20.0,
        device: str = "cpu",
        seed: Optional[int] = None,
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
        self.hang_frac = float(hang_frac)
        self.wide_frac = float(wide_frac)
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
        )

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
        self._goal = torch.tensor(UUU_ID, device=self.device, dtype=torch.long)
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
        self._state = _sample_state(
            mode,
            hang_frac=float(opts.get("hang_frac", self.hang_frac)),
            wide_frac=float(opts.get("wide_frac", self.wide_frac)),
            noise=float(opts.get("init_noise", self.init_noise)),
            device=self.device,
        )
        self._prev_state = self._state.clone()
        self._step_count = 0
        info = {
            "align": float(
                mean_cos_align(self._state.unsqueeze(0), self._goal.unsqueeze(0)).item()
            ),
            "at_goal": bool(
                at_goal(self._state.unsqueeze(0), self._goal.unsqueeze(0)).item()
            ),
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
            reward_mode="product",
            prev_state=prev.unsqueeze(0),
            **self._reward_kwargs,
        )
        reward = float(rew_t.item())

        x = float(nxt[0].item())
        # Walls clamp at trackLimit; only treat as void if somehow past bumper.
        if self.track_walls:
            oob = abs(x) > (self.track_limit + 1e-4)
        else:
            oob = abs(x) > self.track_limit
        align = float(
            mean_cos_align(nxt.unsqueeze(0), self._goal.unsqueeze(0)).item()
        )
        success = bool(at_goal(nxt.unsqueeze(0), self._goal.unsqueeze(0)).item())
        truncated = self._step_count >= self.max_steps
        terminated = bool(oob)
        info = {
            "align": align,
            "at_goal": success,
            "oob": oob,
            "force": force_val,
            "x": x,
            "is_success": success,
        }
        return self._obs_np(), reward, terminated, truncated, info


def make_triple_uuu_env(**kwargs) -> TriplePendulumUUUEnv:
    return TriplePendulumUUUEnv(**kwargs)
