"""Two-policy swing↔hold handoff for cart-triple UUU (P1 staging).

Enter:  max|φ_i| < capture_tol AND max|ω_i| < capture_vel  (+ optional dwell)
Exit:   max|φ_i| > exit_tol (hysteresis) after dwell — not one-way latch alone
Latch:  once holding, stay on catcher until exit fires
LPF:    first-order force filter τ≈0.3 s on the emitted force (soft landing)

Catcher / swing are callables: predict(state_8d) -> action in [-1,1] (Gym).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

ActionFn = Callable[[np.ndarray], np.ndarray]


def angle_err(state: np.ndarray) -> np.ndarray:
    """Principal angles of θ1,θ2,θ3 (indices 2,4,6)."""
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    th = np.array([z[2], z[4], z[6]], dtype=np.float64)
    return (th + np.pi) % (2.0 * np.pi) - np.pi


def omega_abs_max(state: np.ndarray) -> float:
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs([z[3], z[5], z[7]])))


def in_enter_basin(
    state: np.ndarray,
    *,
    # UUU keeper zip, 2026-09-21: θ=0.03 and ω≤0.10 stays; ω=0.5 falls.
    # fawraw's 0.1 rad / 1 rad/s is their plant.
    capture_tol_rad: float = 0.03,
    capture_vel_rad_s: float = 0.10,
) -> bool:
    err = angle_err(state)
    return bool(np.max(np.abs(err)) < capture_tol_rad and omega_abs_max(state) < capture_vel_rad_s)


def in_exit_band(
    state: np.ndarray,
    *,
    exit_tol_rad: float = 0.25,
) -> bool:
    return bool(np.max(np.abs(angle_err(state))) > exit_tol_rad)


@dataclass
class ForceLPF:
    """y += (dt/τ) * (u - y); τ≈0.3 s default (arXiv:2606.22145)."""

    tau_s: float = 0.3
    y: float = 0.0

    def reset(self) -> None:
        self.y = 0.0

    def __call__(self, u: float, dt: float) -> float:
        if self.tau_s <= 0:
            self.y = float(u)
            return self.y
        a = float(dt) / float(self.tau_s)
        a = min(max(a, 0.0), 1.0)
        self.y = self.y + a * (float(u) - self.y)
        return self.y


@dataclass
class TwoPolicyHandoff:
    swing: ActionFn
    catcher: ActionFn
    capture_tol_rad: float = 0.03
    capture_vel_rad_s: float = 0.10
    exit_tol_rad: float = 0.25
    enter_dwell: int = 5
    exit_dwell: int = 5
    latch: bool = True
    force_limit: float = 40.0
    dt: float = 1.0 / 120.0
    lpf_tau_s: float = 0.3
    holding: bool = False
    _enter_count: int = 0
    _exit_count: int = 0
    _lpf: ForceLPF = field(default_factory=ForceLPF)

    def __post_init__(self) -> None:
        self._lpf = ForceLPF(tau_s=self.lpf_tau_s)
        self.reset()

    def reset(self) -> None:
        self.holding = False
        self._enter_count = 0
        self._exit_count = 0
        self._lpf.reset()

    def _update_mode(self, state: np.ndarray) -> None:
        if self.holding:
            if in_exit_band(state, exit_tol_rad=self.exit_tol_rad):
                self._exit_count += 1
            else:
                self._exit_count = 0
            if self._exit_count >= self.exit_dwell:
                self.holding = False
                self._enter_count = 0
                self._exit_count = 0
            return

        if in_enter_basin(
            state,
            capture_tol_rad=self.capture_tol_rad,
            capture_vel_rad_s=self.capture_vel_rad_s,
        ):
            self._enter_count += 1
        else:
            self._enter_count = 0
        if self._enter_count >= self.enter_dwell:
            self.holding = True
            self._exit_count = 0

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return Gym action [-1,1]; applies LPF on the Newtons-scale force."""
        del deterministic  # API parity with SB3
        z = np.asarray(state, dtype=np.float64).reshape(-1)[:8]
        self._update_mode(z)
        raw = self.catcher(z) if self.holding else self.swing(z)
        a = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
        a = float(np.clip(a, -1.0, 1.0))
        force = a * self.force_limit
        force_f = self._lpf(force, self.dt)
        return np.array([force_f / self.force_limit], dtype=np.float32)

    def mode(self) -> str:
        return "hold" if self.holding else "swing"


def wrap_tqc_catcher(model, force_limit: float = 40.0) -> ActionFn:
    """SB3 TQC model: needs 11-D obs from observe(state)."""
    from physics_triple import observe
    import torch

    def _fn(state: np.ndarray) -> np.ndarray:
        z = torch.as_tensor(state, dtype=torch.float32).reshape(1, 8)
        obs = observe(z).detach().cpu().numpy().astype(np.float32)
        act, _ = model.predict(obs, deterministic=True)
        return np.asarray(act, dtype=np.float32).reshape(-1)

    return _fn


def wrap_lqr_catcher(lqr) -> ActionFn:
    return lambda state: lqr.action_normed(state)


def zero_swing(_: np.ndarray) -> np.ndarray:
    """Placeholder swing (no thrust) for basin / LQR-only smoke tests."""
    return np.array([0.0], dtype=np.float32)


def wrap_ppo_policy(checkpoint: str, *, force_limit: float = 40.0, goal: str = "UUU", device: str = "cpu"):
    """Load PPO ActorCritic .pt and return ActionFn -> Gym [-1,1]."""
    from pathlib import Path

    import torch

    from goals_triple import GOAL_INDEX, conditioned_obs
    from physics_triple import load_constants, normalize_obs, observe
    from ppo import ActorCritic, tanh_action

    constants = dict(load_constants())
    constants["forceLimit"] = float(force_limit)
    path = Path(checkpoint)
    payload = torch.load(path, map_location=device, weights_only=False)
    obs_dim = int(payload.get("obs_dim", 25)) if isinstance(payload, dict) else 25
    model = ActorCritic(obs_dim=obs_dim, hidden=constants["hidden"]).to(device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict)
    model.eval()
    gid = int(GOAL_INDEX[goal])

    def _fn(state) -> "np.ndarray":
        z = torch.as_tensor(state, dtype=torch.float32, device=device).reshape(1, 8)
        goals = torch.full((1,), gid, device=device, dtype=torch.long)
        obs = normalize_obs(conditioned_obs(observe(z), goals), constants)
        with torch.no_grad():
            raw = model.deterministic(obs)
            force = tanh_action(raw, constants["forceLimit"]).reshape(-1)[0]
        a = float(force.item()) / float(force_limit)
        return np.array([np.clip(a, -1.0, 1.0)], dtype=np.float32)

    return _fn
