"""Target-parameterized MPPI costs for the cart-triple.

Running cost (integrated over time, so weights are per second):
  angle     w_angle * sum(1 - cos e_i)
  spin      (w_vel + w_vel_near * near) * sum(omega_i^2)
  cart      w_x * x^2 + w_xd * xd^2, plus a soft barrier past ``x_soft``
  energy    w_energy * (1 - near) * (E - E_target)^2
  force     w_u * u^2 and w_du * (du/dt)^2 (smoothness)

Terminal cost blends the target's LQR cost-to-go ``e^T P e`` (inside the near
region) with the energy gap (outside it).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from goals_triple import GOAL_ANGLES, parse_goal
from mppi.dynamics_fast import Plant, pole_energy, potential_energy


@dataclass
class CostConfig:
    goal: str = "UUU"
    w_angle: float = 20.0
    w_vel: float = 0.05
    w_vel_near: float = 2.0
    near_scale: float = 0.15
    w_x: float = 1.0
    w_xd: float = 0.1
    x_soft: float = 1.6
    w_barrier: float = 2000.0
    x_dead: float = 2.35
    w_dead: float = 1e5
    w_energy: float = 20.0
    w_u: float = 1e-4
    w_du: float = 1e-6
    w_terminal_lqr: float = 0.1
    w_terminal_energy: float = 50.0
    lqr_q_theta: float = 100.0
    lqr_r: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


@dataclass
class TargetCost:
    plant: Plant
    cfg: CostConfig
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self):
        gi = parse_goal(self.cfg.goal)
        ang = GOAL_ANGLES[gi].to(dtype=torch.float64)
        self.target_angles = ang.to(device=self.device, dtype=torch.float32)
        self.e_target = float(
            potential_energy(self.plant, ang[0], ang[1], ang[2]).item()
        )
        P, K = self._lqr()
        self.P = torch.as_tensor(P, device=self.device, dtype=torch.float32)
        self.K_lqr = torch.as_tensor(K, device=self.device, dtype=torch.float32)

    def _lqr(self) -> tuple[np.ndarray, np.ndarray]:
        from lqr_uuu import design_lqr_uuu

        ctrl = design_lqr_uuu(
            force_limit=self.plant.force_limit,
            q_theta=self.cfg.lqr_q_theta,
            r=self.cfg.lqr_r,
            goal=self.cfg.goal,
        )
        # Per-step DARE cost-to-go -> per-second units of the running cost.
        return ctrl.P * self.plant.dt, ctrl.K.reshape(-1)

    def target_state(self) -> torch.Tensor:
        z = torch.zeros(8, device=self.device)
        z[2], z[4], z[6] = self.target_angles
        return z

    def errors(self, state: torch.Tensor) -> torch.Tensor:
        """8-D error with wrapped angles."""
        t = self.target_angles
        e = state.clone()
        e[..., 2] = wrap(state[..., 2] - t[0])
        e[..., 4] = wrap(state[..., 4] - t[1])
        e[..., 6] = wrap(state[..., 6] - t[2])
        return e

    def nearness(self, state: torch.Tensor) -> torch.Tensor:
        t = self.target_angles
        d = (
            (1 - torch.cos(state[..., 2] - t[0]))
            + (1 - torch.cos(state[..., 4] - t[1]))
            + (1 - torch.cos(state[..., 6] - t[2]))
        )
        return torch.exp(-d / self.cfg.near_scale)

    def running(self, state: torch.Tensor, u: torch.Tensor, du: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        t = self.target_angles
        x, xd = state[..., 0], state[..., 1]
        ang = (
            (1 - torch.cos(state[..., 2] - t[0]))
            + (1 - torch.cos(state[..., 4] - t[1]))
            + (1 - torch.cos(state[..., 6] - t[2]))
        )
        near = torch.exp(-ang / c.near_scale)
        spin = state[..., 3] ** 2 + state[..., 5] ** 2 + state[..., 7] ** 2
        over = torch.relu(x.abs() - c.x_soft)
        dead = (x.abs() > c.x_dead).to(state.dtype)
        energy = pole_energy(self.plant, state) - self.e_target
        cost = (
            c.w_angle * ang
            + (c.w_vel + c.w_vel_near * near) * spin
            + c.w_x * x * x
            + c.w_xd * xd * xd
            + c.w_barrier * over * over
            + c.w_dead * dead
            + c.w_energy * (1 - near) * energy * energy
            + c.w_u * u * u
            + c.w_du * du * du
        )
        return cost * self.plant.dt

    def cost_du(self, du: torch.Tensor) -> torch.Tensor:
        return self.cfg.w_du * (du * du).sum(-1)

    def terminal(self, state: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        e = self.errors(state)
        quad = torch.einsum("...i,ij,...j->...", e, self.P.to(e.dtype), e)
        near = self.nearness(state)
        energy = pole_energy(self.plant, state) - self.e_target
        return near * c.w_terminal_lqr * quad + (1 - near) * c.w_terminal_energy * energy * energy
