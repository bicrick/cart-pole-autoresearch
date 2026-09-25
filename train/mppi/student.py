"""Distilled student: raw 11-D ``observe()`` features -> cart force.

Same structure as the teacher: the target's gated LQR anchor (computed from
sin/cos in the observation) plus an MLP residual in ``+-residual_scale``:

    force = clip(gate(obs) * (-K e(obs)) + residual_scale * tanh(mlp(obs)), +-force_limit)

The MLP only has to learn the teacher's smooth residual; the high-gain part
near the target is exact. The first layer is a fixed input scaling.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from physics_triple import observe

OBS_DIM = 11
# x, xd, sin/cos x3, omega x3
OBS_SCALE = torch.tensor([2.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0])
MIRROR = torch.tensor([-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])


class Anchor(nn.Module):
    """Gated target LQR from the 11-D observation."""

    def __init__(self, K: torch.Tensor, target_angles: torch.Tensor, gate_in: float, gate_out: float):
        super().__init__()
        self.register_buffer("K", K.float().reshape(8))
        self.register_buffer("target", target_angles.float().reshape(3))
        self.gate_in = float(gate_in)
        self.gate_out = float(gate_out)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        errs = []
        for j in range(3):
            s, c = obs[..., 2 + 2 * j], obs[..., 3 + 2 * j]
            ct, st = math.cos(float(self.target[j])), math.sin(float(self.target[j]))
            # angle of (theta - target) from sin/cos
            errs.append(torch.atan2(s * ct - c * st, c * ct + s * st))
        e1, e2, e3 = errs
        d = (1 - torch.cos(e1)) + (1 - torch.cos(e2)) + (1 - torch.cos(e3))
        gate = ((self.gate_out - d) / (self.gate_out - self.gate_in)).clamp(0.0, 1.0)
        k = self.K
        fb = (
            k[0] * obs[..., 0] + k[1] * obs[..., 1] + k[2] * e1 + k[3] * obs[..., 8]
            + k[4] * e2 + k[5] * obs[..., 9] + k[6] * e3 + k[7] * obs[..., 10]
        )
        return -gate * fb

    def spec(self) -> dict:
        return {
            "K": self.K.tolist(),
            "target": self.target.tolist(),
            "gate_in": self.gate_in,
            "gate_out": self.gate_out,
        }


class Student(nn.Module):
    def __init__(self, anchor: Anchor, hidden: tuple[int, ...] = (256, 256), force_limit: float = 40.0):
        super().__init__()
        self.anchor = anchor
        self.hidden = tuple(hidden)
        self.force_limit = float(force_limit)
        self.residual_scale = 2.0 * self.force_limit
        scale = nn.Linear(OBS_DIM, OBS_DIM)
        with torch.no_grad():
            scale.weight.copy_(torch.diag(1.0 / OBS_SCALE))
            scale.bias.zero_()
        scale.requires_grad_(False)
        layers: list[nn.Module] = [scale]
        d = OBS_DIM
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalized residual in [-1, 1]."""
        return torch.tanh(self.net(obs).squeeze(-1))

    def residual(self, states: torch.Tensor) -> torch.Tensor:
        return self.forward(observe(states.float())) * self.residual_scale

    def force(self, states: torch.Tensor) -> torch.Tensor:
        obs = observe(states.float())
        u = self.anchor(obs) + self.forward(obs) * self.residual_scale
        return u.clamp(-self.force_limit, self.force_limit)


def from_teacher(teacher, hidden: tuple[int, ...]) -> Student:
    c = teacher.cfg
    anchor = Anchor(teacher.cost.K_lqr, teacher.cost.target_angles, c.gate_in, c.gate_out)
    return Student(anchor, hidden, teacher.fmax)


class StudentController:
    """Episode-harness wrapper (``reset`` / ``act``)."""

    def __init__(self, model: Student):
        self.model = model.eval()

    def reset(self, batch: int) -> None:
        pass

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        return self.model.force(states)


def mirror(obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return obs * MIRROR.to(obs.device), -act


def save(model: Student, path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": model.hidden,
            "force_limit": model.force_limit,
            "anchor": model.anchor.spec(),
            "meta": meta or {},
        },
        path,
    )


def load(path: str | Path) -> tuple[Student, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["anchor"]
    anchor = Anchor(torch.tensor(a["K"]), torch.tensor(a["target"]), a["gate_in"], a["gate_out"])
    model = Student(anchor, tuple(ck["hidden"]), ck["force_limit"])
    model.load_state_dict(ck["state_dict"])
    return model, ck.get("meta", {})
