"""Learned MPPI plan update: (observation, shifted plan) -> new plan.

The MPPI teacher's action is not a function of the state alone: it follows a
45-knot residual plan that it shifts and refines at every knot, and the same
state gets very different forces under different plans. ``PlanNet`` distills
that update operator instead of a memoryless policy, so its input carries the
same information the teacher's does:

    plan_k = clip(warm + delta_scale * mlp(obs, warm / plan_scale), +-2 f_max)
    force  = clip(gate(obs) * (-K e(obs)) + plan_k[0], +-f_max)

where ``warm`` is the previous plan shifted by one knot (zeros after a reset).
The gated LQR anchor is the teacher's, so the catch near the target is exact.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from mppi.student import MIRROR, OBS_DIM, OBS_SCALE, Anchor
from physics_triple import observe

PLAN_SCALE = 20.0


class PlanNet(nn.Module):
    def __init__(self, anchor: Anchor, n_knots: int, knot: int, hidden=(256, 256), force_limit: float = 40.0):
        super().__init__()
        self.anchor = anchor
        self.n_knots = int(n_knots)
        self.knot = int(knot)
        self.hidden = tuple(hidden)
        self.force_limit = float(force_limit)
        self.plan_limit = 2.0 * self.force_limit
        d_in = OBS_DIM + self.n_knots
        scale = nn.Linear(d_in, d_in)
        with torch.no_grad():
            inv = torch.cat((1.0 / OBS_SCALE, torch.full((self.n_knots,), 1.0 / PLAN_SCALE)))
            scale.weight.copy_(torch.diag(inv))
            scale.bias.zero_()
        scale.requires_grad_(False)
        layers: list[nn.Module] = [scale]
        d = d_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        out = nn.Linear(d, self.n_knots)
        with torch.no_grad():
            out.weight.mul_(0.1)
            out.bias.zero_()
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def delta(self, obs: torch.Tensor, warm: torch.Tensor) -> torch.Tensor:
        """Plan change in units of ``PLAN_SCALE`` newtons."""
        return self.net(torch.cat((obs, warm), -1))

    def forward(self, obs: torch.Tensor, warm: torch.Tensor) -> torch.Tensor:
        plan = warm + PLAN_SCALE * self.delta(obs, warm)
        return plan.clamp(-self.plan_limit, self.plan_limit)

    def force(self, obs: torch.Tensor, plan: torch.Tensor) -> torch.Tensor:
        return (self.anchor(obs) + plan[..., 0]).clamp(-self.force_limit, self.force_limit)


def shift(plan: torch.Tensor) -> torch.Tensor:
    return torch.cat((plan[..., 1:], torch.zeros_like(plan[..., -1:])), -1)


def mirror(obs: torch.Tensor, warm: torch.Tensor, plan: torch.Tensor):
    return obs * MIRROR.to(obs.device), -warm, -plan


def from_teacher(teacher, hidden) -> PlanNet:
    c = teacher.cfg
    anchor = Anchor(teacher.cost.K_lqr, teacher.cost.target_angles, c.gate_in, c.gate_out)
    return PlanNet(anchor, c.n_knots, c.knot, hidden, teacher.fmax)


class PlanController:
    """Episode-harness wrapper (``reset`` / ``act``): replans every ``knot`` steps."""

    def __init__(self, model: PlanNet):
        self.model = model.eval()
        self.plan: torch.Tensor | None = None
        self._tick = 0

    def reset(self, batch: int) -> None:
        self.plan = torch.zeros(batch, self.model.n_knots)
        self._tick = 0

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        obs = observe(states.float())
        if self.plan is None or self.plan.shape[0] != obs.shape[0]:
            self.reset(obs.shape[0])
        if self._tick % self.model.knot == 0:
            warm = self.plan if self._tick == 0 else shift(self.plan)
            self.plan = self.model(obs, warm)
        self._tick += 1
        return self.model.force(obs, self.plan)


def save(model: PlanNet, path: str | Path, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": model.hidden,
            "n_knots": model.n_knots,
            "knot": model.knot,
            "force_limit": model.force_limit,
            "anchor": model.anchor.spec(),
            "meta": meta or {},
        },
        path,
    )


def load(path: str | Path) -> tuple[PlanNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["anchor"]
    anchor = Anchor(torch.tensor(a["K"]), torch.tensor(a["target"]), a["gate_in"], a["gate_out"])
    model = PlanNet(anchor, ck["n_knots"], ck["knot"], tuple(ck["hidden"]), ck["force_limit"])
    model.load_state_dict(ck["state_dict"])
    return model, ck.get("meta", {})
