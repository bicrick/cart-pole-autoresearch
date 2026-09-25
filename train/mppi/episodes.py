"""Batched closed-loop episodes, start-state samplers, and gate metrics.

A controller is anything with ``reset(batch)`` and ``act(states [B,8]) -> [B]``
forces in Newtons. Shoves are added to the controller force and then clipped,
the same way the web demo adds its a/d key force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import torch

from goals_triple import GOAL_ANGLES, parse_goal
from mppi.costs import wrap
from mppi.dynamics_fast import Plant, step

FALL_UP = 0.6
FALL_DOWN = 1.5
QUIET_ANGLE = 0.03
QUIET_RATE = 0.01


class Controller(Protocol):
    def reset(self, batch: int) -> None: ...
    def act(self, states: torch.Tensor) -> torch.Tensor: ...


def goal_angles(goal: str) -> torch.Tensor:
    return GOAL_ANGLES[parse_goal(goal)].to(torch.float32)


def near_states(goal: str, n: int, noise: float, rate: float, gen: torch.Generator) -> torch.Tensor:
    s = torch.zeros(n, 8)
    ang = goal_angles(goal)
    s[:, 0] = (torch.rand(n, generator=gen) * 2 - 1) * noise
    s[:, 1] = (torch.rand(n, generator=gen) * 2 - 1) * 0.01
    for j, i in enumerate((2, 4, 6)):
        s[:, i] = ang[j] + (torch.rand(n, generator=gen) * 2 - 1) * noise
        s[:, i + 1] = (torch.rand(n, generator=gen) * 2 - 1) * rate
    return s


def hang_states(n: int, noise: float, gen: torch.Generator) -> torch.Tensor:
    return near_states("DDD", n, noise, 0.01, gen)


def wide_states(n: int, gen: torch.Generator, x_max: float = 1.0, rate: float = 2.0) -> torch.Tensor:
    """Arbitrary poses anywhere on the circle, moderate rates."""
    s = torch.zeros(n, 8)
    s[:, 0] = (torch.rand(n, generator=gen) * 2 - 1) * x_max
    s[:, 1] = (torch.rand(n, generator=gen) * 2 - 1) * 0.5
    for i in (2, 4, 6):
        s[:, i] = (torch.rand(n, generator=gen) * 2 - 1) * math.pi
        s[:, i + 1] = (torch.rand(n, generator=gen) * 2 - 1) * rate
    return s


@dataclass
class Shove:
    """Constant extra force [N] from step ``start`` for ``steps`` steps."""

    start: int
    steps: int
    force: float


@dataclass
class EpisodeSpec:
    goal: str
    steps: int
    shoves: list[Shove] = field(default_factory=list)
    kick_step: int = -1
    kick_rate: float = 0.0


def angle_err(goal: str, states: torch.Tensor) -> torch.Tensor:
    ang = goal_angles(goal).to(states.device)
    return torch.stack([wrap(states[..., i] - ang[j]) for j, i in enumerate((2, 4, 6))], dim=-1)


def fall_limits(goal: str) -> torch.Tensor:
    ang = goal_angles(goal)
    return torch.where(ang.abs() < 1e-6, torch.tensor(FALL_UP), torch.tensor(FALL_DOWN))


@torch.no_grad()
def run(plant: Plant, ctrl: Controller, s0: torch.Tensor, spec: EpisodeSpec, gen: torch.Generator | None = None):
    """Returns per-step states [T+1,B,8] and forces [T,B]."""
    b = s0.shape[0]
    ctrl.reset(b)
    s = s0.clone()
    states = [s]
    forces = []
    fmax = plant.force_limit
    for t in range(spec.steps):
        u = ctrl.act(s).to(s.dtype)
        extra = sum(sh.force for sh in spec.shoves if sh.start <= t < sh.start + sh.steps)
        u = (u + extra).clamp(-fmax, fmax)
        if t == spec.kick_step and spec.kick_rate:
            g = gen or torch.Generator().manual_seed(t)
            s = s.clone()
            s[:, [3, 5, 7]] += (torch.rand(b, 3, generator=g) * 2 - 1) * spec.kick_rate
        s = step(plant, s, u)
        states.append(s)
        forces.append(u)
    return torch.stack(states), torch.stack(forces)


def hold_metrics(goal: str, states: torch.Tensor, track: float) -> dict:
    """Survival: never past the fall band or the track for the whole episode."""
    err = angle_err(goal, states).abs()
    fell = (err > fall_limits(goal).to(err.device)).any(-1).any(0)
    oob = (states[..., 0].abs() > track).any(0)
    alive = ~(fell | oob)
    final = err[-1]
    at_goal = alive & (final < 0.1).all(-1)
    align = torch.cos(final).mean(-1)
    return {
        "n": int(states.shape[1]),
        "survival": float(alive.float().mean()),
        "at_goal": float(at_goal.float().mean()),
        "align": float(align.mean()),
        "oob": float(oob.float().mean()),
    }


def quiet_mask(goal: str, states: torch.Tensor) -> torch.Tensor:
    err = angle_err(goal, states).abs()
    rate = states[..., [3, 5, 7]].abs()
    return (err <= QUIET_ANGLE).all(-1) & (rate <= QUIET_RATE).all(-1)


def swing_metrics(goal: str, states: torch.Tensor, track: float, hold_steps: int, dt: float) -> dict:
    """Enter the quiet box, then stay inside the fall band for ``hold_steps``."""
    t_total, b = states.shape[0], states.shape[1]
    quiet = quiet_mask(goal, states)
    entered = quiet.any(0)
    first = torch.where(entered, quiet.float().argmax(0), torch.full((b,), t_total, dtype=torch.long))
    err = angle_err(goal, states).abs()
    out_band = (err > fall_limits(goal).to(err.device)).any(-1)
    oob = (states[..., 0].abs() > track).any(0)
    idx = torch.arange(t_total).unsqueeze(1)
    after = idx >= first.unsqueeze(0)
    fell_after = entered & (out_band & after).any(0)
    late = entered & ((t_total - 1 - first) < hold_steps)
    held = entered & ~fell_after & ~late
    success = held & ~oob
    ok_first = first[entered].float()
    return {
        "n": int(b),
        "enter_rate": float(entered.float().mean()),
        "success": float(success.float().mean()),
        "fell_after_enter": float(fell_after.float().mean()),
        "late_enter": float(late.float().mean()),
        "oob": float(oob.float().mean()),
        "median_enter_s": float(ok_first.median() * dt) if ok_first.numel() else float("nan"),
        "max_abs_x": float(states[..., 0].abs().max()),
        "min_angle_err": float(err.max(-1).values.min()),
    }
