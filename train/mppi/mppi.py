"""Batched MPPI over B plants, K samples each, with a gated LQR anchor.

The upright triple grows errors at ~11/s, so open-loop force samples all
diverge inside a 1.5 s horizon, and the best offline search (multi-start
iLQR) only matches the target's LQR near upright. Each sample is therefore a
residual on a state-feedback anchor:

    u_k,t = clip(gate(s) * (-K_lqr e(s)) + u_ff_t + eps_k,t)

``gate`` is 1 inside the target's catch region and fades to 0 far from it, so
swing-up samples are open loop until the poles approach the target and are
scored on whether the anchor can actually catch that arrival.

The plan ``u_ff`` is ``n_knots`` zero-order-hold residuals of ``knot`` sim
steps each. The controller replans at every knot boundary, keeps the cheapest
of the MPPI mean, the best sample, and the previous plan, then shifts.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch

from mppi.costs import TargetCost
from mppi.dynamics_fast import Plant, step

TWO_PI = 2.0 * math.pi


@dataclass
class MPPIConfig:
    n_samples: int = 512
    n_knots: int = 45
    knot: int = 4
    sigma: float = 8.0
    sigma_wide: float = 25.0
    wide_frac: float = 0.25
    noise_beta: float = 0.7
    ess: float = 32.0
    n_iters: int = 1
    gate_in: float = 0.02
    gate_out: float = 0.10
    anchor: bool = True
    # Rollouts integrate at rollout_sub * dt (numba backend; knot % rollout_sub == 0).
    rollout_sub: int = 1
    # Stop a rollout once the cart is past x_dead and charge the rest as dead.
    early_exit: bool = False
    # Hold mode: plants inside d < hold_d with sum(omega^2) < hold_spin replan
    # with hold_samples samples (0 = off).
    hold_samples: int = 0
    hold_d: float = 0.01
    hold_spin: float = 0.5
    # Path to a ``mppi.value`` net used as the terminal cost instead of the
    # analytic one (numba backend; pair with a shorter n_knots).
    terminal_value: str = ""
    backend: str = "numba"
    compile: bool = True
    device: str = "cpu"
    seed: int = 0

    @property
    def horizon_steps(self) -> int:
        return self.n_knots * self.knot

    def to_dict(self) -> dict:
        return asdict(self)


def make_anchor(cost: TargetCost, cfg: MPPIConfig):
    """Gated target-LQR feedback force for states [..., 8]."""
    k = cost.K_lqr
    tgt = cost.target_angles
    lo, hi = cfg.gate_in, cfg.gate_out
    on = cfg.anchor

    def anchor(s: torch.Tensor) -> torch.Tensor:
        if not on:
            return torch.zeros_like(s[..., 0])
        e1 = torch.remainder(s[..., 2] - tgt[0] + math.pi, TWO_PI) - math.pi
        e2 = torch.remainder(s[..., 4] - tgt[1] + math.pi, TWO_PI) - math.pi
        e3 = torch.remainder(s[..., 6] - tgt[2] + math.pi, TWO_PI) - math.pi
        d = (1 - torch.cos(e1)) + (1 - torch.cos(e2)) + (1 - torch.cos(e3))
        gate = ((hi - d) / (hi - lo)).clamp(0.0, 1.0)
        fb = (
            k[0] * s[..., 0] + k[1] * s[..., 1] + k[2] * e1 + k[3] * s[..., 3]
            + k[4] * e2 + k[5] * s[..., 5] + k[6] * e3 + k[7] * s[..., 7]
        )
        return -gate * fb

    return anchor


class MPPI:
    def __init__(self, plant: Plant, cost: TargetCost, cfg: MPPIConfig):
        self.plant = plant
        self.cost = cost
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.fmax = plant.force_limit
        self.anchor = make_anchor(cost, cfg)
        self._fast = None
        self._chunk = None
        if cfg.backend == "numba":
            from mppi.fast_rollout import pack_params

            self._fast = pack_params(plant, cost, cfg)
        else:
            if cfg.rollout_sub != 1 or cfg.early_exit or cfg.terminal_value:
                raise ValueError("rollout_sub / early_exit / terminal_value need the numba backend")
            self._chunk = self._make_chunk()
        self._value = None
        if cfg.terminal_value:
            from mppi.value import load as load_value

            self._value = load_value(cfg.terminal_value)
        self.uff: torch.Tensor | None = None
        self._tick = 0

    def _make_chunk(self):
        plant, cost, knot, fmax, anchor = self.plant, self.cost, self.cfg.knot, self.fmax, self.anchor

        def chunk(state, u_knot, acc):
            # state [N,8], u_knot [N] residual for this knot, acc [N]
            zero = torch.zeros_like(acc)
            for _ in range(knot):
                u = (anchor(state) + u_knot).clamp(-fmax, fmax)
                state = step(plant, state, u)
                acc = acc + cost.running(state, u, zero)
            return state, acc

        if self.cfg.compile:
            # One compiled graph per (goal, batch shape); 8 goals exceed the default limit.
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
            return torch.compile(chunk, dynamic=False)
        return chunk

    def rollout_costs(self, states: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """states [B,8], residual plans U [B,K,T] -> total cost [B,K]."""
        b, k, t = U.shape
        c = self.cfg
        if self._fast is not None:
            from mppi.fast_rollout import rollout_costs, rollout_costs_final

            starts = np.ascontiguousarray(states.double().unsqueeze(1).expand(b, k, 8).reshape(b * k, 8).numpy())
            plans = np.ascontiguousarray(U.double().reshape(b * k, t).numpy())
            if self._value is not None:
                costs, fins = rollout_costs_final(*self._fast, starts, plans)
                v = self._value(torch.from_numpy(fins)).double().numpy()
                return torch.from_numpy(costs + v).float().view(b, k)
            costs = rollout_costs(*self._fast, starts, plans)
            return torch.from_numpy(costs).float().view(b, k)
        s = states.unsqueeze(1).expand(b, k, 8).reshape(b * k, 8).contiguous()
        u = U.reshape(b * k, t)
        acc = torch.zeros(b * k, device=self.device)
        for j in range(t):
            s, acc = self._chunk(s, u[:, j].contiguous(), acc)
        acc = acc + self.cost.terminal(s)
        du = torch.diff(u, dim=1) / (c.knot * self.plant.dt)
        acc = acc + self.cost.cost_du(du) * c.knot * self.plant.dt
        return acc.view(b, k)

    def _noise(self, b: int, k: int | None = None) -> torch.Tensor:
        c = self.cfg
        k = k or c.n_samples
        n = torch.randn(b, k, c.n_knots, device=self.device, generator=self.gen)
        beta = c.noise_beta
        if beta > 0:
            scale = (1.0 - beta * beta) ** 0.5
            out = torch.empty_like(n)
            out[..., 0] = n[..., 0]
            for t in range(1, c.n_knots):
                out[..., t] = beta * out[..., t - 1] + scale * n[..., t]
            n = out
        sig = torch.full((k,), c.sigma, device=self.device)
        n_wide = int(round(c.wide_frac * k))
        if n_wide:
            sig[-n_wide:] = c.sigma_wide
        n = n * sig.view(1, -1, 1)
        n[:, 0] = 0.0
        return n

    def _weights(self, S: torch.Tensor) -> torch.Tensor:
        """Softmax weights whose temperature puts the effective sample size near ``ess``."""
        smin = S.min(dim=1, keepdim=True).values
        d = S - smin
        ref = torch.quantile(d, 0.5, dim=1, keepdim=True).clamp_min(1e-6)
        lams = ref.unsqueeze(1) * torch.logspace(-4, 0, 25, device=S.device).view(1, -1, 1)
        w = torch.softmax(-d.unsqueeze(1) / lams, dim=-1)
        ess = 1.0 / (w * w).sum(-1)
        pick = (ess - self.cfg.ess).abs().argmin(dim=1)
        return w[torch.arange(S.shape[0], device=S.device), pick]

    def holding(self, s: torch.Tensor) -> torch.Tensor:
        """[B] bool: quiet at the target, where hold_samples suffice."""
        c = self.cfg
        if not c.hold_samples:
            return torch.zeros(s.shape[0], dtype=torch.bool, device=s.device)
        t = self.cost.target_angles
        d = (1 - torch.cos(s[:, 2] - t[0])) + (1 - torch.cos(s[:, 4] - t[1])) + (1 - torch.cos(s[:, 6] - t[2]))
        spin = s[:, 3] ** 2 + s[:, 5] ** 2 + s[:, 7] ** 2
        return (d < c.hold_d) & (spin < c.hold_spin)

    def optimize(self, s0: torch.Tensor, U: torch.Tensor, n_iters: int | None = None) -> torch.Tensor:
        """MPPI update of residual plan U [B,T] from states s0 [B,8]."""
        hold = self.holding(s0)
        if hold.any():
            out = U.clone()
            out[hold] = self._optimize(s0[hold], U[hold], n_iters, self.cfg.hold_samples)
            if (~hold).any():
                out[~hold] = self._optimize(s0[~hold], U[~hold], n_iters, self.cfg.n_samples)
            return out
        return self._optimize(s0, U, n_iters, self.cfg.n_samples)

    def _optimize(self, s0: torch.Tensor, U: torch.Tensor, n_iters: int | None, k: int) -> torch.Tensor:
        lim = 2.0 * self.fmax
        b = s0.shape[0]
        for _ in range(n_iters or self.cfg.n_iters):
            samples = (U.unsqueeze(1) + self._noise(b, k)).clamp(-lim, lim)
            S = self.rollout_costs(s0, samples)
            w = self._weights(S)
            mean = (w.unsqueeze(-1) * samples).sum(1)
            best = samples[torch.arange(b), S.argmin(1)]
            cand = torch.stack((mean, best), 1)
            Sc = self.rollout_costs(s0, cand)
            # samples[:, 0] is the unperturbed previous plan.
            allc = torch.cat((S[:, :1], Sc), 1)
            pick = allc.argmin(1)
            U = torch.where((pick == 0).unsqueeze(1), U, cand[torch.arange(b), (pick - 1).clamp_min(0)])
        return U

    def reset(self, batch: int) -> None:
        self.uff = torch.zeros(batch, self.cfg.n_knots, device=self.device)
        self._tick = 0

    def force(self, s: torch.Tensor, u_knot: torch.Tensor) -> torch.Tensor:
        return (self.anchor(s) + u_knot).clamp(-self.fmax, self.fmax)

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        """Force [B] for this sim step. Replans at each knot boundary."""
        s = states.to(self.device, torch.float32)
        if self.uff is None or self.uff.shape[0] != s.shape[0]:
            self.reset(s.shape[0])
        if self._tick % self.cfg.knot == 0:
            if self._tick > 0:
                self.uff = torch.cat((self.uff[:, 1:], torch.zeros_like(self.uff[:, -1:])), 1)
            self.uff = self.optimize(s, self.uff)
        self._tick += 1
        return self.force(s, self.uff[:, 0])
