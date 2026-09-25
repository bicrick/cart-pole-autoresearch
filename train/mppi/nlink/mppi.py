"""Batched MPPI for the n-link plant (the algorithm of ``mppi/mppi.py``).

B plants replan together, K samples each: AR(1)-colored noise around the
residual plan, rollout costs from the n-link kernel (CUDA f32 or CPU f64), an
ESS-targeted softmax, and the cheapest of the previous plan, the weighted mean
and the best sample. Plants quiet at the target replan with ``hold_samples``.
Noise, weights and plans stay in torch on ``device``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from mppi.nlink.costs import Packed

LAMBDAS = torch.logspace(-4, 0, 25)


@dataclass
class NLinkMPPIConfig:
    n_samples: int = 16384
    n_knots: int = 45
    knot: int = 4
    sigma: float = 8.0
    sigma_wide: float = 25.0
    wide_frac: float = 0.25
    noise_beta: float = 0.7
    ess: float = 32.0
    # MPPI passes per replan; each pass samples around the previous pass's plan.
    n_iters: int = 1
    gate_in: float = 0.02
    gate_out: float = 0.10
    hold_samples: int = 128
    hold_d: float = 0.01
    hold_spin: float = 0.5
    backend: str = "cuda"  # or "cpu"
    seed: int = 0


class NLinkMPPI:
    def __init__(self, pk: Packed, n: int, force_limit: float, cfg: NLinkMPPIConfig, device: str | torch.device):
        self.cfg = cfg
        self.n = n
        self.fmax = float(force_limit)
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        self.K = torch.tensor(pk.K, dtype=torch.float32, device=self.device)
        self.targets = torch.tensor(pk.links[4], dtype=torch.float32, device=self.device)
        self._pk = pk
        if cfg.backend == "cuda":
            f = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.device)
            self._dev = (f(pk.scal), f(pk.links), f(pk.K), f(pk.P))
        self.uff: torch.Tensor | None = None
        self._tick = 0

    # --- dynamics-side pieces (execution uses the true plant; these mirror the kernel)
    def errors(self, s: torch.Tensor) -> torch.Tensor:
        e = s.clone()
        e[:, 2::2] = torch.remainder(s[:, 2::2] - self.targets + math.pi, 2 * math.pi) - math.pi
        return e

    def distance(self, s: torch.Tensor) -> torch.Tensor:
        return (1 - torch.cos(s[:, 2::2] - self.targets)).sum(-1)

    def anchor(self, s: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        gate = ((c.gate_out - self.distance(s)) / (c.gate_out - c.gate_in)).clamp(0.0, 1.0)
        return -gate * (self.errors(s) * self.K).sum(-1)

    def holding(self, s: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if not c.hold_samples:
            return torch.zeros(s.shape[0], dtype=torch.bool, device=s.device)
        return (self.distance(s) < c.hold_d) & ((s[:, 3::2] ** 2).sum(-1) < c.hold_spin)

    # --- sampling
    def rollout_costs(self, starts: torch.Tensor, plans: torch.Tensor, per_start: int) -> torch.Tensor:
        """starts [B, dim], plans [B * per_start, T] -> [B * per_start]."""
        pk = self._pk
        if self.cfg.backend == "cuda":
            from numba import cuda

            from mppi.nlink.kernel_cuda import rollout_costs

            st = starts.float().contiguous()
            pl = plans.float().contiguous()
            out = torch.empty(pl.shape[0], dtype=torch.float32, device=self.device)
            rollout_costs(*self._dev, cuda.as_cuda_array(st), cuda.as_cuda_array(pl), per_start,
                          cuda.as_cuda_array(out))
            cuda.synchronize()
            return out
        from mppi.nlink.kernel_cpu import rollout_costs

        out = rollout_costs(pk.scal, pk.links, pk.K, pk.P, starts.double().cpu().numpy(),
                            np.ascontiguousarray(plans.double().cpu().numpy()), per_start)
        return torch.from_numpy(out).float().to(self.device)

    def _noise(self, b: int, k: int) -> torch.Tensor:
        c = self.cfg
        z = torch.randn(b, k, c.n_knots, device=self.device, generator=self.gen)
        scale = (1.0 - c.noise_beta ** 2) ** 0.5
        out = torch.empty_like(z)
        out[..., 0] = z[..., 0]
        for t in range(1, c.n_knots):
            out[..., t] = c.noise_beta * out[..., t - 1] + scale * z[..., t]
        sig = torch.full((k,), c.sigma, device=self.device)
        n_wide = int(round(c.wide_frac * k))
        if n_wide:
            sig[-n_wide:] = c.sigma_wide
        out = out * sig.view(1, -1, 1)
        out[:, 0] = 0.0
        return out

    def _weights(self, S: torch.Tensor) -> torch.Tensor:
        d = S - S.min(dim=1, keepdim=True).values
        ref = torch.quantile(d, 0.5, dim=1, keepdim=True).clamp_min(1e-6)
        lams = ref.unsqueeze(1) * LAMBDAS.to(S.device).view(1, -1, 1)
        w = torch.softmax(-d.unsqueeze(1) / lams, dim=-1)
        ess = 1.0 / (w * w).sum(-1)
        pick = (ess - self.cfg.ess).abs().argmin(dim=1)
        return w[torch.arange(S.shape[0], device=S.device), pick]

    def _optimize(self, s0: torch.Tensor, U: torch.Tensor, k: int) -> torch.Tensor:
        b = s0.shape[0]
        lim = 2.0 * self.fmax
        rows = torch.arange(b, device=self.device)
        for _ in range(self.cfg.n_iters):
            samples = (U.unsqueeze(1) + self._noise(b, k)).clamp(-lim, lim)
            S = self.rollout_costs(s0, samples.reshape(b * k, -1), k).view(b, k)
            w = self._weights(S)
            mean = (w.unsqueeze(-1) * samples).sum(1)
            best = samples[rows, S.argmin(1)]
            cand = torch.stack((mean, best), 1)
            Sc = self.rollout_costs(s0, cand.reshape(b * 2, -1), 2).view(b, 2)
            # samples[:, 0] is the unperturbed plan U.
            pick = torch.cat((S[:, :1], Sc), 1).argmin(1)
            new = cand[rows, (pick - 1).clamp_min(0)]
            U = torch.where((pick == 0).unsqueeze(1), U, new)
        return U

    def optimize(self, s0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        hold = self.holding(s0)
        if not hold.any():
            return self._optimize(s0, U, self.cfg.n_samples)
        out = U.clone()
        out[hold] = self._optimize(s0[hold], U[hold], self.cfg.hold_samples)
        if (~hold).any():
            out[~hold] = self._optimize(s0[~hold], U[~hold], self.cfg.n_samples)
        return out

    # --- episode interface
    def reset(self, batch: int) -> None:
        self.uff = torch.zeros(batch, self.cfg.n_knots, device=self.device)
        self._tick = 0

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        s = states.to(self.device, torch.float32)
        if self.uff is None or self.uff.shape[0] != s.shape[0]:
            self.reset(s.shape[0])
        if self._tick % self.cfg.knot == 0:
            if self._tick > 0:
                self.uff = torch.cat((self.uff[:, 1:], torch.zeros_like(self.uff[:, -1:])), 1)
            self.uff = self.optimize(s, self.uff)
        self._tick += 1
        return (self.anchor(s) + self.uff[:, 0]).clamp(-self.fmax, self.fmax)
