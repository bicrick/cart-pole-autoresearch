"""Cart + N-link pendulum: the double/triple model for any link count.

Point masses at the link tips, absolute link angles (0 = upright). With
S_i = sum_{k >= i} m_k the mass at or beyond link i (links 1-indexed):

    M00 = M_cart + sum m          M0i = S_i l_i cos th_i
    Mij = S_max(i,j) l_i l_j cos(th_i - th_j)
    rhs0 = u - b xd + sum_i S_i l_i sin th_i w_i^2
    rhs_i = -c_i w_i - sum_{j != i} S_max(i,j) l_i l_j sin(th_i - th_j) w_j^2 + S_i g l_i sin th_i

State [..., 2 + 2n] = x, xd, th1, w1, ..., thn, wn. ``step`` uses the same
clamps and semi-implicit Euler as ``physics.step`` / ``physics_triple.step``,
which ``test_plant`` checks it against for n = 2 and n = 3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CONSTANTS = {2: "constants.json", 3: "constants-triple.json", 4: "constants-quad.json"}

MAX_CART_VEL = 30.0
MAX_ANG_VEL = 50.0
MAX_ACC = 1e4


@dataclass(frozen=True)
class NLinkPlant:
    n: int
    M: float
    m: tuple[float, ...]
    l: tuple[float, ...]
    c: tuple[float, ...]
    g: float
    b: float
    dt: float
    force_limit: float
    track_limit: float = 2.4
    # Diagonal jitter on the mass matrix (physics_triple uses 1e-8; physics.py's closed-form 3x3 has none).
    jitter: float = 1e-8

    @property
    def dim(self) -> int:
        return 2 + 2 * self.n

    @property
    def S(self) -> tuple[float, ...]:
        """Mass at or beyond each link."""
        return tuple(sum(self.m[i:]) for i in range(self.n))

    @classmethod
    def from_constants(cls, c: dict, force_limit: float | None = None) -> "NLinkPlant":
        n = int(c.get("nLinks") or sum(1 for k in c if k.startswith("poleMass")))
        return cls(
            n=n,
            M=float(c["cartMass"]),
            m=tuple(float(c[f"poleMass{i}"]) for i in range(1, n + 1)),
            l=tuple(float(c[f"poleLength{i}"]) for i in range(1, n + 1)),
            c=tuple(float(c[f"jointDamping{i}"]) for i in range(1, n + 1)),
            g=float(c["gravity"]),
            b=float(c["cartFriction"]),
            dt=float(c["dt"]),
            force_limit=float(force_limit if force_limit is not None else c["forceLimit"]),
            track_limit=float(c.get("trackLimit", 2.4)),
        )

    @classmethod
    def load(cls, n: int, force_limit: float | None = None) -> "NLinkPlant":
        return cls.from_constants(json.loads((ROOT / "shared" / CONSTANTS[n]).read_text()), force_limit)


def accelerations(p: NLinkPlant, state: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    """[..., n + 1] = xdd, th1dd, ..., thndd."""
    n = p.n
    th = state[..., 2::2]
    w = state[..., 3::2]
    xd = state[..., 1]
    s, c = torch.sin(th), torch.cos(th)
    S = torch.tensor(p.S, dtype=state.dtype, device=state.device)
    l = torch.tensor(p.l, dtype=state.dtype, device=state.device)
    cd = torch.tensor(p.c, dtype=state.dtype, device=state.device)
    idx = torch.arange(n, device=state.device)
    Smax = S[torch.maximum(idx[:, None], idx[None, :])]  # [n, n]
    ll = l[:, None] * l[None, :]
    dth = th[..., :, None] - th[..., None, :]  # [..., n, n]
    lead = state.shape[:-1]
    mass = torch.empty(*lead, n + 1, n + 1, dtype=state.dtype, device=state.device)
    mass[..., 0, 0] = p.M + sum(p.m)
    mass[..., 0, 1:] = S * l * c
    mass[..., 1:, 0] = S * l * c
    mass[..., 1:, 1:] = Smax * ll * torch.cos(dth)
    rhs = torch.empty(*lead, n + 1, dtype=state.dtype, device=state.device)
    rhs[..., 0] = force - p.b * xd + (S * l * s * w * w).sum(-1)
    coup = (Smax * ll * torch.sin(dth) * (w * w)[..., None, :]).sum(-1)  # diagonal sin(0) = 0
    rhs[..., 1:] = -cd * w - coup + S * p.g * l * s
    eye = torch.eye(n + 1, dtype=state.dtype, device=state.device)
    return torch.linalg.solve(mass + p.jitter * eye, rhs.unsqueeze(-1)).squeeze(-1)


def step(p: NLinkPlant, state: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    fmax = p.force_limit
    s = state.clone()
    s[..., 1] = s[..., 1].clamp(-MAX_CART_VEL, MAX_CART_VEL)
    s[..., 3::2] = s[..., 3::2].clamp(-MAX_ANG_VEL, MAX_ANG_VEL)
    force = torch.nan_to_num(force, nan=0.0, posinf=fmax, neginf=-fmax).clamp(-fmax, fmax)
    acc = accelerations(p, s, force)
    acc = torch.nan_to_num(acc, nan=0.0, posinf=MAX_ACC, neginf=-MAX_ACC).clamp(-MAX_ACC, MAX_ACC)
    out = torch.empty_like(s)
    out[..., 1] = (s[..., 1] + acc[..., 0] * p.dt).clamp(-MAX_CART_VEL, MAX_CART_VEL)
    out[..., 3::2] = (s[..., 3::2] + acc[..., 1:] * p.dt).clamp(-MAX_ANG_VEL, MAX_ANG_VEL)
    out[..., 0] = s[..., 0] + out[..., 1] * p.dt
    out[..., 2::2] = s[..., 2::2] + out[..., 3::2] * p.dt
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def pole_energy(p: NLinkPlant, state: torch.Tensor) -> torch.Tensor:
    """Pole KE + PE in the cart frame (cart velocity excluded)."""
    th = state[..., 2::2]
    w = state[..., 3::2]
    l = torch.tensor(p.l, dtype=state.dtype, device=state.device)
    m = torch.tensor(p.m, dtype=state.dtype, device=state.device)
    vx = torch.cumsum(l * torch.cos(th) * w, -1)
    vy = torch.cumsum(-l * torch.sin(th) * w, -1)
    y = torch.cumsum(l * torch.cos(th), -1)
    return 0.5 * (m * (vx * vx + vy * vy)).sum(-1) + p.g * (m * y).sum(-1)
