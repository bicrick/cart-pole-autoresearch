"""Discrete LQR UUU catcher for cart-triple (ResearchSquare / Lim staging).

Linearizes the plant at upright (all θ=0, ω=0, x=0) via finite differences on
``physics_triple.step``, solves the discrete ARE in pure NumPy, and returns
``u = clip(-K z, ±forceLimit)``.

Defaults follow ResearchSquare 2026: Q_θ ~ 100, R ~ 0.01. Stiffen Q_θ ~ 1e3
only if the soft RoA fails the catch-basin measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from physics_triple import load_constants, observe, step

STATE_DIM = 8  # x, xd, th1, th1d, th2, th2d, th3, th3d


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def equilibrium_state(goal: str = "UUU") -> np.ndarray:
    """8-D plant state at a named equilibrium. UUU is the origin."""
    from goals_triple import GOAL_ANGLES, parse_goal

    angles = GOAL_ANGLES[parse_goal(goal)].detach().cpu().numpy().astype(np.float64)
    z = np.zeros(STATE_DIM, dtype=np.float64)
    z[2], z[4], z[6] = angles
    return z


def state_error(state: np.ndarray, z_eq: np.ndarray) -> np.ndarray:
    """Deviation from equilibrium. Angles wrapped into (-π, π]."""
    e = np.asarray(state, dtype=np.float64).reshape(-1)[:STATE_DIM] - np.asarray(z_eq, dtype=np.float64)
    for i in (2, 4, 6):
        e[i] = float(_wrap_angle(np.array(e[i])))
    return e


def _uuu_eq() -> np.ndarray:
    return equilibrium_state("UUU")


def _finite_diff_AB(
    constants: dict,
    z_eq: np.ndarray | None = None,
    eps_state: float = 1e-5,
    eps_force: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete A,B around z_eq: δ_{t+1} = A δ_t + B u_t (u in Newtons)."""
    eq = np.zeros(STATE_DIM, dtype=np.float64) if z_eq is None else np.asarray(z_eq, dtype=np.float64)
    z0 = torch.tensor(eq, dtype=torch.float64).reshape(1, STATE_DIM)
    u0 = torch.zeros(1, dtype=torch.float64)
    consts = dict(constants)
    # Keep forceLimit high during linearization so clamp doesn't bite.
    consts["forceLimit"] = max(float(consts.get("forceLimit", 40.0)), 1e3)

    def nxt(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return step(z, u, constants=consts)

    f0 = nxt(z0, u0).squeeze(0).numpy()
    A = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    for i in range(STATE_DIM):
        zp = z0.clone()
        zp[0, i] += eps_state
        fp = nxt(zp, u0).squeeze(0).numpy()
        zm = z0.clone()
        zm[0, i] -= eps_state
        fm = nxt(zm, u0).squeeze(0).numpy()
        A[:, i] = (fp - fm) / (2.0 * eps_state)

    up = torch.tensor([eps_force], dtype=torch.float64)
    um = torch.tensor([-eps_force], dtype=torch.float64)
    B = ((nxt(z0, up) - nxt(z0, um)).squeeze(0).numpy()) / (2.0 * eps_force)
    B = B.reshape(STATE_DIM, 1)
    # Subtract equilibrium drift (should be ~0 at UUU with u=0).
    _ = f0
    return A, B


def solve_dare(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Iterative discrete algebraic Riccati equation (no SciPy)."""
    P = Q.copy()
    BT = B.T
    AT = A.T
    for _ in range(max_iter):
        S = R + BT @ P @ B
        K = np.linalg.solve(S, BT @ P @ A)
        P_next = Q + AT @ P @ A - AT @ P @ B @ K
        if np.max(np.abs(P_next - P)) < tol:
            return P_next
        P = 0.5 * (P_next + P_next.T)  # keep symmetric
    return P


def build_Q(
    q_x: float = 1.0,
    q_xd: float = 1.0,
    q_theta: float = 100.0,
    q_omega: float = 1.0,
) -> np.ndarray:
    """Diagonal state cost. q_theta applies to θ1,θ2,θ3."""
    diag = [
        q_x,
        q_xd,
        q_theta,
        q_omega,
        q_theta,
        q_omega,
        q_theta,
        q_omega,
    ]
    return np.diag(np.asarray(diag, dtype=np.float64))


@dataclass
class LQRUUU:
    """u = clip(-K @ (z - z_eq), ±force_limit). z is the 8-D plant state."""

    K: np.ndarray
    force_limit: float = 40.0
    z_eq: Optional[np.ndarray] = None
    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None

    def force(self, state: np.ndarray) -> float:
        z_eq = np.zeros(STATE_DIM) if self.z_eq is None else self.z_eq
        e = state_error(state, z_eq)
        u = float((-self.K @ e).ravel()[0])
        return float(np.clip(u, -self.force_limit, self.force_limit))

    def action_normed(self, state: np.ndarray) -> np.ndarray:
        """Gym action in [-1, 1] for TriplePendulumUUUEnv."""
        return np.array([self.force(state) / self.force_limit], dtype=np.float32)

    def action_from_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Batched gym action in [-1, 1] from the 11-D observation.

        obs is [x, xd, sin/cos ×3, ω ×3]. atan2 recovers the wrapped UUU error.
        """
        th1 = torch.atan2(obs[..., 2], obs[..., 3])
        th2 = torch.atan2(obs[..., 4], obs[..., 5])
        th3 = torch.atan2(obs[..., 6], obs[..., 7])
        z = torch.stack(
            (obs[..., 0], obs[..., 1], th1, obs[..., 8], th2, obs[..., 9], th3, obs[..., 10]),
            dim=-1,
        )
        k = torch.as_tensor(self.K, device=obs.device, dtype=obs.dtype).reshape(-1)
        u = -(z * k).sum(dim=-1) / float(self.force_limit)
        return u.clamp(-1.0, 1.0)

    def predict(self, obs_or_state: np.ndarray, *, from_obs: bool = False) -> np.ndarray:
        """SB3-like predict. Prefer raw state; obs→state is lossy so discouraged."""
        if from_obs:
            raise ValueError("LQRUUU needs raw 8-D state; pass from_obs=False")
        return self.action_normed(obs_or_state)


def design_lqr_uuu(
    *,
    force_limit: float = 40.0,
    q_theta: float = 100.0,
    r: float = 0.01,
    q_x: float = 1.0,
    q_xd: float = 1.0,
    q_omega: float = 1.0,
    constants: Optional[dict] = None,
    goal: str = "UUU",
) -> LQRUUU:
    consts = dict(constants or load_constants())
    consts["forceLimit"] = float(force_limit)
    z_eq = equilibrium_state(goal)
    A, B = _finite_diff_AB(consts, z_eq=z_eq)
    Q = build_Q(q_x=q_x, q_xd=q_xd, q_theta=q_theta, q_omega=q_omega)
    R = np.array([[float(r)]], dtype=np.float64)
    P = solve_dare(A, B, Q, R)
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return LQRUUU(K=K, force_limit=float(force_limit), z_eq=z_eq, A=A, B=B, P=P)


def state_from_env(env) -> np.ndarray:
    """Pull 8-D plant state from TriplePendulumUUUEnv."""
    st = env._state
    if st is None:
        raise RuntimeError("env has no state; call reset first")
    return st.detach().cpu().numpy().astype(np.float64)


if __name__ == "__main__":
    ctrl = design_lqr_uuu()
    print("K shape", ctrl.K.shape)
    print("K", np.array2string(ctrl.K, precision=4))
    z = _uuu_eq()
    print("force@eq", ctrl.force(z))
    z[2] = 0.05  # tip θ1
    print("force@th1=0.05", ctrl.force(z))
