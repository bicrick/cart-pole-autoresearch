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


def _uuu_eq() -> np.ndarray:
    return np.zeros(STATE_DIM, dtype=np.float64)


def _finite_diff_AB(
    constants: dict,
    eps_state: float = 1e-5,
    eps_force: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Discrete A,B around UUU: z_{t+1} = A z_t + B u_t (u in Newtons)."""
    z0 = torch.zeros(1, STATE_DIM, dtype=torch.float64)
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
    """u = clip(-K @ z, ±force_limit). z is the 8-D plant state."""

    K: np.ndarray
    force_limit: float = 40.0
    A: Optional[np.ndarray] = None
    B: Optional[np.ndarray] = None
    P: Optional[np.ndarray] = None

    def force(self, state: np.ndarray) -> float:
        z = np.asarray(state, dtype=np.float64).reshape(-1)[:STATE_DIM]
        u = float((-self.K @ z).ravel()[0])
        return float(np.clip(u, -self.force_limit, self.force_limit))

    def action_normed(self, state: np.ndarray) -> np.ndarray:
        """Gym action in [-1, 1] for TriplePendulumUUUEnv."""
        return np.array([self.force(state) / self.force_limit], dtype=np.float32)

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
) -> LQRUUU:
    consts = dict(constants or load_constants())
    consts["forceLimit"] = float(force_limit)
    A, B = _finite_diff_AB(consts)
    Q = build_Q(q_x=q_x, q_xd=q_xd, q_theta=q_theta, q_omega=q_omega)
    R = np.array([[float(r)]], dtype=np.float64)
    P = solve_dare(A, B, Q, R)
    S = R + B.T @ P @ B
    K = np.linalg.solve(S, B.T @ P @ A)
    return LQRUUU(K=K, force_limit=float(force_limit), A=A, B=B, P=P)


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
