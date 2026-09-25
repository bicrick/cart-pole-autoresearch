"""Classical energy swing-up for the cart-triple.

Plant power is F * xd. Cart kinetic energy is kept out of E, otherwise a fast
cart looks like the poles are already up and the pump quits. The force is

    u = k (E_upright - E_poles) * stroke - k_x x - k_v xd

saturated at forceLimit. `stroke` follows the sign of xd, held while the cart
is nearly stopped, and reversed at the bumper. k saturates the hang-to-upright
gap. Cart gains are the Dyad centering terms in newtons.
"""

from __future__ import annotations

import math

import numpy as np

from physics_triple import load_constants


def mechanical_energy(state: np.ndarray, constants: dict | None = None) -> float:
    """Kinetic + potential. y up, θ=0 upright, point masses at the tips."""
    c = constants or load_constants()
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    x, xd, th1, w1, th2, w2, th3, w3 = [float(v) for v in z[:8]]
    del x
    m1, m2, m3 = float(c["poleMass1"]), float(c["poleMass2"]), float(c["poleMass3"])
    l1, l2, l3 = float(c["poleLength1"]), float(c["poleLength2"]), float(c["poleLength3"])
    g = float(c["gravity"])
    cart_m = float(c["cartMass"])
    s1, c1 = math.sin(th1), math.cos(th1)
    s2, c2 = math.sin(th2), math.cos(th2)
    s3, c3 = math.sin(th3), math.cos(th3)
    v1x = xd + l1 * c1 * w1
    v1y = -l1 * s1 * w1
    v2x = v1x + l2 * c2 * w2
    v2y = v1y - l2 * s2 * w2
    v3x = v2x + l3 * c3 * w3
    v3y = v2y - l3 * s3 * w3
    kinetic = 0.5 * cart_m * xd * xd
    kinetic += 0.5 * m1 * (v1x * v1x + v1y * v1y)
    kinetic += 0.5 * m2 * (v2x * v2x + v2y * v2y)
    kinetic += 0.5 * m3 * (v3x * v3x + v3y * v3y)
    y1 = l1 * c1
    y2 = y1 + l2 * c2
    y3 = y2 + l3 * c3
    potential = g * (m1 * y1 + m2 * y2 + m3 * y3)
    return kinetic + potential


def upright_energy(constants: dict | None = None) -> float:
    return mechanical_energy(np.zeros(8), constants)


def pole_energy(state: np.ndarray, constants: dict | None = None) -> float:
    """Mechanical energy without the cart's own kinetic energy."""
    c = constants or load_constants()
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    cart_ke = 0.5 * float(c["cartMass"]) * float(z[1]) * float(z[1])
    return mechanical_energy(z, c) - cart_ke


def inner_link_energy(state: np.ndarray, constants: dict | None = None) -> float:
    """Energy of the cart-most mass. A shove moves this link and leaves the tip behind."""
    c = constants or load_constants()
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    xd, th1, w1 = float(z[1]), float(z[2]), float(z[3])
    m1 = float(c["poleMass1"])
    l1 = float(c["poleLength1"])
    g = float(c["gravity"])
    s1, c1 = math.sin(th1), math.cos(th1)
    v1x = xd + l1 * c1 * w1
    v1y = -l1 * s1 * w1
    return 0.5 * m1 * (v1x * v1x + v1y * v1y) + m1 * g * l1 * c1


def outer_link_energy(state: np.ndarray, constants: dict | None = None) -> float:
    """Energy of the tip mass only. A constant shove leaves this link behind."""
    c = constants or load_constants()
    z = np.asarray(state, dtype=np.float64).reshape(-1)
    x, xd, th1, w1, th2, w2, th3, w3 = [float(v) for v in z[:8]]
    del x
    m3 = float(c["poleMass3"])
    l1, l2, l3 = float(c["poleLength1"]), float(c["poleLength2"]), float(c["poleLength3"])
    g = float(c["gravity"])
    s1, c1 = math.sin(th1), math.cos(th1)
    s2, c2 = math.sin(th2), math.cos(th2)
    s3, c3 = math.sin(th3), math.cos(th3)
    v1x = xd + l1 * c1 * w1
    v1y = -l1 * s1 * w1
    v2x = v1x + l2 * c2 * w2
    v2y = v1y - l2 * s2 * w2
    v3x = v2x + l3 * c3 * w3
    v3y = v2y - l3 * s3 * w3
    y3 = l1 * c1 + l2 * c2 + l3 * c3
    return 0.5 * m3 * (v3x * v3x + v3y * v3y) + m3 * g * y3


def pole_energy_torch(state, constants: dict):
    """Batched pole energy. Matches pole_energy on a single state."""
    import torch

    xd = state[..., 1]
    th1, w1 = state[..., 2], state[..., 3]
    th2, w2 = state[..., 4], state[..., 5]
    th3, w3 = state[..., 6], state[..., 7]
    m1, m2, m3 = float(constants["poleMass1"]), float(constants["poleMass2"]), float(constants["poleMass3"])
    l1, l2, l3 = float(constants["poleLength1"]), float(constants["poleLength2"]), float(constants["poleLength3"])
    g = float(constants["gravity"])
    s1, c1 = torch.sin(th1), torch.cos(th1)
    s2, c2 = torch.sin(th2), torch.cos(th2)
    s3, c3 = torch.sin(th3), torch.cos(th3)
    v1x = xd + l1 * c1 * w1
    v1y = -l1 * s1 * w1
    v2x = v1x + l2 * c2 * w2
    v2y = v1y - l2 * s2 * w2
    v3x = v2x + l3 * c3 * w3
    v3y = v2y - l3 * s3 * w3
    kinetic = 0.5 * m1 * (v1x * v1x + v1y * v1y)
    kinetic = kinetic + 0.5 * m2 * (v2x * v2x + v2y * v2y)
    kinetic = kinetic + 0.5 * m3 * (v3x * v3x + v3y * v3y)
    y1 = l1 * c1
    y2 = y1 + l2 * c2
    y3 = y2 + l3 * c3
    potential = g * (m1 * y1 + m2 * y2 + m3 * y3)
    return kinetic + potential


class EnergyPump:
    """Gym action in [-1, 1]. Force saturates at force_limit."""

    def __init__(
        self,
        constants: dict | None = None,
        force_limit: float = 40.0,
        track_limit: float = 2.4,
        energy_margin: float = 0.0,
        k_x: float = 2.0,
        k_v: float = 4.0,
        target: str = "poles",
    ):
        self.constants = dict(constants or load_constants())
        self.force_limit = float(force_limit)
        self.target = target
        hang = np.zeros(8)
        hang[2] = hang[4] = hang[6] = math.pi
        if target == "outer":
            self._energy = outer_link_energy
            self.e_star = outer_link_energy(np.zeros(8), self.constants) + float(energy_margin)
            gap = abs(outer_link_energy(hang, self.constants) - self.e_star)
        elif target == "inner":
            self._energy = inner_link_energy
            upright = np.zeros(8)
            upright[4] = upright[6] = math.pi
            self.e_star = inner_link_energy(upright, self.constants) + float(energy_margin)
            gap = abs(inner_link_energy(hang, self.constants) - self.e_star)
        else:
            self._energy = pole_energy
            self.e_star = upright_energy(self.constants) + float(energy_margin)
            gap = abs(mechanical_energy(hang, self.constants) - self.e_star)
        self.k_swing = self.force_limit / max(gap, 1e-6)
        self.k_x = float(k_x)
        self.k_v = float(k_v)
        self.track_limit = float(track_limit)
        self._stroke = 1.0

    def force(self, state: np.ndarray) -> float:
        z = np.asarray(state, dtype=np.float64).reshape(-1)
        x = float(z[0])
        xd = float(z[1])
        energy = self._energy(z, self.constants)
        bumper = self.track_limit - 0.05
        if x > bumper:
            self._stroke = -1.0
        elif x < -bumper:
            self._stroke = 1.0
        elif abs(xd) > 0.05:
            self._stroke = 1.0 if xd > 0.0 else -1.0
        # Below the upright energy, push with the cart. Above it, oppose xd.
        u = self.k_swing * (self.e_star - energy) * self._stroke
        u -= self.k_x * x + self.k_v * xd
        return float(np.clip(u, -self.force_limit, self.force_limit))

    def action_normed(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.force(state) / self.force_limit], dtype=np.float32)

    def predict(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        del deterministic
        return self.action_normed(state)
