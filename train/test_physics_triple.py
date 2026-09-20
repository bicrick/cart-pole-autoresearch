#!/usr/bin/env python3
"""Sanity-check batched cart-triple-pendulum step."""

import math

import torch

from physics_triple import accelerations, load_constants, observe, step
from goals_triple import OBS_DIM, GOAL_IDS, conditioned_obs, goal_encoding, sample_goals


def main():
    constants = load_constants()
    assert constants["obsDim"] == OBS_DIM == 25, (constants["obsDim"], OBS_DIM)
    assert len(GOAL_IDS) == 8

    state = torch.tensor(
        [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.1, -0.2]], dtype=torch.float64
    )
    force = torch.tensor([3.5], dtype=torch.float64)
    nxt = step(state, force, constants=constants)
    acc = accelerations(state, force, constants=constants)
    assert acc.shape == (1, 4), acc.shape
    assert nxt.shape == (1, 8), nxt.shape
    assert torch.isfinite(nxt).all() and torch.isfinite(acc).all()

    obs = observe(state)
    assert obs.shape == (1, 11)
    assert torch.allclose(obs[0, 2], torch.sin(state[0, 2]))
    assert torch.allclose(obs[0, 3], torch.cos(state[0, 2]))
    assert torch.allclose(obs[0, 6], torch.sin(state[0, 6]))
    assert torch.allclose(obs[0, 7], torch.cos(state[0, 6]))

    # Inverted equilibrium unstable: small +theta falls further (+tdd).
    s = torch.tensor(
        [[0.0, 0.0, 0.02, 0.0, 0.02, 0.0, 0.02, 0.0]], dtype=torch.float64
    )
    a = accelerations(s, torch.tensor([0.0], dtype=torch.float64), constants=constants)
    assert a[0, 1] > 0 and a[0, 2] > 0 and a[0, 3] > 0, a

    # Hanging should restore toward pi.
    s = torch.tensor(
        [[0.0, 0.0, math.pi + 0.05, 0.0, math.pi + 0.05, 0.0, math.pi + 0.05, 0.0]],
        dtype=torch.float64,
    )
    a = accelerations(s, torch.tensor([0.0], dtype=torch.float64), constants=constants)
    assert a[0, 1] < 0 and a[0, 2] < 0 and a[0, 3] < 0, a

    # Tiny tip mass still invertible.
    c2 = dict(constants)
    c2["poleMass3"] = 1e-9
    s = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.0, 0.0]], dtype=torch.float64)
    a = accelerations(s, torch.tensor([3.5], dtype=torch.float64), constants=c2)
    assert torch.isfinite(a).all(), a

    # No walls: sustained force lets the cart leave the track limit.
    track = float(constants.get("trackLimit", 2.4))
    s = torch.tensor(
        [[track - 0.01, 8.0, 0.2, 1.0, -0.2, -1.0, 0.1, 0.5]], dtype=torch.float64
    )
    for _ in range(30):
        s = step(s, torch.tensor([20.0], dtype=torch.float64), constants=constants)
    assert float(s[0, 0]) > track, s

    g = sample_goals(4, device=torch.device("cpu"))
    enc = goal_encoding(g)
    assert enc.shape == (4, 14)
    full = conditioned_obs(observe(state.expand(4, -1).float()), g)
    assert full.shape == (4, OBS_DIM)

    print("physics_triple ok", nxt.tolist()[0][:4], "acc", [float(x) for x in a[0]])
    print("nowalls ok", float(s[0, 0]), "OBS_DIM", OBS_DIM, "goals", GOAL_IDS)

    # Hard inelastic walls: cart held at trackLimit, xd -> 0 (cart-only, no bounce).
    c_walls = dict(constants)
    c_walls["trackWalls"] = True
    track = float(c_walls.get("trackLimit", 2.4))
    s = torch.tensor(
        [[track - 0.01, 8.0, 0.2, 1.0, -0.2, -1.0, 0.1, 0.5]], dtype=torch.float64
    )
    for _ in range(30):
        s = step(s, torch.tensor([20.0], dtype=torch.float64), constants=c_walls)
    assert abs(float(s[0, 0])) <= track + 1e-9, s
    assert abs(float(s[0, 1])) < 1e-9, s
    s = torch.tensor(
        [[-track + 0.01, -8.0, 0.2, 1.0, -0.2, -1.0, 0.1, 0.5]], dtype=torch.float64
    )
    for _ in range(30):
        s = step(s, torch.tensor([-20.0], dtype=torch.float64), constants=c_walls)
    assert abs(float(s[0, 0])) <= track + 1e-9, s
    assert abs(float(s[0, 1])) < 1e-9, s
    print("walls ok", float(s[0, 0]), float(s[0, 1]))


def _mech_energy(state, constants):
    """Kinetic + potential (y up, θ=0 upright) for tip point-masses."""
    x, xd, th1, th1d, th2, th2d, th3, th3d = [float(v) for v in state[0]]
    M = constants["cartMass"]
    m1, m2, m3 = constants["poleMass1"], constants["poleMass2"], constants["poleMass3"]
    l1, l2, l3 = constants["poleLength1"], constants["poleLength2"], constants["poleLength3"]
    g = constants["gravity"]
    s1, c1 = math.sin(th1), math.cos(th1)
    s2, c2 = math.sin(th2), math.cos(th2)
    s3, c3 = math.sin(th3), math.cos(th3)
    v1x = xd + l1 * c1 * th1d
    v1y = -l1 * s1 * th1d
    v2x = v1x + l2 * c2 * th2d
    v2y = v1y - l2 * s2 * th2d
    v3x = v2x + l3 * c3 * th3d
    v3y = v2y - l3 * s3 * th3d
    T = 0.5 * M * xd * xd + 0.5 * m1 * (v1x * v1x + v1y * v1y)
    T += 0.5 * m2 * (v2x * v2x + v2y * v2y) + 0.5 * m3 * (v3x * v3x + v3y * v3y)
    y1 = l1 * c1
    y2 = y1 + l2 * c2
    y3 = y2 + l3 * c3
    V = m1 * g * y1 + m2 * g * y2 + m3 * g * y3
    return T + V


def test_undamped_energy_bound():
    """Undamped near-rest energy should not explode over ~0.4 s of SI Euler."""
    constants = load_constants()
    c = dict(constants)
    c["cartFriction"] = 0.0
    c["jointDamping1"] = 0.0
    c["jointDamping2"] = 0.0
    c["jointDamping3"] = 0.0
    s = torch.tensor(
        [[0.0, 0.0, 0.05, 0.1, -0.04, -0.08, 0.03, 0.05]], dtype=torch.float64
    )
    e0 = _mech_energy(s, c)
    for _ in range(50):
        s = step(s, torch.tensor([0.0], dtype=torch.float64), constants=c)
        assert torch.isfinite(s).all()
    e1 = _mech_energy(s, c)
    rel = abs(e1 - e0) / (abs(e0) + 1.0)
    assert rel < 0.15, (e0, e1, rel)


if __name__ == "__main__":
    main()
    test_undamped_energy_bound()
    print("energy-step ok")
