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


if __name__ == "__main__":
    main()
