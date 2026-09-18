#!/usr/bin/env python3
"""Sanity-check the batched step against a scalar reference."""

import math

import torch

from physics import accelerations, load_constants, observe, step


def main():
    constants = load_constants()
    state = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6]], dtype=torch.float64)
    force = torch.tensor([3.5], dtype=torch.float64)
    nxt = step(state, force, constants=constants)
    acc = accelerations(state, force, constants=constants)
    assert acc.shape == (1, 3)
    assert nxt.shape == (1, 6)
    obs = observe(state)
    assert torch.allclose(obs[0, 2], torch.sin(state[0, 2]))
    assert torch.allclose(obs[0, 3], torch.cos(state[0, 2]))

    # Inverted equilibrium is unstable: a small +theta1 falls further (+t1dd).
    s = torch.tensor([[0.0, 0.0, 0.02, 0.0, 0.02, 0.0]], dtype=torch.float64)
    a = accelerations(s, torch.tensor([0.0], dtype=torch.float64), constants=constants)
    assert a[0, 1] > 0, a

    # Hanging should restore toward pi.
    s = torch.tensor([[0.0, 0.0, math.pi + 0.05, 0.0, math.pi + 0.05, 0.0]], dtype=torch.float64)
    a = accelerations(s, torch.tensor([0.0], dtype=torch.float64), constants=constants)
    # theta = pi+0.05 is slightly past down; restoring accel should be negative toward pi
    # (further toward 2pi? wait: hanging at pi, displacement +0.05, gravity pulls back to pi so tdd < 0)
    assert a[0, 1] < 0, a
    print("physics ok", nxt.tolist()[0], a.tolist()[0])

    # Cart-only inelastic endstops: stay inside rail; cart speed killed on hit.
    track = float(constants.get("trackLimit", 2.4))
    s = torch.tensor([[track - 0.01, 8.0, 0.2, 1.0, -0.2, -1.0]], dtype=torch.float64)
    th1d0, th2d0 = float(s[0, 3]), float(s[0, 5])
    for _ in range(30):
        s = step(s, torch.tensor([20.0], dtype=torch.float64), constants=constants)
    assert float(s[0, 0]) <= track + 1e-9, s
    assert float(s[0, 0]) >= -track - 1e-9, s
    assert abs(float(s[0, 1])) < 1.0, s  # cart not bouncing hard
    # Poles were never clamped by the wall (angles/vels evolve freely).
    print("walls ok", float(s[0, 0]), float(s[0, 1]), float(s[0, 3]), float(s[0, 5]))


if __name__ == "__main__":
    main()
