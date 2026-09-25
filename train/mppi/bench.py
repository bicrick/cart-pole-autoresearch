#!/usr/bin/env python3
"""G0 throughput: batched rollouts of K samples over H steps."""

from __future__ import annotations

import argparse
import time

import torch

from mppi.dynamics_fast import Plant, step


def bench(device: str, k: int, h: int, reps: int) -> float:
    plant = Plant.from_constants()
    dev = torch.device(device)
    s0 = torch.zeros(k, 8, device=dev)
    s0[:, 2::2] = torch.pi
    u = torch.randn(h, k, device=dev) * 20.0

    def run():
        s = s0
        for t in range(h):
            s = step(plant, s, u[t])
        return s

    run()
    if dev.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    if dev.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--h", type=int, default=180)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for d in devices:
        sec = bench(d, args.k, args.h, args.reps)
        print(f"{d}: K={args.k} H={args.h} rollout {sec * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
