#!/usr/bin/env python3
"""n-link rollout kernels vs references.

1. n=3 CPU kernel vs the triple's ``fast_rollout`` (same K, P and weights).
2. CUDA (f32) kernel vs the CPU (f64) kernel for n=4: bulk relative error and
   the 32 cheapest samples. Without a GPU run under the CUDA simulator:
   NUMBA_ENABLE_CUDASIM=1 python -m mppi.nlink.test_kernels --cuda --samples 16

    python -m mppi.nlink.test_kernels [--cuda] [--samples 4096]
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from mppi.nlink.costs import pack
from mppi.nlink.kernel_cpu import rollout_costs as cpu_costs
from mppi.nlink.plant import NLinkPlant


def random_starts(n: int, count: int, g: np.random.Generator) -> np.ndarray:
    s = np.zeros((count, 2 + 2 * n))
    s[:, 0] = g.uniform(-1.5, 1.5, count)
    s[:, 1] = g.uniform(-2, 2, count)
    s[:, 2::2] = g.uniform(-math.pi, math.pi, (count, n))
    s[:, 3::2] = g.uniform(-6, 6, (count, n))
    return s


def vs_fast_rollout() -> bool:
    from mppi.config import build_teacher
    from mppi.fast_rollout import pack_params, rollout_costs

    plant, ctrl, _ = build_teacher({"cost": {"goal": "UUU"}, "mppi": {"compile": False}})
    p3, K3, P3 = pack_params(plant, ctrl.cost, ctrl.cfg)
    pk = pack(NLinkPlant.load(3), "UUU", knot=ctrl.cfg.knot, early_exit=False)
    g = np.random.default_rng(0)
    starts = random_starts(3, 256, g)
    plans = g.normal(0, 15, (256, ctrl.cfg.n_knots))
    ref = rollout_costs(p3, K3, P3, starts, plans)
    got = cpu_costs(pk.scal, pk.links, K3, P3, starts, plans, 1)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)
    print(f"n=3 kernel vs fast_rollout: rel err median {np.median(rel):.1e}, p95 {np.percentile(rel, 95):.1e}")
    return np.median(rel) < 1e-9 and np.percentile(rel, 95) < 1e-5


def vs_cuda(samples: int) -> bool:
    from numba import cuda

    from mppi.nlink.kernel_cuda import rollout_costs as gpu_costs

    pk = pack(NLinkPlant.load(4), "UUUU", knot=4)
    g = np.random.default_rng(1)
    per = samples // 2
    starts = random_starts(4, 2, g)
    plans = g.normal(0, 15, (2 * per, 45))
    ref = cpu_costs(pk.scal, pk.links, pk.K, pk.P, starts, plans, per)
    f32 = lambda a: cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))
    out = cuda.device_array(2 * per, dtype=np.float32)
    args = (f32(pk.scal), f32(pk.links), f32(pk.K), f32(pk.P), f32(starts), f32(plans), per, out)
    gpu_costs(*args)
    cuda.synchronize()
    t0 = time.perf_counter()
    gpu_costs(*args)
    cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    got = out.copy_to_host().astype(np.float64)
    rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)
    k = min(32, per)
    overlap = len(set(np.argsort(ref[:per])[:k]) & set(np.argsort(got[:per])[:k]))
    print(f"n=4 cuda f32 vs cpu f64 ({2 * per} samples): rel err median {np.median(rel):.1e}, "
          f"p95 {np.percentile(rel, 95):.1e}; top-{k} overlap {overlap}/{k}; {ms:.1f} ms")
    return np.median(rel) < 1e-3 and overlap >= int(0.8 * k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--samples", type=int, default=4096)
    args = ap.parse_args()
    ok = vs_fast_rollout()
    if args.cuda:
        ok = vs_cuda(args.samples) and ok
    print("NLINK KERNELS OK" if ok else "NLINK KERNELS FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
