"""numba.cuda (float32) build of the n-link rollout kernel: one thread per sample.

Takes any __cuda_array_interface__ arrays (torch CUDA tensors included), so the
MPPI loop keeps samples on the GPU. Local runs without a GPU can exercise it
with NUMBA_ENABLE_CUDASIM=1 (slow, for tests only).
"""

from __future__ import annotations

import numpy as np
from numba import cuda

from mppi.nlink.kernel_body import MAX_D, MAX_DIM, build

_one = build(cuda.jit(device=True), np.float32)
THREADS = 128
# Local-array shapes must be compile-time literals, so precompute the product here.
MAX_A = MAX_D * MAX_D


@cuda.jit
def _kernel(scal, links, K, P, starts, plans, per_start, out):
    i = cuda.grid(1)
    if i >= plans.shape[0]:
        return
    st = cuda.local.array(MAX_DIM, np.float32)
    sn = cuda.local.array(MAX_D, np.float32)
    cs = cuda.local.array(MAX_D, np.float32)
    tc = cuda.local.array(MAX_D, np.float32)
    ts = cuda.local.array(MAX_D, np.float32)
    A = cuda.local.array(MAX_A, np.float32)
    x = cuda.local.array(MAX_DIM, np.float32)
    out[i] = _one(scal, links, K, P, starts[i // per_start], plans[i], links.shape[1], plans.shape[1],
                  st, sn, cs, tc, ts, A, x)


def rollout_costs(scal, links, K, P, starts, plans, per_start, out):
    """All arrays float32 on the device; fills out [N]."""
    n = plans.shape[0]
    _kernel[(n + THREADS - 1) // THREADS, THREADS](scal, links, K, P, starts, plans, per_start, out)
    return out
