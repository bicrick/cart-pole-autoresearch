"""Multicore numba (float64) build of the n-link rollout kernel: tests and local runs."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mppi.nlink.kernel_body import MAX_D, MAX_DIM, build

_one = build(njit, np.float64)


@njit(parallel=True)
def rollout_costs(scal, links, K, P, starts, plans, per_start):
    """starts [B, dim]; plans [B * per_start, T] (sample i starts from starts[i // per_start]) -> cost [N]."""
    n = links.shape[1]
    N, T = plans.shape
    out = np.empty(N)
    for i in prange(N):
        st = np.empty(MAX_DIM)
        sn = np.empty(MAX_D)
        cs = np.empty(MAX_D)
        tc = np.empty(MAX_D)
        ts = np.empty(MAX_D)
        A = np.empty(MAX_D * MAX_D)
        x = np.empty(MAX_DIM)
        out[i] = _one(scal, links, K, P, starts[i // per_start], plans[i], n, T, st, sn, cs, tc, ts, A, x)
    return out
