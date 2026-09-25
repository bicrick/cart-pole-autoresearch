"""One MPPI sample's rollout cost for the n-link plant, shared by both kernels.

``build(jit, F)`` compiles the same source with ``numba.njit`` and F = float64
(kernel_cpu) or ``numba.cuda.jit(device=True)`` and F = float32 (kernel_cuda),
so the CPU reference and the GPU kernel cannot drift apart. Every constant goes
through F: a bare Python float literal would promote the GPU math to FP64,
which an L4 runs at 1/64 rate.

The math is ``fast_rollout._rollout_one`` for n links: gated target-LQR anchor
+ residual knot, clamps, mass-matrix solve (Cholesky, 1e-8 diagonal jitter like
physics_triple), semi-implicit Euler, running cost, early exit off the track,
then terminal and smoothness costs. Scratch arrays are passed in (CUDA local
arrays or numpy), sized for up to MAX_LINKS links.
"""

from __future__ import annotations

import math

MAX_LINKS = 6
MAX_DIM = 2 + 2 * MAX_LINKS
MAX_D = MAX_LINKS + 1


def build(jit, F):
    zero = F(0.0)
    one = F(1.0)
    half = F(0.5)
    pi = F(math.pi)
    two_pi = F(2.0 * math.pi)
    max_cart_vel = F(30.0)
    max_ang_vel = F(50.0)
    max_acc = F(1e4)
    jitter = F(1e-8)
    big = F(3.0e38)

    @jit
    def clip(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    @jit
    def fin(v):
        # NaN fails v == v; inf fails the magnitude test.
        return v if (v == v and abs(v) < big) else zero

    @jit
    def wrap(a):
        return a - two_pi * F(math.floor((a + pi) / two_pi))

    @jit
    def chol_solve(A, x, D):
        """Solve SPD A x = rhs in place (A row-major D x D, lower triangle used)."""
        for j in range(D):
            s = A[j * D + j]
            for k in range(j):
                s -= A[j * D + k] * A[j * D + k]
            ljj = math.sqrt(s)
            A[j * D + j] = ljj
            for i in range(j + 1, D):
                t = A[i * D + j]
                for k in range(j):
                    t -= A[i * D + k] * A[j * D + k]
                A[i * D + j] = t / ljj
        for i in range(D):
            t = x[i]
            for k in range(i):
                t -= A[i * D + k] * x[k]
            x[i] = t / A[i * D + i]
        for ii in range(D):
            i = D - 1 - ii
            t = x[i]
            for k in range(i + 1, D):
                t -= A[k * D + i] * x[k]
            x[i] = t / A[i * D + i]

    @jit
    def pole_energy(links, st, sn, cs, n, g):
        vx = zero
        vy = zero
        y = zero
        e = zero
        for i in range(n):
            li = links[1, i]
            w = st[3 + 2 * i]
            vx += li * cs[i] * w
            vy -= li * sn[i] * w
            y += li * cs[i]
            e += links[0, i] * (half * (vx * vx + vy * vy) + g * y)
        return e

    @jit
    def du_cost(U, T, w_du, kdt):
        acc = zero
        for j in range(T - 1):
            du = (U[j + 1] - U[j]) / kdt
            acc += w_du * du * du * kdt
        return acc

    @jit
    def trig(st, sn, cs, n):
        for i in range(n):
            th = st[2 + 2 * i]
            sn[i] = math.sin(th)
            cs[i] = math.cos(th)

    @jit
    def rollout_one(scal, links, K, P, s0, U, n, T, st, sn, cs, tc, ts, A, x):
        M = scal[0]
        g = scal[1]
        b = scal[2]
        dt = scal[3]
        fmax = scal[4]
        e_t = scal[5]
        w_angle = scal[6]
        w_vel = scal[7]
        w_vel_near = scal[8]
        near_scale = scal[9]
        w_x = scal[10]
        w_xd = scal[11]
        x_soft = scal[12]
        w_barrier = scal[13]
        x_dead = scal[14]
        w_dead = scal[15]
        w_energy = scal[16]
        w_u = scal[17]
        w_du = scal[18]
        w_tl = scal[19]
        w_te = scal[20]
        g_in = scal[21]
        g_out = scal[22]
        anchor_on = scal[23]
        knot = int(scal[24])
        early = scal[25] > zero
        D = n + 1
        dim = 2 + 2 * n
        kdt = scal[24] * dt
        total = T * knot
        msum = zero
        for i in range(n):
            msum += links[0, i]
            tc[i] = math.cos(links[4, i])
            ts[i] = math.sin(links[4, i])
        for k in range(dim):
            st[k] = s0[k]
        trig(st, sn, cs, n)
        acc = zero
        done = 0
        for j in range(T):
            r = U[j]
            for q in range(knot):
                # gated target-LQR anchor + residual
                u = r
                if anchor_on > zero:
                    d = zero
                    for i in range(n):
                        d += one - (cs[i] * tc[i] + sn[i] * ts[i])
                    gate = clip((g_out - d) / (g_out - g_in), zero, one)
                    if gate > zero:
                        fb = K[0] * st[0] + K[1] * st[1]
                        for i in range(n):
                            fb += K[2 + 2 * i] * wrap(st[2 + 2 * i] - links[4, i]) + K[3 + 2 * i] * st[3 + 2 * i]
                        u = r - gate * fb
                u = fin(clip(u, -fmax, fmax))
                # plant step (nlink.plant.step)
                st[1] = clip(st[1], -max_cart_vel, max_cart_vel)
                for i in range(n):
                    st[3 + 2 * i] = clip(st[3 + 2 * i], -max_ang_vel, max_ang_vel)
                A[0] = M + msum + jitter
                x[0] = u - b * st[1]
                for i in range(n):
                    li = links[1, i]
                    Si = links[3, i]
                    wi = st[3 + 2 * i]
                    v = Si * li * cs[i]
                    A[1 + i] = v
                    A[(1 + i) * D] = v
                    x[0] += Si * li * sn[i] * wi * wi
                    ri = -links[2, i] * wi + Si * g * li * sn[i]
                    for k in range(n):
                        Sm = links[3, i] if i > k else links[3, k]
                        lk = links[1, k]
                        A[(1 + i) * D + 1 + k] = Sm * li * lk * (cs[i] * cs[k] + sn[i] * sn[k])
                        if k != i:
                            wk = st[3 + 2 * k]
                            ri -= Sm * li * lk * (sn[i] * cs[k] - cs[i] * sn[k]) * wk * wk
                    A[(1 + i) * D + 1 + i] += jitter
                    x[1 + i] = ri
                chol_solve(A, x, D)
                st[1] = clip(st[1] + clip(fin(x[0]), -max_acc, max_acc) * dt, -max_cart_vel, max_cart_vel)
                for i in range(n):
                    st[3 + 2 * i] = clip(st[3 + 2 * i] + clip(fin(x[1 + i]), -max_acc, max_acc) * dt,
                                         -max_ang_vel, max_ang_vel)
                st[0] += st[1] * dt
                for i in range(n):
                    st[2 + 2 * i] += st[3 + 2 * i] * dt
                trig(st, sn, cs, n)
                # running cost on the new state
                ang = zero
                spin = zero
                for i in range(n):
                    ang += one - (cs[i] * tc[i] + sn[i] * ts[i])
                    spin += st[3 + 2 * i] * st[3 + 2 * i]
                near = math.exp(-ang / near_scale)
                ax = abs(st[0])
                over = ax - x_soft if ax > x_soft else zero
                dead = one if ax > x_dead else zero
                en = pole_energy(links, st, sn, cs, n, g) - e_t
                acc += (
                    w_angle * ang
                    + (w_vel + w_vel_near * near) * spin
                    + w_x * st[0] * st[0]
                    + w_xd * st[1] * st[1]
                    + w_barrier * over * over
                    + w_dead * dead
                    + w_energy * (one - near) * en * en
                    + w_u * u * u
                ) * dt
                done += 1
                if early and dead > zero:
                    return acc + w_dead * dt * F(total - done) + du_cost(U, T, w_du, kdt)
        # terminal: LQR cost-to-go near the target, energy gap away from it (x reused as e)
        for k in range(dim):
            x[k] = st[k]
        for i in range(n):
            x[2 + 2 * i] = wrap(st[2 + 2 * i] - links[4, i])
        quad = zero
        for a in range(dim):
            row = zero
            for c in range(dim):
                row += P[a, c] * x[c]
            quad += x[a] * row
        ang = zero
        for i in range(n):
            ang += one - (cs[i] * tc[i] + sn[i] * ts[i])
        near = math.exp(-ang / near_scale)
        en = pole_energy(links, st, sn, cs, n, g) - e_t
        acc += near * w_tl * quad + (one - near) * w_te * en * en
        return acc + du_cost(U, T, w_du, kdt)

    return rollout_one
