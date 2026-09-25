/**
 * MPPI sample cost for the cart + n-link pendulum: train/mppi/nlink/kernel_body.py
 * `rollout_one` in f64 (gated target-LQR anchor + residual knot, velocity
 * clamps, Cholesky mass-matrix solve with 1e-8 jitter, semi-implicit Euler,
 * running cost, early exit off the track, terminal LQR / energy cost).
 *
 * `p` is the exported layout (train/mppi/export_mppi_nlink.py): the NS kernel
 * scalars, then the per-link rows m, l, c, S, target (row-major, n each).
 */
const TWO_PI = 2 * Math.PI;
const MAX_CART_VEL = 30;
const MAX_ANG_VEL = 50;
const MAX_ACC = 1e4;
const JITTER = 1e-8;

export const NS = 26;
export const ROW_M = 0;
export const ROW_L = 1;
export const ROW_C = 2;
export const ROW_S = 3;
export const ROW_T = 4;

function clip(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function fin(v) {
  return v === v && Math.abs(v) < 3e38 ? v : 0;
}

function wrap(a) {
  return a - TWO_PI * Math.floor((a + Math.PI) / TWO_PI);
}

/** Solve SPD A x = rhs in place (A row-major D x D, lower triangle used). */
export function cholSolve(A, x, D) {
  for (let j = 0; j < D; j += 1) {
    let s = A[j * D + j];
    for (let k = 0; k < j; k += 1) s -= A[j * D + k] * A[j * D + k];
    const ljj = Math.sqrt(s);
    A[j * D + j] = ljj;
    for (let i = j + 1; i < D; i += 1) {
      let t = A[i * D + j];
      for (let k = 0; k < j; k += 1) t -= A[i * D + k] * A[j * D + k];
      A[i * D + j] = t / ljj;
    }
  }
  for (let i = 0; i < D; i += 1) {
    let t = x[i];
    for (let k = 0; k < i; k += 1) t -= A[i * D + k] * x[k];
    x[i] = t / A[i * D + i];
  }
  for (let i = D - 1; i >= 0; i -= 1) {
    let t = x[i];
    for (let k = i + 1; k < D; k += 1) t -= A[k * D + i] * x[k];
    x[i] = t / A[i * D + i];
  }
}

/** Rollout kernel for `n` links: (p, K, P, s0, so, U, uo, nKnots, fin?) -> cost. */
export function makeRolloutNlink(n) {
  const D = n + 1;
  const dim = 2 + 2 * n;
  const st = new Float64Array(dim);
  const sn = new Float64Array(n);
  const cs = new Float64Array(n);
  const tc = new Float64Array(n);
  const ts = new Float64Array(n);
  const A = new Float64Array(D * D);
  const x = new Float64Array(dim);
  const L = (p, row, i) => p[NS + row * n + i];

  function trig() {
    for (let i = 0; i < n; i += 1) {
      const th = st[2 + 2 * i];
      sn[i] = Math.sin(th);
      cs[i] = Math.cos(th);
    }
  }

  function poleEnergy(p, g) {
    let vx = 0;
    let vy = 0;
    let y = 0;
    let e = 0;
    for (let i = 0; i < n; i += 1) {
      const li = L(p, ROW_L, i);
      const w = st[3 + 2 * i];
      vx += li * cs[i] * w;
      vy -= li * sn[i] * w;
      y += li * cs[i];
      e += L(p, ROW_M, i) * (0.5 * (vx * vx + vy * vy) + g * y);
    }
    return e;
  }

  function duCost(U, uo, T, wDu, kdt) {
    let acc = 0;
    for (let j = 0; j < T - 1; j += 1) {
      const du = (U[uo + j + 1] - U[uo + j]) / kdt;
      acc += wDu * du * du * kdt;
    }
    return acc;
  }

  return function rolloutCostNlink(p, K, P, s0, so, U, uo, T, out = null) {
    const M = p[0], g = p[1], b = p[2], dt = p[3], fmax = p[4], eT = p[5];
    const wAngle = p[6], wVel = p[7], wVelNear = p[8], nearScale = p[9];
    const wX = p[10], wXd = p[11], xSoft = p[12], wBarrier = p[13];
    const xDead = p[14], wDead = p[15], wEnergy = p[16], wU = p[17], wDu = p[18];
    const wTl = p[19], wTe = p[20], gIn = p[21], gOut = p[22], anchorOn = p[23];
    const knot = p[24] | 0;
    const early = p[25] > 0;
    const kdt = p[24] * dt;
    const total = T * knot;
    let msum = 0;
    for (let i = 0; i < n; i += 1) {
      msum += L(p, ROW_M, i);
      tc[i] = Math.cos(L(p, ROW_T, i));
      ts[i] = Math.sin(L(p, ROW_T, i));
    }
    for (let k = 0; k < dim; k += 1) st[k] = s0[so + k];
    trig();
    let acc = 0;
    let done = 0;
    for (let j = 0; j < T; j += 1) {
      const r = U[uo + j];
      for (let q = 0; q < knot; q += 1) {
        let u = r;
        if (anchorOn > 0) {
          let d = 0;
          for (let i = 0; i < n; i += 1) d += 1 - (cs[i] * tc[i] + sn[i] * ts[i]);
          const gate = clip((gOut - d) / (gOut - gIn), 0, 1);
          if (gate > 0) {
            let fb = K[0] * st[0] + K[1] * st[1];
            for (let i = 0; i < n; i += 1) {
              fb += K[2 + 2 * i] * wrap(st[2 + 2 * i] - L(p, ROW_T, i)) + K[3 + 2 * i] * st[3 + 2 * i];
            }
            u = r - gate * fb;
          }
        }
        u = fin(clip(u, -fmax, fmax));
        st[1] = clip(st[1], -MAX_CART_VEL, MAX_CART_VEL);
        for (let i = 0; i < n; i += 1) st[3 + 2 * i] = clip(st[3 + 2 * i], -MAX_ANG_VEL, MAX_ANG_VEL);
        A[0] = M + msum + JITTER;
        x[0] = u - b * st[1];
        for (let i = 0; i < n; i += 1) {
          const li = L(p, ROW_L, i);
          const Si = L(p, ROW_S, i);
          const wi = st[3 + 2 * i];
          const v = Si * li * cs[i];
          A[1 + i] = v;
          A[(1 + i) * D] = v;
          x[0] += Si * li * sn[i] * wi * wi;
          let ri = -L(p, ROW_C, i) * wi + Si * g * li * sn[i];
          for (let k = 0; k < n; k += 1) {
            const Sm = i > k ? Si : L(p, ROW_S, k);
            const lk = L(p, ROW_L, k);
            A[(1 + i) * D + 1 + k] = Sm * li * lk * (cs[i] * cs[k] + sn[i] * sn[k]);
            if (k !== i) {
              const wk = st[3 + 2 * k];
              ri -= Sm * li * lk * (sn[i] * cs[k] - cs[i] * sn[k]) * wk * wk;
            }
          }
          A[(1 + i) * D + 1 + i] += JITTER;
          x[1 + i] = ri;
        }
        cholSolve(A, x, D);
        st[1] = clip(st[1] + clip(fin(x[0]), -MAX_ACC, MAX_ACC) * dt, -MAX_CART_VEL, MAX_CART_VEL);
        for (let i = 0; i < n; i += 1) {
          st[3 + 2 * i] = clip(st[3 + 2 * i] + clip(fin(x[1 + i]), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
        }
        st[0] += st[1] * dt;
        for (let i = 0; i < n; i += 1) st[2 + 2 * i] += st[3 + 2 * i] * dt;
        trig();
        let ang = 0;
        let spin = 0;
        for (let i = 0; i < n; i += 1) {
          ang += 1 - (cs[i] * tc[i] + sn[i] * ts[i]);
          spin += st[3 + 2 * i] * st[3 + 2 * i];
        }
        const near = Math.exp(-ang / nearScale);
        const ax = Math.abs(st[0]);
        const over = ax > xSoft ? ax - xSoft : 0;
        const dead = ax > xDead ? 1 : 0;
        const en = poleEnergy(p, g) - eT;
        acc += (
          wAngle * ang
          + (wVel + wVelNear * near) * spin
          + wX * st[0] * st[0]
          + wXd * st[1] * st[1]
          + wBarrier * over * over
          + wDead * dead
          + wEnergy * (1 - near) * en * en
          + wU * u * u
        ) * dt;
        done += 1;
        if (early && dead > 0) {
          if (out) out.set(st);
          return acc + wDead * dt * (total - done) + duCost(U, uo, T, wDu, kdt);
        }
      }
    }
    if (out) out.set(st);
    for (let k = 0; k < dim; k += 1) x[k] = st[k];
    for (let i = 0; i < n; i += 1) x[2 + 2 * i] = wrap(st[2 + 2 * i] - L(p, ROW_T, i));
    let quad = 0;
    for (let a = 0; a < dim; a += 1) {
      let row = 0;
      for (let c = 0; c < dim; c += 1) row += P[a * dim + c] * x[c];
      quad += x[a] * row;
    }
    let ang = 0;
    for (let i = 0; i < n; i += 1) ang += 1 - (cs[i] * tc[i] + sn[i] * ts[i]);
    const near = Math.exp(-ang / nearScale);
    const en = poleEnergy(p, g) - eT;
    acc += near * wTl * quad + (1 - near) * wTe * en * en;
    return acc + duCost(U, uo, T, wDu, kdt);
  };
}
