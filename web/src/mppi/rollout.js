/**
 * MPPI sample cost for the cart-triple: port of train/mppi/fast_rollout.py.
 *
 * One sample runs its whole horizon (gated LQR anchor + residual knot -> plant
 * step -> running cost), then the terminal cost and the residual smoothness.
 * `p` is the packed scalar layout from train/mppi/export_mppi.py (FIELDS below),
 * K[8] and P[64] (row-major) are the goal's LQR gain and cost-to-go.
 */
const MAX_CART_VEL = 30;
const MAX_ANG_VEL = 50;
const MAX_ACC = 1e4;
const TWO_PI = 2 * Math.PI;

export const FIELDS = [
  "M", "m1", "m2", "m3", "l1", "l2", "l3", "g", "b", "c1", "c2", "c3", "dt", "fmax",
  "t1", "t2", "t3", "e_target",
  "w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft", "w_barrier",
  "x_dead", "w_dead", "w_energy", "w_u", "w_du", "w_terminal_lqr", "w_terminal_energy",
  "gate_in", "gate_out", "anchor_on", "knot", "sub", "early_exit",
];

// Rotation-updated sin/cos (see rolloutCost). The exact libm path is kept for
// bit-level parity tests against the numba kernel.
let fastTrig = true;

export function setFastTrig(on) {
  fastTrig = Boolean(on);
}

export function fastTrigOn() {
  return fastTrig;
}

function wrap(a) {
  // Python's (a + pi) % 2pi - pi (floored modulo).
  const r = (a + Math.PI) % TWO_PI;
  return (r < 0 ? r + TWO_PI : r) - Math.PI;
}

function clip(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function fin(v) {
  return Number.isFinite(v) ? v : 0;
}

/** Cost of one residual plan U (offset `uo`, `nKnots` knots) from state s0 (offset `so`). */
export function rolloutCost(p, K, P, s0, so, U, uo, nKnots) {
  const M = p[0], m1 = p[1], m2 = p[2], m3 = p[3];
  const l1 = p[4], l2 = p[5], l3 = p[6], g = p[7];
  const b = p[8], cd1 = p[9], cd2 = p[10], cd3 = p[11];
  const fmax = p[13];
  const t1 = p[14], t2 = p[15], t3 = p[16], eT = p[17];
  const wAngle = p[18], wVel = p[19], wVelNear = p[20], nearScale = p[21];
  const wX = p[22], wXd = p[23], xSoft = p[24], wBarrier = p[25];
  const xDead = p[26], wDead = p[27], wEnergy = p[28], wU = p[29], wDu = p[30];
  const wTl = p[31], wTe = p[32];
  const gIn = p[33], gOut = p[34], anchorOn = p[35];
  const knot = p[36] | 0;
  const sub = p[37] | 0 || 1;
  const early = p[38] > 0;
  const kdt = knot * p[12];
  // Rollouts may integrate at sub * dt (fewer, coarser steps per knot).
  const dt = p[12] * sub;
  const stepsPerKnot = (knot / sub) | 0;
  const total = nKnots * stepsPerKnot;
  let done = 0;
  const ct1 = Math.cos(t1), st1 = Math.sin(t1);
  const ct2 = Math.cos(t2), st2 = Math.sin(t2);
  const ct3 = Math.cos(t3), st3 = Math.sin(t3);
  const m123 = m1 + m2 + m3;
  const m23 = m2 + m3;
  const eps = 1e-8;
  const a00 = M + m123 + eps;
  const a11 = m123 * l1 * l1 + eps;
  const a22 = m23 * l2 * l2 + eps;
  const a33 = m3 * l3 * l3 + eps;
  const k0 = K[0], k1 = K[1], k2 = K[2], k3 = K[3], k4 = K[4], k5 = K[5], k6 = K[6], k7 = K[7];
  const gSpan = gOut - gIn;

  let x = s0[so], xd = s0[so + 1];
  let th1 = s0[so + 2], w1 = s0[so + 3];
  let th2 = s0[so + 4], w2 = s0[so + 5];
  let th3 = s0[so + 6], w3 = s0[so + 7];
  let s1 = Math.sin(th1), c1 = Math.cos(th1);
  let s2 = Math.sin(th2), c2 = Math.cos(th2);
  let s3 = Math.sin(th3), c3 = Math.cos(th3);
  let acc = 0;

  for (let j = 0; j < nKnots; j += 1) {
    const r = U[uo + j];
    for (let q = 0; q < stepsPerKnot; q += 1) {
      let fbU = 0;
      if (anchorOn > 0) {
        const d = (1 - (c1 * ct1 + s1 * st1)) + (1 - (c2 * ct2 + s2 * st2)) + (1 - (c3 * ct3 + s3 * st3));
        const gate = clip((gOut - d) / gSpan, 0, 1);
        if (gate > 0) {
          const e1 = wrap(th1 - t1);
          const e2 = wrap(th2 - t2);
          const e3 = wrap(th3 - t3);
          fbU = -gate * (k0 * x + k1 * xd + k2 * e1 + k3 * w1 + k4 * e2 + k5 * w2 + k6 * e3 + k7 * w3);
        }
      }
      let u = clip(fbU + r, -fmax, fmax);
      if (!Number.isFinite(u)) u = 0;

      xd = clip(xd, -MAX_CART_VEL, MAX_CART_VEL);
      w1 = clip(w1, -MAX_ANG_VEL, MAX_ANG_VEL);
      w2 = clip(w2, -MAX_ANG_VEL, MAX_ANG_VEL);
      w3 = clip(w3, -MAX_ANG_VEL, MAX_ANG_VEL);
      const s12 = s1 * c2 - c1 * s2, c12 = c1 * c2 + s1 * s2;
      const s13 = s1 * c3 - c1 * s3, c13 = c1 * c3 + s1 * s3;
      const s23 = s2 * c3 - c2 * s3, c23 = c2 * c3 + s2 * s3;
      const a01 = m123 * l1 * c1;
      const a02 = m23 * l2 * c2;
      const a03 = m3 * l3 * c3;
      const a12 = m23 * l1 * l2 * c12;
      const a13 = m3 * l1 * l3 * c13;
      const a23 = m3 * l2 * l3 * c23;
      const q1s = w1 * w1, q2s = w2 * w2, q3s = w3 * w3;
      const r0 = u - b * xd + m123 * l1 * s1 * q1s + m23 * l2 * s2 * q2s + m3 * l3 * s3 * q3s;
      const r1 = -cd1 * w1 - m23 * l1 * l2 * s12 * q2s - m3 * l1 * l3 * s13 * q3s + m123 * g * l1 * s1;
      const r2 = -cd2 * w2 + m23 * l1 * l2 * s12 * q1s - m3 * l2 * l3 * s23 * q3s + m23 * g * l2 * s2;
      const r3 = -cd3 * w3 + m3 * l1 * l3 * s13 * q1s + m3 * l2 * l3 * s23 * q2s + m3 * g * l3 * s3;
      const d0 = a00;
      const l10 = a01 / d0, l20 = a02 / d0, l30 = a03 / d0;
      const d1 = a11 - l10 * l10 * d0;
      const l21 = (a12 - l20 * l10 * d0) / d1;
      const l31 = (a13 - l30 * l10 * d0) / d1;
      const d2 = a22 - l20 * l20 * d0 - l21 * l21 * d1;
      const l32 = (a23 - l30 * l20 * d0 - l31 * l21 * d1) / d2;
      const d3 = a33 - l30 * l30 * d0 - l31 * l31 * d1 - l32 * l32 * d2;
      const y0 = r0;
      const y1 = r1 - l10 * y0;
      const y2 = r2 - l20 * y0 - l21 * y1;
      const y3 = r3 - l30 * y0 - l31 * y1 - l32 * y2;
      const x3 = y3 / d3;
      const x2 = y2 / d2 - l32 * x3;
      const x1 = y1 / d1 - l21 * x2 - l31 * x3;
      const x0 = y0 / d0 - l10 * x1 - l20 * x2 - l30 * x3;
      xd = clip(xd + clip(fin(x0), -MAX_ACC, MAX_ACC) * dt, -MAX_CART_VEL, MAX_CART_VEL);
      w1 = clip(w1 + clip(fin(x1), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
      w2 = clip(w2 + clip(fin(x2), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
      w3 = clip(w3 + clip(fin(x3), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
      x += xd * dt;
      const a1 = w1 * dt, a2 = w2 * dt, a3 = w3 * dt;
      th1 += a1;
      th2 += a2;
      th3 += a3;
      if (fastTrig) {
        // Rotate (sin, cos) by the step angle (|a| <= 50 rad/s * dt): Taylor
        // sin to a^7 and cos to a^8, error < 1e-6 at sub = 2.
        let q = a1 * a1;
        let sa = a1 * (1 - q * (1 / 6 - q * (1 / 120 - q / 5040)));
        let ca = 1 - q * (0.5 - q * (1 / 24 - q * (1 / 720 - q / 40320)));
        let t = s1 * ca + c1 * sa;
        c1 = c1 * ca - s1 * sa;
        s1 = t;
        q = a2 * a2;
        sa = a2 * (1 - q * (1 / 6 - q * (1 / 120 - q / 5040)));
        ca = 1 - q * (0.5 - q * (1 / 24 - q * (1 / 720 - q / 40320)));
        t = s2 * ca + c2 * sa;
        c2 = c2 * ca - s2 * sa;
        s2 = t;
        q = a3 * a3;
        sa = a3 * (1 - q * (1 / 6 - q * (1 / 120 - q / 5040)));
        ca = 1 - q * (0.5 - q * (1 / 24 - q * (1 / 720 - q / 40320)));
        t = s3 * ca + c3 * sa;
        c3 = c3 * ca - s3 * sa;
        s3 = t;
      } else {
        s1 = Math.sin(th1); c1 = Math.cos(th1);
        s2 = Math.sin(th2); c2 = Math.cos(th2);
        s3 = Math.sin(th3); c3 = Math.cos(th3);
      }

      const ang = (1 - (c1 * ct1 + s1 * st1)) + (1 - (c2 * ct2 + s2 * st2)) + (1 - (c3 * ct3 + s3 * st3));
      const near = Math.exp(-ang / nearScale);
      const spin = w1 * w1 + w2 * w2 + w3 * w3;
      const ax = Math.abs(x);
      const over = ax > xSoft ? ax - xSoft : 0;
      const dead = ax > xDead ? 1 : 0;
      const en = poleEnergy(l1, l2, l3, m1, m2, m3, g, s1, c1, w1, s2, c2, w2, s3, c3, w3) - eT;
      acc += (
        wAngle * ang
        + (wVel + wVelNear * near) * spin
        + wX * x * x
        + wXd * xd * xd
        + wBarrier * over * over
        + wDead * dead
        + wEnergy * (1 - near) * en * en
        + wU * u * u
      ) * dt;
      done += 1;
      if (early && dead) {
        // Off the track for good: charge the dead penalty for the rest.
        return acc + wDead * dt * (total - done) + duCost(U, uo, nKnots, wDu, kdt);
      }
    }
  }

  const e0 = x, e1 = xd, e2 = wrap(th1 - t1), e3 = w1, e4 = wrap(th2 - t2), e5 = w2, e6 = wrap(th3 - t3), e7 = w3;
  const quad = quadForm(P, e0, e1, e2, e3, e4, e5, e6, e7);
  const ang = (1 - (c1 * ct1 + s1 * st1)) + (1 - (c2 * ct2 + s2 * st2)) + (1 - (c3 * ct3 + s3 * st3));
  const near = Math.exp(-ang / nearScale);
  const en = poleEnergy(l1, l2, l3, m1, m2, m3, g, s1, c1, w1, s2, c2, w2, s3, c3, w3) - eT;
  acc += near * wTl * quad + (1 - near) * wTe * en * en;
  return acc + duCost(U, uo, nKnots, wDu, kdt);
}

function duCost(U, uo, nKnots, wDu, kdt) {
  let acc = 0;
  for (let j = 0; j < nKnots - 1; j += 1) {
    const du = (U[uo + j + 1] - U[uo + j]) / kdt;
    acc += wDu * du * du * kdt;
  }
  return acc;
}

/** e^T P e for row-major P[64], without allocating e. */
function quadForm(P, e0, e1, e2, e3, e4, e5, e6, e7) {
  let quad = 0;
  for (let i = 0; i < 8; i += 1) {
    const o = i * 8;
    const row = P[o] * e0 + P[o + 1] * e1 + P[o + 2] * e2 + P[o + 3] * e3
      + P[o + 4] * e4 + P[o + 5] * e5 + P[o + 6] * e6 + P[o + 7] * e7;
    const ei = i === 0 ? e0 : i === 1 ? e1 : i === 2 ? e2 : i === 3 ? e3 : i === 4 ? e4 : i === 5 ? e5 : i === 6 ? e6 : e7;
    quad += ei * row;
  }
  return quad;
}

function poleEnergy(l1, l2, l3, m1, m2, m3, g, s1, c1, w1, s2, c2, w2, s3, c3, w3) {
  const v1x = l1 * c1 * w1;
  const v1y = -l1 * s1 * w1;
  const v2x = v1x + l2 * c2 * w2;
  const v2y = v1y - l2 * s2 * w2;
  const v3x = v2x + l3 * c3 * w3;
  const v3y = v2y - l3 * s3 * w3;
  const ke = 0.5 * (m1 * (v1x * v1x + v1y * v1y) + m2 * (v2x * v2x + v2y * v2y) + m3 * (v3x * v3x + v3y * v3y));
  const y1 = l1 * c1;
  const y2 = y1 + l2 * c2;
  const y3 = y2 + l3 * c3;
  return ke + g * (m1 * y1 + m2 * y2 + m3 * y3);
}

/** Costs for `n` plans (rows of `plans`, stride nKnots) from one start state, into `out`. */
export function rolloutCosts(p, K, P, s0, plans, n, nKnots, out) {
  for (let i = 0; i < n; i += 1) out[i] = rolloutCost(p, K, P, s0, 0, plans, i * nKnots, nKnots);
  return out;
}
