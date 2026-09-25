/**
 * MPPI sample cost for the cart + double pendulum: the triple kernel
 * (rollout.js) with two links, stepping exactly like the page's plant
 * (physics.js: force clip, closed-form 3x3 solve, semi-implicit Euler, no
 * velocity clamps). `p` follows FIELDS (train/mppi/export_mppi_double.py);
 * K[6] and P[36] (row-major) are the goal's LQR gain and cost-to-go.
 */
import { fastTrigOn } from "./rollout.js";

const TWO_PI = 2 * Math.PI;

export const FIELDS = [
  "M", "m1", "m2", "l1", "l2", "g", "b", "c1", "c2", "dt", "fmax",
  "t1", "t2", "e_target",
  "w_angle", "w_vel", "w_vel_near", "near_scale", "w_x", "w_xd", "x_soft", "w_barrier",
  "x_dead", "w_dead", "w_energy", "w_u", "w_du", "w_terminal_lqr", "w_terminal_energy",
  "gate_in", "gate_out", "anchor_on", "knot", "sub", "early_exit",
];

function wrap(a) {
  const r = (a + Math.PI) % TWO_PI;
  return (r < 0 ? r + TWO_PI : r) - Math.PI;
}

function clip(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function poleEnergy(l1, l2, m1, m2, g, s1, c1, w1, s2, c2, w2) {
  const v1x = l1 * c1 * w1;
  const v1y = -l1 * s1 * w1;
  const v2x = v1x + l2 * c2 * w2;
  const v2y = v1y - l2 * s2 * w2;
  const ke = 0.5 * (m1 * (v1x * v1x + v1y * v1y) + m2 * (v2x * v2x + v2y * v2y));
  return ke + g * (m1 * l1 * c1 + m2 * (l1 * c1 + l2 * c2));
}

function duCost(U, uo, nKnots, wDu, kdt) {
  let acc = 0;
  for (let j = 0; j < nKnots - 1; j += 1) {
    const du = (U[uo + j + 1] - U[uo + j]) / kdt;
    acc += wDu * du * du * kdt;
  }
  return acc;
}

/**
 * Cost of residual plan U (offset `uo`) from state s0 (offset `so`). If `fin`
 * is given, the final state [6] is written to it (tests).
 */
export function rolloutCostDouble(p, K, P, s0, so, U, uo, nKnots, fin = null) {
  const M = p[0], m1 = p[1], m2 = p[2], l1 = p[3], l2 = p[4], g = p[5];
  const b = p[6], cd1 = p[7], cd2 = p[8], fmax = p[10];
  const t1 = p[11], t2 = p[12], eT = p[13];
  const wAngle = p[14], wVel = p[15], wVelNear = p[16], nearScale = p[17];
  const wX = p[18], wXd = p[19], xSoft = p[20], wBarrier = p[21];
  const xDead = p[22], wDead = p[23], wEnergy = p[24], wU = p[25], wDu = p[26];
  const wTl = p[27], wTe = p[28];
  const gIn = p[29], gOut = p[30], anchorOn = p[31];
  const knot = p[32] | 0;
  const sub = p[33] | 0 || 1;
  const early = p[34] > 0;
  const kdt = knot * p[9];
  const dt = p[9] * sub;
  const stepsPerKnot = (knot / sub) | 0;
  const total = nKnots * stepsPerKnot;
  const fast = fastTrigOn();
  const ct1 = Math.cos(t1), st1 = Math.sin(t1);
  const ct2 = Math.cos(t2), st2 = Math.sin(t2);
  const m12s = m1 + m2;
  const a = M + m1 + m2;
  const d = m12s * l1 * l1;
  const f = m2 * l2 * l2;
  const gSpan = gOut - gIn;
  const k0 = K[0], k1 = K[1], k2 = K[2], k3 = K[3], k4 = K[4], k5 = K[5];

  let x = s0[so], xd = s0[so + 1];
  let th1 = s0[so + 2], w1 = s0[so + 3];
  let th2 = s0[so + 4], w2 = s0[so + 5];
  let s1 = Math.sin(th1), c1 = Math.cos(th1);
  let s2 = Math.sin(th2), c2 = Math.cos(th2);
  let acc = 0;
  let done = 0;

  for (let j = 0; j < nKnots; j += 1) {
    const r = U[uo + j];
    for (let q = 0; q < stepsPerKnot; q += 1) {
      let fbU = 0;
      if (anchorOn > 0) {
        const dist = (1 - (c1 * ct1 + s1 * st1)) + (1 - (c2 * ct2 + s2 * st2));
        const gate = clip((gOut - dist) / gSpan, 0, 1);
        if (gate > 0) {
          fbU = -gate * (k0 * x + k1 * xd + k2 * wrap(th1 - t1) + k3 * w1 + k4 * wrap(th2 - t2) + k5 * w2);
        }
      }
      let u = clip(fbU + r, -fmax, fmax);
      if (!Number.isFinite(u)) u = 0;

      // physics.js accelerations(): closed-form symmetric 3x3 solve.
      const s12 = s1 * c2 - c1 * s2;
      const c12 = c1 * c2 + s1 * s2;
      const bb = m12s * l1 * c1;
      const cc = m2 * l2 * c2;
      const e = m2 * l1 * l2 * c12;
      const rx = u - b * xd + m12s * l1 * s1 * w1 * w1 + m2 * l2 * s2 * w2 * w2;
      const r1 = -cd1 * w1 - m2 * l1 * l2 * s12 * w2 * w2 + m12s * g * l1 * s1;
      const r2 = -cd2 * w2 + m2 * l1 * l2 * s12 * w1 * w1 + m2 * g * l2 * s2;
      let det = a * (d * f - e * e) - bb * (bb * f - cc * e) + cc * (bb * e - d * cc);
      if (Math.abs(det) < 1e-8) det = 1e-8;
      const a11 = d * f - e * e;
      const a12 = cc * e - bb * f;
      const a13 = bb * e - d * cc;
      const a22 = a * f - cc * cc;
      const a23 = cc * bb - a * e;
      const a33 = a * d - bb * bb;
      const xdd = (a11 * rx + a12 * r1 + a13 * r2) / det;
      const t1dd = (a12 * rx + a22 * r1 + a23 * r2) / det;
      const t2dd = (a13 * rx + a23 * r1 + a33 * r2) / det;
      xd += xdd * dt;
      w1 += t1dd * dt;
      w2 += t2dd * dt;
      x += xd * dt;
      const a1 = w1 * dt, a2 = w2 * dt;
      th1 += a1;
      th2 += a2;
      if (fast) {
        let q2 = a1 * a1;
        let sa = a1 * (1 - q2 * (1 / 6 - q2 * (1 / 120 - q2 / 5040)));
        let ca = 1 - q2 * (0.5 - q2 * (1 / 24 - q2 * (1 / 720 - q2 / 40320)));
        let t = s1 * ca + c1 * sa;
        c1 = c1 * ca - s1 * sa;
        s1 = t;
        q2 = a2 * a2;
        sa = a2 * (1 - q2 * (1 / 6 - q2 * (1 / 120 - q2 / 5040)));
        ca = 1 - q2 * (0.5 - q2 * (1 / 24 - q2 * (1 / 720 - q2 / 40320)));
        t = s2 * ca + c2 * sa;
        c2 = c2 * ca - s2 * sa;
        s2 = t;
      } else {
        s1 = Math.sin(th1); c1 = Math.cos(th1);
        s2 = Math.sin(th2); c2 = Math.cos(th2);
      }

      const ang = (1 - (c1 * ct1 + s1 * st1)) + (1 - (c2 * ct2 + s2 * st2));
      const near = Math.exp(-ang / nearScale);
      const spin = w1 * w1 + w2 * w2;
      const ax = Math.abs(x);
      const over = ax > xSoft ? ax - xSoft : 0;
      const dead = ax > xDead ? 1 : 0;
      const en = poleEnergy(l1, l2, m1, m2, g, s1, c1, w1, s2, c2, w2) - eT;
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
        if (fin) fin.set([x, xd, th1, w1, th2, w2]);
        return acc + wDead * dt * (total - done) + duCost(U, uo, nKnots, wDu, kdt);
      }
    }
  }
  if (fin) fin.set([x, xd, th1, w1, th2, w2]);

  const e0 = x, e1 = xd, e2 = wrap(th1 - t1), e3 = w1, e4 = wrap(th2 - t2), e5 = w2;
  let quad = 0;
  for (let i = 0; i < 6; i += 1) {
    const o = i * 6;
    const row = P[o] * e0 + P[o + 1] * e1 + P[o + 2] * e2 + P[o + 3] * e3 + P[o + 4] * e4 + P[o + 5] * e5;
    const ei = i === 0 ? e0 : i === 1 ? e1 : i === 2 ? e2 : i === 3 ? e3 : i === 4 ? e4 : e5;
    quad += ei * row;
  }
  const ang = (1 - (c1 * ct1 + s1 * st1)) + (1 - (c2 * ct2 + s2 * st2));
  const near = Math.exp(-ang / nearScale);
  const en = poleEnergy(l1, l2, m1, m2, g, s1, c1, w1, s2, c2, w2) - eT;
  acc += near * wTl * quad + (1 - near) * wTe * en * en;
  return acc + duCost(U, uo, nKnots, wDu, kdt);
}
