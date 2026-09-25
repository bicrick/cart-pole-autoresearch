/**
 * WGSL for the WebGPU sample evaluator: one invocation per MPPI sample.
 * Same math as rollout.js / fast_rollout.py, in f32. `mode` 0 generates the
 * sample (AR(1) noise around U, sample 0 unperturbed); mode 1 scores the plans
 * already in `samples` (parity tests).
 *
 * prm layout: p[0..38] (rollout.js FIELDS), K at 39, P (row-major) at 47, s0 at 111.
 */
export const PRM_K = 39;
export const PRM_P = 47;
export const PRM_S0 = 111;
export const PRM_LEN = 119;

export const WGSL = /* wgsl */ `
struct Cfg {
  n: u32, T: u32, mode: u32, seed: u32,
  sigma: f32, sigmaWide: f32, wideFrom: u32, beta: f32,
  lim: f32, pad0: f32, pad1: f32, pad2: f32,
};

@group(0) @binding(0) var<uniform> cfg: Cfg;
@group(0) @binding(1) var<storage, read> prm: array<f32>;
@group(0) @binding(2) var<storage, read> U: array<f32>;
@group(0) @binding(3) var<storage, read_write> samples: array<f32>;
@group(0) @binding(4) var<storage, read_write> costs: array<f32>;

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
const MAX_CART_VEL: f32 = 30.0;
const MAX_ANG_VEL: f32 = 50.0;
const MAX_ACC: f32 = 10000.0;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn uni(h: u32) -> f32 {
  return (f32(h >> 8u) + 0.5) / 16777216.0;
}

fn gauss(i: u32, t: u32) -> f32 {
  let h = pcg(cfg.seed ^ pcg(i ^ pcg(t + 0x9e3779b9u)));
  let u1 = uni(h);
  let u2 = uni(pcg(h));
  return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

fn wrapA(a: f32) -> f32 {
  return a - TWO_PI * floor((a + PI) / TWO_PI);
}

fn fin(v: f32) -> f32 {
  // NaN fails v == v; inf fails the magnitude test.
  if (v == v && abs(v) < 3.0e38) { return v; }
  return 0.0;
}

fn poleEnergy(s1: f32, c1: f32, w1: f32, s2: f32, c2: f32, w2: f32, s3: f32, c3: f32, w3: f32) -> f32 {
  let m1 = prm[1]; let m2 = prm[2]; let m3 = prm[3];
  let l1 = prm[4]; let l2 = prm[5]; let l3 = prm[6]; let g = prm[7];
  let v1x = l1 * c1 * w1;
  let v1y = -l1 * s1 * w1;
  let v2x = v1x + l2 * c2 * w2;
  let v2y = v1y - l2 * s2 * w2;
  let v3x = v2x + l3 * c3 * w3;
  let v3y = v2y - l3 * s3 * w3;
  let ke = 0.5 * (m1 * (v1x * v1x + v1y * v1y) + m2 * (v2x * v2x + v2y * v2y) + m3 * (v3x * v3x + v3y * v3y));
  let y1 = l1 * c1;
  let y2 = y1 + l2 * c2;
  let y3 = y2 + l3 * c3;
  return ke + g * (m1 * y1 + m2 * y2 + m3 * y3);
}

fn duCost(off: u32, T: u32, wDu: f32, kdt: f32) -> f32 {
  var acc: f32 = 0.0;
  for (var j: u32 = 0u; j + 1u < T; j = j + 1u) {
    let du = (samples[off + j + 1u] - samples[off + j]) / kdt;
    acc = acc + wDu * du * du * kdt;
  }
  return acc;
}

fn rollout(off: u32, T: u32) -> f32 {
  let M = prm[0]; let m1 = prm[1]; let m2 = prm[2]; let m3 = prm[3];
  let l1 = prm[4]; let l2 = prm[5]; let l3 = prm[6]; let g = prm[7];
  let b = prm[8]; let cd1 = prm[9]; let cd2 = prm[10]; let cd3 = prm[11];
  let fmax = prm[13];
  let t1 = prm[14]; let t2 = prm[15]; let t3 = prm[16]; let eT = prm[17];
  let wAngle = prm[18]; let wVel = prm[19]; let wVelNear = prm[20]; let nearScale = prm[21];
  let wX = prm[22]; let wXd = prm[23]; let xSoft = prm[24]; let wBarrier = prm[25];
  let xDead = prm[26]; let wDead = prm[27]; let wEnergy = prm[28]; let wU = prm[29]; let wDu = prm[30];
  let wTl = prm[31]; let wTe = prm[32];
  let gIn = prm[33]; let gOut = prm[34]; let anchorOn = prm[35];
  let knot = u32(prm[36]);
  let sub = max(1u, u32(prm[37]));
  let early = prm[38] > 0.0;
  let kdt = f32(knot) * prm[12];
  let dt = prm[12] * f32(sub);
  let stepsPerKnot = knot / sub;
  let total = T * stepsPerKnot;
  let ct1 = cos(t1); let st1 = sin(t1);
  let ct2 = cos(t2); let st2 = sin(t2);
  let ct3 = cos(t3); let st3 = sin(t3);
  let m123 = m1 + m2 + m3;
  let m23 = m2 + m3;
  let a00 = M + m123 + 1e-8;
  let a11 = m123 * l1 * l1 + 1e-8;
  let a22 = m23 * l2 * l2 + 1e-8;
  let a33 = m3 * l3 * l3 + 1e-8;

  var x = prm[${PRM_S0}]; var xd = prm[${PRM_S0 + 1}];
  var th1 = prm[${PRM_S0 + 2}]; var w1 = prm[${PRM_S0 + 3}];
  var th2 = prm[${PRM_S0 + 4}]; var w2 = prm[${PRM_S0 + 5}];
  var th3 = prm[${PRM_S0 + 6}]; var w3 = prm[${PRM_S0 + 7}];
  var s1 = sin(th1); var c1 = cos(th1);
  var s2 = sin(th2); var c2 = cos(th2);
  var s3 = sin(th3); var c3 = cos(th3);
  var acc: f32 = 0.0;
  var done: u32 = 0u;

  for (var j: u32 = 0u; j < T; j = j + 1u) {
    let r = samples[off + j];
    for (var q: u32 = 0u; q < stepsPerKnot; q = q + 1u) {
      var fbU: f32 = 0.0;
      if (anchorOn > 0.0) {
        let d = (1.0 - (c1 * ct1 + s1 * st1)) + (1.0 - (c2 * ct2 + s2 * st2)) + (1.0 - (c3 * ct3 + s3 * st3));
        let gate = clamp((gOut - d) / (gOut - gIn), 0.0, 1.0);
        if (gate > 0.0) {
          let e1 = wrapA(th1 - t1);
          let e2 = wrapA(th2 - t2);
          let e3 = wrapA(th3 - t3);
          fbU = -gate * (prm[${PRM_K}] * x + prm[${PRM_K + 1}] * xd + prm[${PRM_K + 2}] * e1 + prm[${PRM_K + 3}] * w1
            + prm[${PRM_K + 4}] * e2 + prm[${PRM_K + 5}] * w2 + prm[${PRM_K + 6}] * e3 + prm[${PRM_K + 7}] * w3);
        }
      }
      let u = clamp(fin(fbU + r), -fmax, fmax);

      xd = clamp(xd, -MAX_CART_VEL, MAX_CART_VEL);
      w1 = clamp(w1, -MAX_ANG_VEL, MAX_ANG_VEL);
      w2 = clamp(w2, -MAX_ANG_VEL, MAX_ANG_VEL);
      w3 = clamp(w3, -MAX_ANG_VEL, MAX_ANG_VEL);
      let s12 = s1 * c2 - c1 * s2; let c12 = c1 * c2 + s1 * s2;
      let s13 = s1 * c3 - c1 * s3; let c13 = c1 * c3 + s1 * s3;
      let s23 = s2 * c3 - c2 * s3; let c23 = c2 * c3 + s2 * s3;
      let a01 = m123 * l1 * c1;
      let a02 = m23 * l2 * c2;
      let a03 = m3 * l3 * c3;
      let a12 = m23 * l1 * l2 * c12;
      let a13 = m3 * l1 * l3 * c13;
      let a23 = m3 * l2 * l3 * c23;
      let q1s = w1 * w1; let q2s = w2 * w2; let q3s = w3 * w3;
      let r0 = u - b * xd + m123 * l1 * s1 * q1s + m23 * l2 * s2 * q2s + m3 * l3 * s3 * q3s;
      let r1 = -cd1 * w1 - m23 * l1 * l2 * s12 * q2s - m3 * l1 * l3 * s13 * q3s + m123 * g * l1 * s1;
      let r2 = -cd2 * w2 + m23 * l1 * l2 * s12 * q1s - m3 * l2 * l3 * s23 * q3s + m23 * g * l2 * s2;
      let r3 = -cd3 * w3 + m3 * l1 * l3 * s13 * q1s + m3 * l2 * l3 * s23 * q2s + m3 * g * l3 * s3;
      let d0 = a00;
      let l10 = a01 / d0; let l20 = a02 / d0; let l30 = a03 / d0;
      let d1 = a11 - l10 * l10 * d0;
      let l21 = (a12 - l20 * l10 * d0) / d1;
      let l31 = (a13 - l30 * l10 * d0) / d1;
      let d2 = a22 - l20 * l20 * d0 - l21 * l21 * d1;
      let l32 = (a23 - l30 * l20 * d0 - l31 * l21 * d1) / d2;
      let d3 = a33 - l30 * l30 * d0 - l31 * l31 * d1 - l32 * l32 * d2;
      let y0 = r0;
      let y1 = r1 - l10 * y0;
      let y2 = r2 - l20 * y0 - l21 * y1;
      let y3 = r3 - l30 * y0 - l31 * y1 - l32 * y2;
      let x3 = y3 / d3;
      let x2 = y2 / d2 - l32 * x3;
      let x1 = y1 / d1 - l21 * x2 - l31 * x3;
      let x0 = y0 / d0 - l10 * x1 - l20 * x2 - l30 * x3;
      xd = clamp(xd + clamp(fin(x0), -MAX_ACC, MAX_ACC) * dt, -MAX_CART_VEL, MAX_CART_VEL);
      w1 = clamp(w1 + clamp(fin(x1), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
      w2 = clamp(w2 + clamp(fin(x2), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
      w3 = clamp(w3 + clamp(fin(x3), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
      x = x + xd * dt;
      th1 = th1 + w1 * dt;
      th2 = th2 + w2 * dt;
      th3 = th3 + w3 * dt;
      s1 = sin(th1); c1 = cos(th1);
      s2 = sin(th2); c2 = cos(th2);
      s3 = sin(th3); c3 = cos(th3);

      let ang = (1.0 - (c1 * ct1 + s1 * st1)) + (1.0 - (c2 * ct2 + s2 * st2)) + (1.0 - (c3 * ct3 + s3 * st3));
      let near = exp(-ang / nearScale);
      let spin = w1 * w1 + w2 * w2 + w3 * w3;
      let ax = abs(x);
      let over = max(ax - xSoft, 0.0);
      let dead = select(0.0, 1.0, ax > xDead);
      let en = poleEnergy(s1, c1, w1, s2, c2, w2, s3, c3, w3) - eT;
      acc = acc + (wAngle * ang + (wVel + wVelNear * near) * spin + wX * x * x + wXd * xd * xd
        + wBarrier * over * over + wDead * dead + wEnergy * (1.0 - near) * en * en + wU * u * u) * dt;
      done = done + 1u;
      if (early && dead > 0.0) {
        return acc + wDead * dt * f32(total - done) + duCost(off, T, wDu, kdt);
      }
    }
  }

  let e = array<f32, 8>(x, xd, wrapA(th1 - t1), w1, wrapA(th2 - t2), w2, wrapA(th3 - t3), w3);
  var quad: f32 = 0.0;
  for (var i: u32 = 0u; i < 8u; i = i + 1u) {
    var row: f32 = 0.0;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      row = row + prm[${PRM_P}u + i * 8u + k] * e[k];
    }
    quad = quad + e[i] * row;
  }
  let angT = (1.0 - (c1 * ct1 + s1 * st1)) + (1.0 - (c2 * ct2 + s2 * st2)) + (1.0 - (c3 * ct3 + s3 * st3));
  let nearT = exp(-angT / nearScale);
  let enT = poleEnergy(s1, c1, w1, s2, c2, w2, s3, c3, w3) - eT;
  acc = acc + nearT * wTl * quad + (1.0 - nearT) * wTe * enT * enT;
  return acc + duCost(off, T, wDu, kdt);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= cfg.n) { return; }
  let T = cfg.T;
  let off = i * T;
  if (cfg.mode == 0u) {
    if (i == 0u) {
      for (var t: u32 = 0u; t < T; t = t + 1u) { samples[off + t] = clamp(U[t], -cfg.lim, cfg.lim); }
    } else {
      let sig = select(cfg.sigma, cfg.sigmaWide, i >= cfg.wideFrom);
      let sc = sqrt(1.0 - cfg.beta * cfg.beta);
      var e = gauss(i, 0u);
      for (var t: u32 = 0u; t < T; t = t + 1u) {
        if (t > 0u) { e = cfg.beta * e + sc * gauss(i, t); }
        samples[off + t] = clamp(U[t] + sig * e, -cfg.lim, cfg.lim);
      }
    }
  }
  costs[i] = rollout(off, T);
}
`;
