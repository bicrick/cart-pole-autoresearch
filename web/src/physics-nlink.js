/**
 * Cart + n-link pendulum for the page: train/mppi/nlink/plant.py `step`
 * (point masses at the link tips, absolute angles, velocity clamps, mass
 * matrix + 1e-8 jitter, semi-implicit Euler). State keys x, xd, th1, th1d, ...
 * The link count comes from `constants.nLinks`.
 */
import { cholSolve } from "./mppi/rollout-nlink.js";

export const QUAD_CONSTANTS = {
  cartMass: 1.0,
  poleMass1: 0.1,
  poleMass2: 0.1,
  poleMass3: 0.1,
  poleMass4: 0.1,
  poleLength1: 0.5,
  poleLength2: 0.5,
  poleLength3: 0.5,
  poleLength4: 0.5,
  gravity: 9.81,
  cartFriction: 0.08,
  jointDamping1: 0.002,
  jointDamping2: 0.002,
  jointDamping3: 0.002,
  jointDamping4: 0.002,
  forceLimit: 40.0,
  dt: 0.008333333333333333,
  nLinks: 4,
  trackLimit: 2.4,
  trackWalls: false,
};

const MAX_CART_VEL = 30;
const MAX_ANG_VEL = 50;
const MAX_ACC = 1e4;
const JITTER = 1e-8;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function finite(v) {
  return Number.isFinite(v) ? v : 0;
}

/** Per-link constants, cached on the constants object. */
function links(constants) {
  if (constants._links) return constants._links;
  const n = constants.nLinks;
  const m = [], l = [], c = [];
  for (let i = 1; i <= n; i += 1) {
    m.push(constants[`poleMass${i}`]);
    l.push(constants[`poleLength${i}`]);
    c.push(constants[`jointDamping${i}`]);
  }
  const S = m.map((_, i) => m.slice(i).reduce((a, b) => a + b, 0));
  const out = { n, m, l, c, S };
  Object.defineProperty(constants, "_links", { value: out, enumerable: false });
  return out;
}

/** [xdd, th1dd, ..., thndd]; extraQ = { qx, q1..qn } generalized grab forces. */
export function accelerations(state, force, extraQ, constants = QUAD_CONSTANTS) {
  const { n, m, l, c, S } = links(constants);
  const D = n + 1;
  const th = [], w = [], sn = [], cs = [];
  for (let i = 0; i < n; i += 1) {
    th.push(state[`th${i + 1}`]);
    w.push(state[`th${i + 1}d`]);
    sn.push(Math.sin(th[i]));
    cs.push(Math.cos(th[i]));
  }
  const g = constants.gravity;
  const A = new Float64Array(D * D);
  const x = new Float64Array(D);
  A[0] = constants.cartMass + m.reduce((a, b) => a + b, 0) + JITTER;
  x[0] = force - constants.cartFriction * state.xd + (extraQ?.qx ?? 0);
  for (let i = 0; i < n; i += 1) {
    const v = S[i] * l[i] * cs[i];
    A[1 + i] = v;
    A[(1 + i) * D] = v;
    x[0] += S[i] * l[i] * sn[i] * w[i] * w[i];
    let ri = -c[i] * w[i] + S[i] * g * l[i] * sn[i] + (extraQ?.[`q${i + 1}`] ?? 0);
    for (let k = 0; k < n; k += 1) {
      const Sm = S[Math.max(i, k)];
      A[(1 + i) * D + 1 + k] = Sm * l[i] * l[k] * Math.cos(th[i] - th[k]);
      if (k !== i) ri -= Sm * l[i] * l[k] * Math.sin(th[i] - th[k]) * w[k] * w[k];
    }
    A[(1 + i) * D + 1 + i] += JITTER;
    x[1 + i] = ri;
  }
  cholSolve(A, x, D);
  return x;
}

export function step(state, force, extraQ, constants = QUAD_CONSTANTS) {
  const n = constants.nLinks;
  const fmax = constants.forceLimit ?? 40;
  const s = { ...state, xd: clamp(state.xd, -MAX_CART_VEL, MAX_CART_VEL) };
  for (let i = 1; i <= n; i += 1) s[`th${i}d`] = clamp(state[`th${i}d`], -MAX_ANG_VEL, MAX_ANG_VEL);
  const acc = accelerations(s, clamp(finite(force), -fmax, fmax), extraQ, constants);
  const dt = constants.dt;
  const out = {};
  let xd = clamp(s.xd + clamp(finite(acc[0]), -MAX_ACC, MAX_ACC) * dt, -MAX_CART_VEL, MAX_CART_VEL);
  let x = s.x + xd * dt;
  if (constants.trackWalls) {
    const track = constants.trackLimit ?? 2.4;
    if (x > track || x < -track) {
      x = clamp(x, -track, track);
      xd = 0;
    }
  }
  out.x = finite(x);
  out.xd = finite(xd);
  for (let i = 1; i <= n; i += 1) {
    const w = clamp(s[`th${i}d`] + clamp(finite(acc[i]), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
    out[`th${i}`] = finite(s[`th${i}`] + w * dt);
    out[`th${i}d`] = finite(w);
  }
  return out;
}

export function observe(state, constants = QUAD_CONSTANTS) {
  const n = constants.nLinks;
  const obs = [state.x, state.xd];
  for (let i = 1; i <= n; i += 1) obs.push(Math.sin(state[`th${i}`]), Math.cos(state[`th${i}`]));
  for (let i = 1; i <= n; i += 1) obs.push(state[`th${i}d`]);
  return obs;
}

/** Joint bodies are "j1".."jn" (tip of each link), cart-most first. */
export function tipPositions(state, constants = QUAD_CONSTANTS) {
  const { n, l } = links(constants);
  const poles = [];
  let px = state.x;
  let py = 0;
  for (let i = 0; i < n; i += 1) {
    const th = state[`th${i + 1}`];
    px += l[i] * Math.sin(th);
    py += l[i] * Math.cos(th);
    poles.push({ x: px, y: py, body: `j${i + 1}` });
  }
  return { cart: { x: state.x, y: 0 }, poles, tip: poles[n - 1] };
}

export function grabForces(state, target, body, stiffness, damping, constants = QUAD_CONSTANTS) {
  const { n, l } = links(constants);
  const k = body === "cart" ? 0 : Number(String(body).slice(1)) || 0;
  let px = state.x;
  let py = 0;
  let vx = state.xd;
  let vy = 0;
  for (let i = 0; i < k; i += 1) {
    const th = state[`th${i + 1}`];
    const w = state[`th${i + 1}d`];
    px += l[i] * Math.sin(th);
    py += l[i] * Math.cos(th);
    vx += l[i] * Math.cos(th) * w;
    vy -= l[i] * Math.sin(th) * w;
  }
  const fx = stiffness * (target.x - px) - damping * vx;
  const fy = stiffness * (target.y - py) - damping * vy;
  const q = { qx: fx };
  for (let i = 0; i < n; i += 1) {
    const th = state[`th${i + 1}`];
    q[`q${i + 1}`] = i < k ? l[i] * Math.cos(th) * fx - l[i] * Math.sin(th) * fy : 0;
  }
  return q;
}
