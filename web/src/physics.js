export const DEFAULT_CONSTANTS = {
  cartMass: 1.0,
  poleMass1: 0.1,
  poleMass2: 0.1,
  poleLength1: 0.5,
  poleLength2: 0.5,
  gravity: 9.81,
  cartFriction: 0.08,
  jointDamping1: 0.002,
  jointDamping2: 0.002,
  forceLimit: 20.0,
  dt: 0.008333333333333333,
  hidden: 128,
  obsDim: 16,
  obsLow: [-4.0, -6.0, -1.0, -1.0, -1.0, -1.0, -12.0, -12.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
  obsHigh: [4.0, 6.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
};

function solveMass(a, b, c, d, e, f, rx, r1, r2) {
  let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - d * c);
  if (Math.abs(det) < 1e-8) det = 1e-8;
  const a11 = d * f - e * e;
  const a12 = c * e - b * f;
  const a13 = b * e - d * c;
  const a22 = a * f - c * c;
  const a23 = c * b - a * e;
  const a33 = a * d - b * b;
  return {
    xdd: (a11 * rx + a12 * r1 + a13 * r2) / det,
    t1dd: (a12 * rx + a22 * r1 + a23 * r2) / det,
    t2dd: (a13 * rx + a23 * r1 + a33 * r2) / det,
  };
}

export function createState(x = 0, xd = 0, th1 = 0, th1d = 0, th2 = 0, th2d = 0) {
  return { x, xd, th1, th1d, th2, th2d };
}

export function cloneState(state) {
  return { ...state };
}

export function accelerations(state, force, extraQ, constants = DEFAULT_CONSTANTS) {
  const { x, xd, th1, th1d, th2, th2d } = state;
  const M = constants.cartMass;
  const m1 = constants.poleMass1;
  const m2 = constants.poleMass2;
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const g = constants.gravity;
  const b = constants.cartFriction;
  const c1 = constants.jointDamping1;
  const c2 = constants.jointDamping2;
  const qx = extraQ?.qx ?? 0;
  const q1 = extraQ?.q1 ?? 0;
  const q2 = extraQ?.q2 ?? 0;

  const s1 = Math.sin(th1);
  const cth1 = Math.cos(th1);
  const s2 = Math.sin(th2);
  const cth2 = Math.cos(th2);
  const s12 = Math.sin(th1 - th2);
  const c12 = Math.cos(th1 - th2);

  const m11 = M + m1 + m2;
  const m12 = (m1 + m2) * l1 * cth1;
  const m13 = m2 * l2 * cth2;
  const m22 = (m1 + m2) * l1 * l1;
  const m23 = m2 * l1 * l2 * c12;
  const m33 = m2 * l2 * l2;

  const rhsX = force - b * xd + (m1 + m2) * l1 * s1 * th1d * th1d + m2 * l2 * s2 * th2d * th2d + qx;
  const rhs1 = q1 - c1 * th1d - m2 * l1 * l2 * s12 * th2d * th2d + (m1 + m2) * g * l1 * s1;
  const rhs2 = q2 - c2 * th2d + m2 * l1 * l2 * s12 * th1d * th1d + m2 * g * l2 * s2;
  return solveMass(m11, m12, m13, m22, m23, m33, rhsX, rhs1, rhs2);
}

export function step(state, force, extraQ, constants = DEFAULT_CONSTANTS) {
  const fmax = constants.forceLimit;
  const clipped = Math.max(-fmax, Math.min(fmax, force));
  const acc = accelerations(state, clipped, extraQ, constants);
  const dt = constants.dt;
  const xd = state.xd + acc.xdd * dt;
  const th1d = state.th1d + acc.t1dd * dt;
  const th2d = state.th2d + acc.t2dd * dt;
  return {
    x: state.x + xd * dt,
    xd,
    th1: state.th1 + th1d * dt,
    th1d,
    th2: state.th2 + th2d * dt,
    th2d,
  };
}

export function observe(state) {
  return [
    state.x,
    state.xd,
    Math.sin(state.th1),
    Math.cos(state.th1),
    Math.sin(state.th2),
    Math.cos(state.th2),
    state.th1d,
    state.th2d,
  ];
}

export function normalizeObs(obs, constants = DEFAULT_CONSTANTS) {
  const out = new Array(obs.length);
  for (let i = 0; i < obs.length; i += 1) {
    const low = constants.obsLow[i];
    const high = constants.obsHigh[i];
    const mid = 0.5 * (low + high);
    const scale = Math.max(1e-6, 0.5 * (high - low));
    const n = (obs[i] - mid) / scale;
    out[i] = Math.max(-3, Math.min(3, n));
  }
  return out;
}

export function tipPositions(state, constants = DEFAULT_CONSTANTS) {
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const p1x = state.x + l1 * Math.sin(state.th1);
  const p1y = l1 * Math.cos(state.th1);
  return {
    cart: { x: state.x, y: 0 },
    lower: { x: p1x, y: p1y },
    upper: { x: p1x + l2 * Math.sin(state.th2), y: p1y + l2 * Math.cos(state.th2) },
  };
}

export function grabForces(state, target, body, stiffness, damping, constants = DEFAULT_CONSTANTS) {
  const tips = tipPositions(state, constants);
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  let px = state.x;
  let py = 0;
  let vx = state.xd;
  let vy = 0;
  if (body === "lower") {
    px = tips.lower.x;
    py = tips.lower.y;
    vx = state.xd + l1 * Math.cos(state.th1) * state.th1d;
    vy = -l1 * Math.sin(state.th1) * state.th1d;
  } else if (body === "upper") {
    px = tips.upper.x;
    py = tips.upper.y;
    vx = state.xd + l1 * Math.cos(state.th1) * state.th1d + l2 * Math.cos(state.th2) * state.th2d;
    vy = -l1 * Math.sin(state.th1) * state.th1d - l2 * Math.sin(state.th2) * state.th2d;
  }
  const fx = stiffness * (target.x - px) - damping * vx;
  const fy = stiffness * (target.y - py) - damping * vy;
  const qx = fx;
  let q1 = 0;
  let q2 = 0;
  if (body === "lower" || body === "upper") {
    q1 = l1 * Math.cos(state.th1) * fx - l1 * Math.sin(state.th1) * fy;
  }
  if (body === "upper") {
    q2 = l2 * Math.cos(state.th2) * fx - l2 * Math.sin(state.th2) * fy;
  }
  return { qx, q1, q2 };
}
