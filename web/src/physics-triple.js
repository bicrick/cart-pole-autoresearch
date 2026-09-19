export const TRIPLE_CONSTANTS = {
  cartMass: 1.0,
  poleMass1: 0.1,
  poleMass2: 0.1,
  poleMass3: 0.1,
  poleLength1: 0.5,
  poleLength2: 0.5,
  poleLength3: 0.5,
  gravity: 9.81,
  cartFriction: 0.08,
  jointDamping1: 0.002,
  jointDamping2: 0.002,
  jointDamping3: 0.002,
  forceLimit: 20.0,
  dt: 0.008333333333333333,
  hidden: 128,
  obsDim: 25,
  nLinks: 3,
  obsLow: [
    -4.0, -6.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -12.0, -12.0, -12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0,
  ],
  obsHigh: [
    4.0, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
  ],
  trackLimit: 2.4,
  trackWalls: false,
  wallRestitution: 0.0,
};

const MAX_CART_VEL = 30;
const MAX_ANG_VEL = 50;
const MAX_ACC = 1e4;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function finite(v) {
  return Number.isFinite(v) ? v : 0;
}

function solve4(mass, rhs) {
  const a = mass.map((row, i) => row.slice().concat(rhs[i]));
  for (let col = 0; col < 4; col += 1) {
    let pivot = col;
    let best = Math.abs(a[col][col]);
    for (let r = col + 1; r < 4; r += 1) {
      const mag = Math.abs(a[r][col]);
      if (mag > best) {
        best = mag;
        pivot = r;
      }
    }
    if (best < 1e-12) a[col][col] += 1e-8;
    if (pivot !== col) {
      const tmp = a[col];
      a[col] = a[pivot];
      a[pivot] = tmp;
    }
    const diag = a[col][col] || 1e-8;
    for (let r = col + 1; r < 4; r += 1) {
      const f = a[r][col] / diag;
      for (let c = col; c < 5; c += 1) a[r][c] -= f * a[col][c];
    }
  }
  const x = new Array(4);
  for (let i = 3; i >= 0; i -= 1) {
    let s = a[i][4];
    for (let j = i + 1; j < 4; j += 1) s -= a[i][j] * x[j];
    x[i] = s / (a[i][i] || 1e-8);
  }
  return x;
}

export function accelerations(state, force, extraQ, constants = TRIPLE_CONSTANTS) {
  const { xd, th1, th1d, th2, th2d, th3, th3d } = state;
  const M = constants.cartMass;
  const m1 = constants.poleMass1;
  const m2 = constants.poleMass2;
  const m3 = constants.poleMass3;
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const l3 = constants.poleLength3;
  const g = constants.gravity;
  const b = constants.cartFriction;
  const c1 = constants.jointDamping1;
  const c2 = constants.jointDamping2;
  const c3 = constants.jointDamping3;
  const qx = extraQ?.qx ?? 0;
  const q1 = extraQ?.q1 ?? 0;
  const q2 = extraQ?.q2 ?? 0;
  const q3 = extraQ?.q3 ?? 0;

  const s1 = Math.sin(th1);
  const cth1 = Math.cos(th1);
  const s2 = Math.sin(th2);
  const cth2 = Math.cos(th2);
  const s3 = Math.sin(th3);
  const cth3 = Math.cos(th3);
  const s12 = Math.sin(th1 - th2);
  const c12 = Math.cos(th1 - th2);
  const s13 = Math.sin(th1 - th3);
  const c13 = Math.cos(th1 - th3);
  const s23 = Math.sin(th2 - th3);
  const c23 = Math.cos(th2 - th3);

  const m11 = M + m1 + m2 + m3;
  const m12 = (m1 + m2 + m3) * l1 * cth1;
  const m13 = (m2 + m3) * l2 * cth2;
  const m14 = m3 * l3 * cth3;
  const m22 = (m1 + m2 + m3) * l1 * l1;
  const m23 = (m2 + m3) * l1 * l2 * c12;
  const m24 = m3 * l1 * l3 * c13;
  const m33 = (m2 + m3) * l2 * l2;
  const m34 = m3 * l2 * l3 * c23;
  const m44 = m3 * l3 * l3;
  const jitter = 1e-8;
  const mass = [
    [m11 + jitter, m12, m13, m14],
    [m12, m22 + jitter, m23, m24],
    [m13, m23, m33 + jitter, m34],
    [m14, m24, m34, m44 + jitter],
  ];
  const rhsX = force - b * xd + (m1 + m2 + m3) * l1 * s1 * th1d * th1d + (m2 + m3) * l2 * s2 * th2d * th2d + m3 * l3 * s3 * th3d * th3d + qx;
  const rhs1 = q1 - c1 * th1d - (m2 + m3) * l1 * l2 * s12 * th2d * th2d - m3 * l1 * l3 * s13 * th3d * th3d + (m1 + m2 + m3) * g * l1 * s1;
  const rhs2 = q2 - c2 * th2d + (m2 + m3) * l1 * l2 * s12 * th1d * th1d - m3 * l2 * l3 * s23 * th3d * th3d + (m2 + m3) * g * l2 * s2;
  const rhs3 = q3 - c3 * th3d + m3 * l1 * l3 * s13 * th1d * th1d + m3 * l2 * l3 * s23 * th2d * th2d + m3 * g * l3 * s3;
  const [xdd, t1dd, t2dd, t3dd] = solve4(mass, [rhsX, rhs1, rhs2, rhs3]);
  return { xdd, t1dd, t2dd, t3dd };
}

export function step(state, force, extraQ, constants = TRIPLE_CONSTANTS) {
  const fmax = constants.forceLimit ?? 20;
  const clipped = clamp(finite(force), -fmax, fmax);
  const acc = accelerations(
    {
      ...state,
      xd: clamp(state.xd, -MAX_CART_VEL, MAX_CART_VEL),
      th1d: clamp(state.th1d, -MAX_ANG_VEL, MAX_ANG_VEL),
      th2d: clamp(state.th2d, -MAX_ANG_VEL, MAX_ANG_VEL),
      th3d: clamp(state.th3d, -MAX_ANG_VEL, MAX_ANG_VEL),
    },
    clipped,
    extraQ,
    constants,
  );
  const dt = constants.dt;
  const xd = clamp(state.xd + clamp(finite(acc.xdd), -MAX_ACC, MAX_ACC) * dt, -MAX_CART_VEL, MAX_CART_VEL);
  const th1d = clamp(state.th1d + clamp(finite(acc.t1dd), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
  const th2d = clamp(state.th2d + clamp(finite(acc.t2dd), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
  const th3d = clamp(state.th3d + clamp(finite(acc.t3dd), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);
  return {
    x: state.x + xd * dt,
    xd,
    th1: state.th1 + th1d * dt,
    th1d,
    th2: state.th2 + th2d * dt,
    th2d,
    th3: state.th3 + th3d * dt,
    th3d,
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
    Math.sin(state.th3),
    Math.cos(state.th3),
    state.th1d,
    state.th2d,
    state.th3d,
  ];
}

export function tipPositions(state, constants = TRIPLE_CONSTANTS) {
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const l3 = constants.poleLength3;
  const p1x = state.x + l1 * Math.sin(state.th1);
  const p1y = l1 * Math.cos(state.th1);
  const p2x = p1x + l2 * Math.sin(state.th2);
  const p2y = p1y + l2 * Math.cos(state.th2);
  const lower = { x: p1x, y: p1y, body: "lower" };
  const mid = { x: p2x, y: p2y, body: "mid" };
  const tip = { x: p2x + l3 * Math.sin(state.th3), y: p2y + l3 * Math.cos(state.th3), body: "tip" };
  return {
    cart: { x: state.x, y: 0 },
    lower,
    mid,
    tip,
    poles: [lower, mid, tip],
  };
}

export function grabForces(state, target, body, stiffness, damping, constants = TRIPLE_CONSTANTS) {
  const tips = tipPositions(state, constants);
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const l3 = constants.poleLength3;
  const c1 = Math.cos(state.th1);
  const s1 = Math.sin(state.th1);
  const c2 = Math.cos(state.th2);
  const s2 = Math.sin(state.th2);
  const c3 = Math.cos(state.th3);
  const s3 = Math.sin(state.th3);
  let px = state.x;
  let py = 0;
  let vx = state.xd;
  let vy = 0;
  if (body === "lower") {
    px = tips.lower.x;
    py = tips.lower.y;
    vx = state.xd + l1 * c1 * state.th1d;
    vy = -l1 * s1 * state.th1d;
  } else if (body === "mid") {
    px = tips.mid.x;
    py = tips.mid.y;
    vx = state.xd + l1 * c1 * state.th1d + l2 * c2 * state.th2d;
    vy = -l1 * s1 * state.th1d - l2 * s2 * state.th2d;
  } else if (body === "tip") {
    px = tips.tip.x;
    py = tips.tip.y;
    vx = state.xd + l1 * c1 * state.th1d + l2 * c2 * state.th2d + l3 * c3 * state.th3d;
    vy = -l1 * s1 * state.th1d - l2 * s2 * state.th2d - l3 * s3 * state.th3d;
  }
  const fx = stiffness * (target.x - px) - damping * vx;
  const fy = stiffness * (target.y - py) - damping * vy;
  const beyond1 = body === "lower" || body === "mid" || body === "tip";
  const beyond2 = body === "mid" || body === "tip";
  return {
    qx: fx,
    q1: beyond1 ? l1 * c1 * fx - l1 * s1 * fy : 0,
    q2: beyond2 ? l2 * c2 * fx - l2 * s2 * fy : 0,
    q3: body === "tip" ? l3 * c3 * fx - l3 * s3 * fy : 0,
  };
}
