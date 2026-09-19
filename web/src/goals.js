/** Discrete plant equilibria (theta = 0 upright). Xin / IFAC 2008. */
export const GOAL_IDS = ["UU", "UD", "DU", "DD"];

export const GOAL_ANGLES = {
  UU: [0, 0],
  UD: [0, Math.PI],
  DU: [Math.PI, 0],
  DD: [Math.PI, Math.PI],
};

export function goalEncoding(goalId) {
  const onehot = GOAL_IDS.map((id) => (id === goalId ? 1 : 0));
  const [th1, th2] = GOAL_ANGLES[goalId] ?? GOAL_ANGLES.UU;
  return [...onehot, Math.sin(th1), Math.cos(th1), Math.sin(th2), Math.cos(th2)];
}

export function conditionedObs(stateObs, goalId) {
  return stateObs.concat(goalEncoding(goalId));
}

export function atGoal(state, goalId, cosThresh = 0.95) {
  const [t1, t2] = GOAL_ANGLES[goalId] ?? GOAL_ANGLES.UU;
  const a1 = Math.cos(state.th1) * Math.cos(t1) + Math.sin(state.th1) * Math.sin(t1);
  const a2 = Math.cos(state.th2) * Math.cos(t2) + Math.sin(state.th2) * Math.sin(t2);
  return a1 > cosThresh && a2 > cosThresh;
}

export function ghostTips(cartX, goalId, constants) {
  const [th1, th2] = GOAL_ANGLES[goalId] ?? GOAL_ANGLES.UU;
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const p1x = cartX + l1 * Math.sin(th1);
  const p1y = l1 * Math.cos(th1);
  const lower = { x: p1x, y: p1y, body: "lower" };
  const upper = { x: p1x + l2 * Math.sin(th2), y: p1y + l2 * Math.cos(th2), body: "upper" };
  return {
    cart: { x: cartX, y: 0 },
    lower,
    upper,
    poles: [lower, upper],
  };
}
