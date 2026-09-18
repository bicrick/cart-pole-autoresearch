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
