/** Eight equilibria. U = 0 upright, D = π hanging. MSB is the cart-most link. */
export const GOAL_IDS = ["DDD", "DDU", "DUD", "DUU", "UDD", "UDU", "UUD", "UUU"];

export const GOAL_ANGLES = {
  DDD: [Math.PI, Math.PI, Math.PI],
  DDU: [Math.PI, Math.PI, 0],
  DUD: [Math.PI, 0, Math.PI],
  DUU: [Math.PI, 0, 0],
  UDD: [0, Math.PI, Math.PI],
  UDU: [0, Math.PI, 0],
  UUD: [0, 0, Math.PI],
  UUU: [0, 0, 0],
};

export function goalEncoding(goalId) {
  const onehot = GOAL_IDS.map((id) => (id === goalId ? 1 : 0));
  const [th1, th2, th3] = GOAL_ANGLES[goalId] ?? GOAL_ANGLES.UUU;
  return [
    ...onehot,
    Math.sin(th1),
    Math.cos(th1),
    Math.sin(th2),
    Math.cos(th2),
    Math.sin(th3),
    Math.cos(th3),
  ];
}

export function conditionedObs(stateObs, goalId) {
  return stateObs.concat(goalEncoding(goalId));
}

export function atGoal(state, goalId, cosThresh = 0.95) {
  const [t1, t2, t3] = GOAL_ANGLES[goalId] ?? GOAL_ANGLES.UUU;
  const a1 = Math.cos(state.th1) * Math.cos(t1) + Math.sin(state.th1) * Math.sin(t1);
  const a2 = Math.cos(state.th2) * Math.cos(t2) + Math.sin(state.th2) * Math.sin(t2);
  const a3 = Math.cos(state.th3) * Math.cos(t3) + Math.sin(state.th3) * Math.sin(t3);
  return a1 > cosThresh && a2 > cosThresh && a3 > cosThresh;
}

export function ghostTips(cartX, goalId, constants) {
  const [th1, th2, th3] = GOAL_ANGLES[goalId] ?? GOAL_ANGLES.UUU;
  const l1 = constants.poleLength1;
  const l2 = constants.poleLength2;
  const l3 = constants.poleLength3;
  const p1x = cartX + l1 * Math.sin(th1);
  const p1y = l1 * Math.cos(th1);
  const p2x = p1x + l2 * Math.sin(th2);
  const p2y = p1y + l2 * Math.cos(th2);
  return {
    cart: { x: cartX, y: 0 },
    poles: [
      { x: p1x, y: p1y, body: "lower" },
      { x: p2x, y: p2y, body: "mid" },
      { x: p2x + l3 * Math.sin(th3), y: p2y + l3 * Math.cos(th3), body: "tip" },
    ],
  };
}
