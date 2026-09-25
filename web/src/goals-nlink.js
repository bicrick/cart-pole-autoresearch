/**
 * The 2^n equilibria of the n-link plant: "U" (0, upright) or "D" (π, hanging)
 * per link, cart-most link first (train/mppi/nlink/goals.py).
 */
import { tipPositions } from "./physics-nlink.js";

export function goalIds(n) {
  const out = [""];
  for (let i = 0; i < n; i += 1) out.splice(0, out.length, ...out.flatMap((g) => [`${g}D`, `${g}U`]));
  return out;
}

export function goalAngles(goalId) {
  return [...goalId].map((ch) => (ch === "U" ? 0 : Math.PI));
}

export function atGoal(state, goalId, cosThresh = 0.95) {
  return goalAngles(goalId).every((t, i) => Math.cos(state[`th${i + 1}`] - t) > cosThresh);
}

export function equilibriumState(goalId) {
  const s = { x: 0, xd: 0 };
  goalAngles(goalId).forEach((t, i) => {
    s[`th${i + 1}`] = t;
    s[`th${i + 1}d`] = 0;
  });
  return s;
}

export function ghostTips(cartX, goalId, constants) {
  return tipPositions({ ...equilibriumState(goalId), x: cartX }, constants);
}
