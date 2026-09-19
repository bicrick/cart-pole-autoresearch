import { DEFAULT_CONSTANTS, step as stepDouble, observe as observeDouble, normalizeObs, tipPositions as tipsDouble, grabForces as grabDouble } from "./physics.js";
import { GOAL_IDS as DOUBLE_GOALS, conditionedObs as condDouble, atGoal as atDouble, ghostTips as ghostDouble } from "./goals.js";
import { TRIPLE_CONSTANTS, step as stepTriple, observe as observeTriple, tipPositions as tipsTriple, grabForces as grabTriple } from "./physics-triple.js";
import { GOAL_IDS as TRIPLE_GOALS, conditionedObs as condTriple, atGoal as atTriple, ghostTips as ghostTriple } from "./goals-triple.js";
import { HANGING, HANGING_TRIPLE } from "./falloff.js";

export const PLANTS = {
  double: {
    id: "double",
    label: "double pendulum",
    policyUrl: "/policy.json",
    obsDim: 16,
    defaultGoal: "UU",
    goalIds: DOUBLE_GOALS,
    hanging: HANGING,
    hint: "grab the cart or either pole · 1–4 goal · tab cycle · p policy · a/d shove",
    constants: DEFAULT_CONSTANTS,
    step: stepDouble,
    observe: observeDouble,
    normalizeObs,
    tipPositions: tipsDouble,
    grabForces: grabDouble,
    conditionedObs: condDouble,
    atGoal: atDouble,
    ghostTips: ghostDouble,
  },
  triple: {
    id: "triple",
    label: "triple pendulum",
    policyUrl: "/policy-triple.json",
    obsDim: 25,
    defaultGoal: "UUU",
    goalIds: TRIPLE_GOALS,
    hanging: HANGING_TRIPLE,
    hint: "grab the cart or a joint · 1–8 goal · tab cycle · p policy · a/d shove",
    constants: TRIPLE_CONSTANTS,
    step: stepTriple,
    observe: observeTriple,
    normalizeObs,
    tipPositions: tipsTriple,
    grabForces: grabTriple,
    conditionedObs: condTriple,
    atGoal: atTriple,
    ghostTips: ghostTriple,
  },
};

export function plantFromHash() {
  const raw = (window.location.hash || "").replace("#", "").toLowerCase();
  return PLANTS[raw] ? raw : "double";
}

export function nextPlantId(id) {
  return id === "triple" ? "double" : "triple";
}
