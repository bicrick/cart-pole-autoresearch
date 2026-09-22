import { DEFAULT_CONSTANTS, step as stepDouble, observe as observeDouble, normalizeObs, tipPositions as tipsDouble, grabForces as grabDouble } from "./physics.js";
import { GOAL_IDS as DOUBLE_GOALS, conditionedObs as condDouble, atGoal as atDouble, ghostTips as ghostDouble } from "./goals.js";
import { TRIPLE_CONSTANTS, step as stepTriple, observe as observeTriple, tipPositions as tipsTriple, grabForces as grabTriple } from "./physics-triple.js";
import { GOAL_IDS as TRIPLE_GOALS, atGoal as atTriple, ghostTips as ghostTriple } from "./goals-triple.js";
import { HANGING, HANGING_TRIPLE, DOWN_TRIPLE } from "./falloff.js";

function passObs(obs) {
  return obs;
}

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
    specialists: {
      DDD: "/policy-triple.json",
      DDU: "/policy-triple-ddu.json",
      DUD: "/policy-triple-dud.json",
      DUU: "/policy-triple-duu.json",
      UDD: "/policy-triple-udd.json",
      UDU: "/policy-triple-udu.json",
      UUD: "/policy-triple-uud.json",
      UUU: "/policy-triple-uuu.json",
    },
    obsDim: 11,
    defaultGoal: "DDD",
    goalIds: TRIPLE_GOALS,
    hanging: HANGING_TRIPLE,
    initialState: DOWN_TRIPLE,
    hint: "grab the cart or a joint · 1–8 goal · tab cycle · p policy · a/d shove",
    constants: TRIPLE_CONSTANTS,
    step: stepTriple,
    observe: observeTriple,
    normalizeObs: passObs,
    tipPositions: tipsTriple,
    grabForces: grabTriple,
    conditionedObs: passObs,
    atGoal: atTriple,
    ghostTips: ghostTriple,
  },
};

export function plantFromHash() {
  const raw = (window.location.hash || "").replace("#", "").toLowerCase();
  return PLANTS[raw] ? raw : "triple";
}

export function nextPlantId(id) {
  return id === "triple" ? "double" : "triple";
}
