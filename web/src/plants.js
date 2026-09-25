import { DEFAULT_CONSTANTS, step as stepDouble, observe as observeDouble, normalizeObs, tipPositions as tipsDouble, grabForces as grabDouble } from "./physics.js";
import { GOAL_IDS as DOUBLE_GOALS, conditionedObs as condDouble, atGoal as atDouble, ghostTips as ghostDouble } from "./goals.js";
import { TRIPLE_CONSTANTS, step as stepTriple, observe as observeTriple, tipPositions as tipsTriple, grabForces as grabTriple } from "./physics-triple.js";
import { GOAL_IDS as TRIPLE_GOALS, atGoal as atTriple, ghostTips as ghostTriple } from "./goals-triple.js";
import { HANGING, HANGING_TRIPLE, DOWN_DOUBLE, DOWN_TRIPLE } from "./falloff.js";

function passObs(obs) {
  return obs;
}

export const PLANTS = {
  double: {
    id: "double",
    label: "double pendulum",
    // Same in-browser MPPI as the triple (mppi/double.json); the old MLP
    // (policy.json) is no longer loaded.
    mppiUrl: "/mppi/double.json",
    obsDim: 16,
    defaultGoal: "UU",
    goalIds: DOUBLE_GOALS,
    hanging: HANGING,
    initialState: DOWN_DOUBLE,
    autostart: true,
    hint: "grab the cart or either pole · 1–4 goal · tab cycle · p policy · a/d shove",
    touchHint: "drag the cart or a joint · tap a goal",
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
    serverCapable: true,
    mppiUrl: "/mppi/triple.json",
    obsDim: 11,
    defaultGoal: "UUU",
    // Upright first: key 1 and the first button are UUU.
    goalIds: [...TRIPLE_GOALS].reverse(),
    hanging: HANGING_TRIPLE,
    // Opens hanging at rest with UUU selected; the policy switches on shortly
    // after load (main.js AUTOSTART_MS) and swings it up.
    initialState: DOWN_TRIPLE,
    autostart: true,
    hint: "grab the cart or a joint · 1–8 goal · tab cycle · p policy · a/d shove",
    touchHint: "drag the cart or a joint · tap a goal",
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
