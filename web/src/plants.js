import { DEFAULT_CONSTANTS, step as stepDouble, observe as observeDouble, normalizeObs, tipPositions as tipsDouble, grabForces as grabDouble } from "./physics.js";
import { GOAL_IDS as DOUBLE_GOALS, conditionedObs as condDouble, atGoal as atDouble, ghostTips as ghostDouble } from "./goals.js";
import { TRIPLE_CONSTANTS, step as stepTriple, observe as observeTriple, tipPositions as tipsTriple, grabForces as grabTriple } from "./physics-triple.js";
import { GOAL_IDS as TRIPLE_GOALS, atGoal as atTriple, ghostTips as ghostTriple } from "./goals-triple.js";
import {
  QUAD_CONSTANTS,
  SINGLE_CONSTANTS,
  step as stepNlink,
  observe as observeNlink,
  tipPositions as tipsNlink,
  grabForces as grabNlink,
} from "./physics-nlink.js";
import { goalIds as nlinkGoals, atGoal as atNlink, ghostTips as ghostNlink, equilibriumState } from "./goals-nlink.js";
import { HANGING, HANGING_TRIPLE, DOWN_DOUBLE, DOWN_TRIPLE } from "./falloff.js";

const HANGING_QUAD = { ...HANGING_TRIPLE, th4: Math.PI - 0.06, th4d: -0.1 };
const HANGING_SINGLE = { x: 0, xd: 0, th1: HANGING.th1, th1d: HANGING.th1d };

function passObs(obs) {
  return obs;
}

/** n-link plant functions bound to `defaults` (the controller calls `step` without constants). */
function nlinkPlant(defaults) {
  return {
    constants: defaults,
    step: (s, u, q, c = defaults) => stepNlink(s, u, q, c),
    observe: (s, c = defaults) => observeNlink(s, c),
    tipPositions: (s, c = defaults) => tipsNlink(s, c),
    grabForces: (s, t, body, k, d, c = defaults) => grabNlink(s, t, body, k, d, c),
  };
}

export const PLANTS = {
  single: {
    id: "single",
    label: "single pendulum",
    mppiUrl: "/mppi/single.json",
    obsDim: 4,
    defaultGoal: "U",
    goalIds: ["U", "D"],
    hanging: HANGING_SINGLE,
    initialState: equilibriumState("D"),
    autostart: true,
    hint: "grab the cart or the pole · 1–2 goal · tab cycle · p controller · a/d shove",
    touchHint: "drag the cart or the pole · tap a goal",
    ...nlinkPlant(SINGLE_CONSTANTS),
    normalizeObs: passObs,
    conditionedObs: passObs,
    atGoal: atNlink,
    ghostTips: ghostNlink,
  },
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
    hint: "grab the cart or either pole · 1–4 goal · tab cycle · p controller · a/d shove",
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
    hint: "grab the cart or a joint · 1–8 goal · tab cycle · p controller · a/d shove",
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
  quad: {
    id: "quad",
    label: "quad pendulum",
    mppiUrl: "/mppi/quad.json",
    obsDim: 14,
    // UUUU swings up only ~40% of the time within 25 s; DDUU is reliable.
    defaultGoal: "DDUU",
    goalIds: [...nlinkGoals(4)].reverse(),
    hanging: HANGING_QUAD,
    initialState: equilibriumState("DDDD"),
    autostart: true,
    hint: "grab the cart or a joint · tab cycle goals · p controller · a/d shove",
    touchHint: "drag the cart or a joint · tap a goal",
    ...nlinkPlant(QUAD_CONSTANTS),
    normalizeObs: passObs,
    conditionedObs: passObs,
    atGoal: atNlink,
    ghostTips: ghostNlink,
  },
};

/** Carousel order: by link count. The page opens on the triple. */
export const PLANT_ORDER = ["single", "double", "triple", "quad"];

export function plantFromHash() {
  const raw = (window.location.hash || "").replace("#", "").toLowerCase();
  return PLANTS[raw] ? raw : "triple";
}

export function stepPlantId(id, dir = 1) {
  const n = PLANT_ORDER.length;
  return PLANT_ORDER[(PLANT_ORDER.indexOf(id) + dir + n) % n];
}
