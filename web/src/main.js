import { DEFAULT_CONSTANTS } from "./physics.js";
import { GOAL_IDS, atGoal } from "./goals.js";
import { loadPolicy } from "./policy.js";
import { createInput, createKeys } from "./input.js";
import { createCamera, draw } from "./render.js";
import { startLoop } from "./loop.js";

const canvas = document.getElementById("stage");
const statusEl = document.getElementById("status");
const readoutEl = document.getElementById("readout");
const policyBtn = document.getElementById("policy-toggle");
const goalButtons = [...document.querySelectorAll("[data-goal]")];
const ctx = canvas.getContext("2d");
const camera = createCamera();
const input = createInput(canvas, camera);

let currentGoal = "UU";
let policyOn = false;
let policyReady = false;

function setGoal(goal) {
  if (!GOAL_IDS.includes(goal)) return;
  currentGoal = goal;
  goalButtons.forEach((btn) => {
    const on = btn.dataset.goal === goal;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function setPolicyOn(on) {
  policyOn = Boolean(on) && policyReady;
  if (!policyBtn) return;
  policyBtn.textContent = policyOn ? "policy on" : policyReady ? "policy off" : "no policy";
  policyBtn.classList.toggle("is-active", policyOn);
  policyBtn.disabled = !policyReady;
}

function cycleGoal() {
  const i = GOAL_IDS.indexOf(currentGoal);
  setGoal(GOAL_IDS[(i + 1) % GOAL_IDS.length]);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function bindGoalPicker() {
  goalButtons.forEach((btn) => {
    btn.addEventListener("click", () => setGoal(btn.dataset.goal));
  });
}

function fmt(n, digits = 2) {
  const v = Number.isFinite(n) ? n : 0;
  return (v >= 0 ? "+" : "") + v.toFixed(digits);
}

function updateReadout(state, force, goalId) {
  if (!readoutEl) return;
  const hit = atGoal(state, goalId);
  readoutEl.textContent = [
    goalId,
    hit ? "at goal" : "seeking",
    `x ${fmt(state.x)}`,
    `θ1 ${fmt(state.th1)}`,
    `θ2 ${fmt(state.th2)}`,
    `u ${fmt(force, 1)} N`,
  ].join("  ·  ");
}

async function boot() {
  bindGoalPicker();
  setGoal(currentGoal);

  let policy = null;
  let constants = DEFAULT_CONSTANTS;
  try {
    policy = await loadPolicy("/policy.json");
    constants = { ...DEFAULT_CONSTANTS, ...(policy.spec.physics || {}) };
    const dim = policy.spec.obs_dim ?? constants.obsDim;
    policyReady = dim === 16;
    if (!policyReady) {
      setStatus(`policy obs_dim=${dim}`);
      policy = null;
    } else {
      const hidden = policy.spec.hidden ?? constants.hidden;
      setStatus(`${hidden}d mlp · on device`);
    }
  } catch (err) {
    console.warn(err);
    setStatus("physics only");
    policy = null;
    policyReady = false;
  }
  setPolicyOn(policyReady);

  if (policyBtn) {
    policyBtn.addEventListener("click", () => setPolicyOn(!policyOn));
  }

  const keys = createKeys({
    onGoal: setGoal,
    onTogglePolicy: () => setPolicyOn(!policyOn),
    onCycleGoal: cycleGoal,
  });

  let hudAt = 0;
  startLoop({
    canvas,
    ctx,
    camera,
    input,
    policy,
    constants,
    getGoal: () => currentGoal,
    getPolicyOn: () => policyOn,
    getManualForce: () => keys.manualForce(constants.forceLimit ?? 20),
    onFrame({ state, tips, pointer, force, goalId }) {
      draw(canvas, ctx, state, tips, camera, pointer, constants, goalId);
      const now = performance.now();
      if (now - hudAt > 80) {
        hudAt = now;
        updateReadout(state, force, goalId);
      }
    },
  });
}

boot();
