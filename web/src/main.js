import { DEFAULT_CONSTANTS } from "./physics.js";
import { GOAL_IDS } from "./goals.js";
import { loadPolicy } from "./policy.js";
import { createInput } from "./input.js";
import { createCamera, draw } from "./render.js";
import { startLoop } from "./loop.js";

const canvas = document.getElementById("stage");
const statusEl = document.getElementById("status");
const ctx = canvas.getContext("2d");
const camera = createCamera();
const input = createInput(canvas, camera);

let currentGoal = "UU";

function bindGoalPicker() {
  const buttons = document.querySelectorAll(".goal-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const goal = btn.dataset.goal;
      if (!GOAL_IDS.includes(goal)) return;
      currentGoal = goal;
      buttons.forEach((b) => b.classList.toggle("is-active", b === btn));
    });
  });
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

async function boot() {
  bindGoalPicker();
  let policy = null;
  let constants = DEFAULT_CONSTANTS;
  try {
    policy = await loadPolicy("/policy.json");
    constants = { ...DEFAULT_CONSTANTS, ...(policy.spec.physics || {}) };
    const dim = policy.spec.obs_dim ?? constants.obsDim;
    if (dim && dim !== 16) {
      setStatus(`policy obs_dim=${dim} (expected 16)`);
    } else {
      setStatus("policy on device");
    }
  } catch (err) {
    console.warn(err);
    setStatus("no policy — physics only");
  }

  startLoop({
    canvas,
    ctx,
    camera,
    input,
    policy,
    constants,
    getGoal: () => currentGoal,
    onFrame(state, tips, pointer) {
      draw(canvas, ctx, state, tips, camera, pointer);
    },
  });
}

boot();
