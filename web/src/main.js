import { DEFAULT_CONSTANTS } from "./physics.js";
import { loadPolicy } from "./policy.js";
import { createInput } from "./input.js";
import { createCamera, draw } from "./render.js";
import { startLoop } from "./loop.js";

const canvas = document.getElementById("stage");
const statusEl = document.getElementById("status");
const ctx = canvas.getContext("2d");
const camera = createCamera();
const input = createInput(canvas, camera);

function resize() {
  // drawing code sizes the backing store from CSS pixels
}
window.addEventListener("resize", resize);

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

async function boot() {
  let policy = null;
  let constants = DEFAULT_CONSTANTS;
  try {
    policy = await loadPolicy("/policy.json");
    constants = { ...DEFAULT_CONSTANTS, ...(policy.spec.physics || {}) };
    setStatus("policy on device");
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
    onFrame(state, tips, pointer) {
      draw(canvas, ctx, state, tips, camera, pointer);
    },
  });
}

boot();
