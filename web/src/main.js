import { loadPolicy } from "./policy.js";
import { createInput, createKeys } from "./input.js";
import { createCamera, draw } from "./render.js";
import { startLoop } from "./loop.js";
import { PLANTS, plantFromHash, nextPlantId } from "./plants.js";
import { equilibriumState } from "./goals-triple.js";
import { bindTheme } from "./theme.js";

const page = document.querySelector(".page");
const canvas = document.getElementById("stage");
const statusEl = document.getElementById("status");
const readoutEl = document.getElementById("readout");
const policyBtn = document.getElementById("policy-toggle");
const plantBtn = document.getElementById("plant-toggle");
const goalsNav = document.getElementById("goals");
const hintEl = document.querySelector(".hint");
const ctx = canvas.getContext("2d");
const camera = createCamera();
const input = createInput(canvas, camera);

let plant = PLANTS[plantFromHash()];
let currentGoal = plant.defaultGoal;
let policyOn = false;
let policyReady = false;
let started = false;
let loop = null;
let constants = plant.constants;
let activePolicy = null;
const specialistCache = new Map();

function setGoal(goal) {
  if (!plant.goalIds.includes(goal)) return;
  currentGoal = goal;
  [...goalsNav.querySelectorAll("[data-goal]")].forEach((btn) => {
    const on = btn.dataset.goal === goal;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
  if (plant.specialists) useSpecialist(goal);
}

async function useSpecialist(goal) {
  const url = plant.specialists?.[goal];
  if (!url) {
    if (currentGoal !== goal) return;
    activePolicy = null;
    policyReady = false;
    setPolicyOn(false);
    setStatus("no hold yet");
    if (loop) loop.setState(equilibriumState(goal));
    return;
  }
  if (!specialistCache.has(url)) specialistCache.set(url, loadPolicy(url));
  try {
    const next = await specialistCache.get(url);
    if (currentGoal !== goal) return;
    const dim = next.spec.obs_dim ?? next.spec.physics?.obsDim;
    activePolicy = dim === plant.obsDim ? next : null;
    policyReady = Boolean(activePolicy);
    setPolicyOn(started && policyReady);
    if (loop && policyReady) loop.setState(equilibriumState(goal));
    const hidden = next.spec.hidden ?? "";
    setStatus(policyReady ? `${goal} hold · ${hidden}` : `policy obs_dim=${dim}`);
  } catch (err) {
    console.warn(err);
    if (currentGoal !== goal) return;
    activePolicy = null;
    policyReady = false;
    setPolicyOn(false);
    setStatus("physics only");
  }
}

function renderGoalButtons() {
  goalsNav.replaceChildren();
  plant.goalIds.forEach((id, i) => {
    if (i) {
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.setAttribute("aria-hidden", "true");
      dot.textContent = "·";
      goalsNav.append(dot);
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "text-btn";
    btn.dataset.goal = id;
    btn.textContent = id;
    btn.setAttribute("aria-pressed", "false");
    btn.addEventListener("click", () => setGoal(id));
    goalsNav.append(btn);
  });
  setGoal(currentGoal);
}

function setPolicyOn(on) {
  policyOn = Boolean(on) && policyReady;
  if (!policyBtn) return;
  policyBtn.textContent = policyOn ? "policy on" : policyReady ? "policy off" : "no policy";
  policyBtn.classList.toggle("is-active", policyOn);
  policyBtn.disabled = !policyReady;
}

function cycleGoal(dir = 1) {
  const n = plant.goalIds.length;
  if (!n) return;
  const i = plant.goalIds.indexOf(currentGoal);
  setGoal(plant.goalIds[(i + dir + n) % n]);
}

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function begin() {
  if (started) return;
  started = true;
  page.classList.add("is-started");
  setPolicyOn(true);
}

function fmt(n, digits = 2) {
  const v = Number.isFinite(n) ? n : 0;
  return (v >= 0 ? "+" : "") + v.toFixed(digits);
}

function updateReadout(state, force, goalId) {
  if (!readoutEl) return;
  const hit = plant.atGoal(state, goalId);
  const parts = [
    goalId,
    hit ? "at goal" : "seeking",
    `x ${fmt(state.x)}`,
    `θ1 ${fmt(state.th1)}`,
    `θ2 ${fmt(state.th2)}`,
  ];
  if (state.th3 != null) parts.push(`θ3 ${fmt(state.th3)}`);
  parts.push(`u ${fmt(force, 1)} N`);
  readoutEl.textContent = parts.join("  ·  ");
}

function applyPlantChrome() {
  document.title = plant.label;
  if (plantBtn) plantBtn.textContent = plant.label;
  page.dataset.plant = plant.id;
  if (hintEl) hintEl.textContent = plant.hint;
  if (window.location.hash.replace("#", "") !== plant.id) {
    history.replaceState(null, "", `#${plant.id}`);
  }
}

async function loadPlantPolicy(next) {
  let nextPolicy = null;
  let nextConstants = { ...next.constants };
  let ready = false;
  try {
    nextPolicy = await loadPolicy(next.policyUrl);
    nextConstants = { ...next.constants, ...(nextPolicy.spec.physics || {}) };
    const dim = nextPolicy.spec.obs_dim ?? nextConstants.obsDim;
    ready = dim === next.obsDim;
    if (!ready) {
      setStatus(`policy obs_dim=${dim}`);
      nextPolicy = null;
    } else {
      const hidden = nextPolicy.spec.hidden ?? nextConstants.hidden;
      setStatus(`${hidden}d mlp · ${next.id}`);
    }
  } catch (err) {
    console.warn(err);
    setStatus("physics only");
    nextPolicy = null;
    ready = false;
  }
  return { nextPolicy, nextConstants, ready };
}

function startPlantLoop(nextPolicy, nextConstants) {
  if (loop) loop.stop();
  constants = nextConstants;
  activePolicy = nextPolicy;
  loop = startLoop({
    plant,
    input,
    policy: nextPolicy,
    getPolicy: () => activePolicy,
    constants,
    getGoal: () => currentGoal,
    getPolicyOn: () => policyOn,
    getStarted: () => started,
    getManualForce: () => (started ? keys.manualForce(constants.forceLimit ?? 20) : 0),
    initialState: plant.initialState || plant.hanging,
    onFrame({ state, tips, pointer, force, goalId, policyOn: driving, visual }) {
      draw(canvas, ctx, state, tips, camera, pointer, constants, goalId, driving, visual, plant.ghostTips);
      const now = performance.now();
      if (now - hudAt > 80) {
        hudAt = now;
        updateReadout(state, force, goalId);
      }
    },
  });
}

async function switchPlant(id) {
  const next = PLANTS[id];
  if (!next || next.id === plant.id) return;
  plant = next;
  currentGoal = plant.defaultGoal;
  applyPlantChrome();
  renderGoalButtons();
  setStatus("loading");
  if (plant.specialists) {
    await useSpecialist(currentGoal);
    startPlantLoop(activePolicy, { ...plant.constants });
    return;
  }
  const { nextPolicy, nextConstants, ready } = await loadPlantPolicy(plant);
  policyReady = ready;
  setPolicyOn(started && ready);
  startPlantLoop(nextPolicy, nextConstants);
}

const keys = createKeys({
  onGoal: (goal) => {
    if (started) setGoal(goal);
  },
  onTogglePolicy: () => {
    if (started) setPolicyOn(!policyOn);
  },
  onCycleGoal: (shift = false) => {
    if (started) cycleGoal(shift ? -1 : 1);
    if (document.activeElement?.blur) document.activeElement.blur();
  },
  getGoals: () => plant.goalIds,
});

let hudAt = 0;

async function boot() {
  applyPlantChrome();
  renderGoalButtons();
  if (plant.specialists) {
    await useSpecialist(currentGoal);
    setPolicyOn(false);
    startPlantLoop(activePolicy, { ...plant.constants });
  } else {
    const { nextPolicy, nextConstants, ready } = await loadPlantPolicy(plant);
    policyReady = ready;
    setPolicyOn(false);
    startPlantLoop(nextPolicy, nextConstants);
  }

  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) bindTheme(page, themeBtn);

  if (policyBtn) {
    policyBtn.addEventListener("click", () => setPolicyOn(!policyOn));
  }
  if (plantBtn) {
    plantBtn.addEventListener("click", () => switchPlant(nextPlantId(plant.id)));
  }
  window.addEventListener("hashchange", () => {
    const id = plantFromHash();
    if (id !== plant.id) switchPlant(id);
  });

  canvas.addEventListener("pointerdown", begin);
  window.addEventListener("keydown", (event) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    begin();
  });
}

boot();
