import { loadPolicy } from "./policy.js";
import { createInput, createKeys } from "./input.js";
import { createCamera, draw } from "./render.js";
import { startLoop } from "./loop.js";
import { PLANTS, plantFromHash, nextPlantId } from "./plants.js";
import { applyEmbedTheme, bindTheme } from "./theme.js";
import { startRemoteLoop } from "./remote-loop.js";
import { loadMppiActor } from "./mppi/load.js";
import { runBench } from "./mppi/bench.js";

const page = document.querySelector(".page");
const canvas = document.getElementById("stage");
const statusEl = document.getElementById("status");
const readoutEl = document.getElementById("readout");
const policyBtn = document.getElementById("policy-toggle");
const plantBtn = document.getElementById("plant-toggle");
const goalsNav = document.getElementById("goals");
const hintEl = document.querySelector(".hint");
const startEl = document.getElementById("start");
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

// The triple runs the MPPI teacher in the browser (Web Workers). With ?sim (or
// ?sim=ws://host:port) it instead renders the Python sim server
// (scripts/mppi-server.sh).
const params = new URLSearchParams(window.location.search);
const SIM_URL = params.has("sim") ? params.get("sim") || "ws://127.0.0.1:8765" : null;
// ?debug shows the engine line (backend, samples, fps, state) and status text.
const DEBUG = params.has("debug") || params.has("bench");
// ?embed=1 is the chromeless cut used inside the personal-site iframe.
// ?theme=light|dark selects the palette and does not touch localStorage.
const EMBED = params.has("embed");
const EMBED_HOLD_MS = 2800;
const THEME_MSG = "cart-pole-theme";
const TOUCH = window.matchMedia?.("(pointer: coarse)").matches ?? false;
// Plants with `autostart` switch the policy on this long after the controller loads.
const AUTOSTART_MS = 1200;
if (DEBUG) page.classList.add("is-debug");
if (EMBED) {
  page.classList.add("is-embed");
  camera.embed = true;
  const theme = params.get("theme");
  if (theme === "dark" || theme === "light") applyEmbedTheme(theme === "dark");
}

function isRemote(p) {
  return Boolean(p.serverCapable && SIM_URL);
}

function showBench(actor) {
  const pre = document.createElement("pre");
  pre.className = "bench";
  page.append(pre);
  runBench(actor, actor.spec, (text) => {
    pre.textContent = text;
  }).then((text) => {
    window.__benchResult = text;
  });
}

function currentActor() {
  return activePolicy;
}

function setGoal(goal) {
  if (!plant.goalIds.includes(goal)) return;
  currentGoal = goal;
  [...goalsNav.querySelectorAll("[data-goal]")].forEach((btn) => {
    const on = btn.dataset.goal === goal;
    btn.classList.toggle("is-active", on);
    btn.setAttribute("aria-pressed", on ? "true" : "false");
  });
  if (isRemote(plant) || plant.mppiUrl) {
    policyReady = true;
    setPolicyOn(started);
  }
}

// Phones stack the goal grid in the footer (above state and hint) instead of the top bar.
const PHONE = window.matchMedia("(max-width: 760px)");
const navInner = document.querySelector(".nav-inner");
const barInner = document.querySelector(".bar-inner");

function placeGoals() {
  if (PHONE.matches) {
    if (goalsNav.parentElement !== barInner) barInner.prepend(goalsNav);
  } else if (goalsNav.parentElement !== navInner) {
    navInner.insertBefore(goalsNav, navInner.querySelector(".meta"));
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
    btn.addEventListener("click", () => {
      begin();
      setGoal(id);
    });
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

/** Status text is engine detail (debug only) unless `alert` (loading, offline, errors). */
function setStatus(text, alert = false) {
  if (!statusEl) return;
  statusEl.textContent = text;
  statusEl.classList.toggle("is-alert", alert);
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

function field(label, value) {
  const k = document.createElement("span");
  k.className = "k";
  k.textContent = label;
  const v = document.createElement("span");
  v.className = "v";
  v.textContent = value;
  return [k, v];
}

function updateReadout(state, force, goalId) {
  if (!readoutEl) return;
  const hit = plant.atGoal(state, goalId);
  if (!DEBUG) {
    const state = !policyOn ? "policy off" : hit ? "at goal" : "seeking";
    const key = `${goalId}|${state}`;
    if (readoutEl.dataset.key !== key) {
      readoutEl.dataset.key = key;
      readoutEl.replaceChildren(...field("target", goalId), ...field("state", state));
    }
    return;
  }
  delete readoutEl.dataset.key;
  const parts = [
    goalId,
    hit ? "at goal" : "seeking",
    `x ${fmt(state.x)}`,
    `θ1 ${fmt(state.th1)}`,
    `θ2 ${fmt(state.th2)}`,
  ];
  if (state.th3 != null) parts.push(`θ3 ${fmt(state.th3)}`);
  parts.push(`u ${fmt(force, 1)} N`);
  if (loop?.stats && (isRemote(plant) || plant.mppiUrl)) {
    const { rt, ms, samples, fps, stall } = loop.stats();
    if (samples) parts.push(`mppi ${samples}`);
    parts.push(`${Math.round((rt ?? 0) * 100)}% speed · ${(ms ?? 0).toFixed(0)} ms/plan`);
    if (fps) parts.push(`${Math.round(fps)} fps${stall > 0.005 ? ` · ${Math.round(stall * 100)}% waits` : ""}`);
  }
  readoutEl.textContent = parts.join("  ·  ");
}

function applyPlantChrome() {
  document.title = plant.label;
  if (plantBtn) plantBtn.textContent = plant.label;
  page.dataset.plant = plant.id;
  if (hintEl) hintEl.textContent = TOUCH ? plant.touchHint ?? plant.hint : plant.hint;
  if (startEl) startEl.textContent = TOUCH ? "tap to start" : "click to start";
  if (window.location.hash.replace("#", "") !== plant.id) {
    history.replaceState(null, "", `#${plant.id}`);
  }
}

async function loadPlantPolicy(next) {
  if (next.mppiUrl) {
    try {
      const actor = await loadMppiActor(next.mppiUrl, next.step);
      const { backend, workers } = actor.stats();
      setStatus(backend === "workers" ? `mppi · ${workers} workers` : `mppi · ${backend}`);
      if (params.has("bench")) showBench(actor);
      return { nextPolicy: actor, nextConstants: { ...next.constants }, ready: true };
    } catch (err) {
      console.warn(err);
      setStatus("controller failed to load", true);
      return { nextPolicy: null, nextConstants: { ...next.constants }, ready: false };
    }
  }
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
    setStatus("physics only", true);
    nextPolicy = null;
    ready = false;
  }
  return { nextPolicy, nextConstants, ready };
}

let embedHeld = 0;
let embedAt = performance.now();

function onPlantFrame({ state, tips, pointer, force, goalId, policyOn: driving, visual }) {
  if (EMBED && started && loop) {
    const now = performance.now();
    const dt = now - embedAt;
    embedAt = now;
    if (plant.atGoal(state, goalId)) embedHeld += dt;
    else embedHeld = 0;
    if (embedHeld > EMBED_HOLD_MS) {
      embedHeld = 0;
      loop.setState({ ...(plant.initialState || plant.hanging) });
      activePolicy?.reset?.();
    }
  }
  draw(canvas, ctx, state, tips, camera, pointer, constants, goalId, driving, visual, plant.ghostTips);
  const now = performance.now();
  if (now - hudAt > 80) {
    hudAt = now;
    updateReadout(state, force, goalId);
  }
}

function startPlantLoop(nextPolicy, nextConstants) {
  if (loop) loop.stop();
  constants = nextConstants;
  activePolicy = nextPolicy;
  if (isRemote(plant)) {
    loop = startRemoteLoop({
      url: SIM_URL,
      plant,
      input,
      constants,
      getGoal: () => currentGoal,
      getPolicyOn: () => policyOn,
      getStarted: () => started,
      getManualForce: () => keys.manualForce(constants.forceLimit ?? 40),
      onFrame: onPlantFrame,
      onStatus: (text) => setStatus(text, /offline|connecting/.test(text)),
    });
    return;
  }
  loop = startLoop({
    plant,
    input,
    policy: nextPolicy,
    getPolicy: currentActor,
    constants,
    getGoal: () => currentGoal,
    getPolicyOn: () => policyOn,
    getStarted: () => started,
    getManualForce: () => (started ? keys.manualForce(constants.forceLimit ?? 20) : 0),
    initialState: plant.initialState || plant.hanging,
    onFrame: onPlantFrame,
  });
}

async function switchPlant(id) {
  const next = PLANTS[id];
  if (!next || next.id === plant.id) return;
  plant = next;
  currentGoal = plant.defaultGoal;
  applyPlantChrome();
  renderGoalButtons();
  setStatus("loading", true);
  if (isRemote(plant)) {
    startPlantLoop(null, { ...plant.constants });
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
  placeGoals();
  PHONE.addEventListener("change", placeGoals);
  applyPlantChrome();
  renderGoalButtons();
  if (isRemote(plant)) {
    setPolicyOn(false);
    startPlantLoop(null, { ...plant.constants });
  } else {
    const { nextPolicy, nextConstants, ready } = await loadPlantPolicy(plant);
    policyReady = ready;
    setPolicyOn(false);
    startPlantLoop(nextPolicy, nextConstants);
    if (ready && plant.autostart && !params.has("bench")) setTimeout(begin, EMBED ? 400 : AUTOSTART_MS);
  }

  if (EMBED) {
    window.addEventListener("message", (event) => {
      const data = event.data;
      if (!data || data.type !== THEME_MSG) return;
      if (data.theme !== "dark" && data.theme !== "light") return;
      applyEmbedTheme(data.theme === "dark");
    });
  }

  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn && !EMBED) bindTheme(themeBtn);

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
