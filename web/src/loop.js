import { createFalloff } from "./falloff.js";

const STIFFNESS = 40;
const DAMPING = 4;
const PHYS_HZ = 120;
const POSE_KEYS = ["x", "th1", "th2", "th3"];

/**
 * Pose between two physics states for drawing, so motion is even at any
 * display rate. Teleports (respawn, reset) snap instead of sweeping.
 */
function blend(prev, next, alpha) {
  if (!prev || alpha >= 1) return next;
  const out = { ...next };
  for (const k of POSE_KEYS) {
    if (next[k] == null) continue;
    const d = next[k] - prev[k];
    if (Math.abs(d) > 0.5) return next;
    out[k] = prev[k] + d * alpha;
  }
  return out;
}

export function startLoop({
  plant,
  input,
  policy,
  getPolicy,
  constants,
  getGoal,
  getPolicyOn,
  getManualForce,
  getStarted,
  initialState,
  onFrame,
}) {
  const spawn = initialState || plant.initialState || plant.hanging;
  let state = { ...spawn };
  let prev = null;
  const fall = createFalloff(constants, spawn);
  let last = performance.now();
  let acc = 0;
  const dtMs = (constants.dt ?? 1 / PHYS_HZ) * 1000;
  const fmax = constants.forceLimit ?? 20;
  let raf = 0;
  let lastForce = 0;
  let wasActing = false;
  let simSteps = 0;
  let rateAt = performance.now();
  let rt = 1;
  let frames = 0;
  let waitFrames = 0;
  let stall = 0;
  let fps = 0;

  function frame(now) {
    raf = requestAnimationFrame(frame);
    acc += Math.min(64, now - last);
    last = now;
    let waited = false;
    const pointer = input.pointer;
    const goalId = typeof getGoal === "function" ? getGoal() : plant.defaultGoal;
    const driving = typeof getPolicyOn === "function" ? getPolicyOn() : true;
    try {
      while (acc >= dtMs) {
        const started = typeof getStarted !== "function" || getStarted();
        if (!started) {
          acc = 0;
          break;
        }
        const control = fall.inControl();
        let extraQ = null;
        if (control && pointer.active && pointer.body) {
          extraQ = plant.grabForces(state, pointer.world, pointer.body, STIFFNESS, DAMPING, constants);
        }
        const obs = plant.normalizeObs(plant.conditionedObs(plant.observe(state), goalId), constants);
        const actor = typeof getPolicy === "function" ? getPolicy() : policy;
        let force = 0;
        const acting = Boolean(control && actor && driving);
        // Stateful actors (the triple MPPI) restart when control resumes.
        if (acting && !wasActing) actor.reset?.();
        wasActing = acting;
        // A replanning actor holds the sim until its plan for this step is ready.
        if (acting && actor.prepare && !actor.prepare(state, goalId)) {
          acc = Math.min(acc, 64);
          waited = true;
          break;
        }
        if (acting) force = actor.act(obs, state, goalId);
        const manual = control && typeof getManualForce === "function" ? getManualForce() : 0;
        force = Math.max(-fmax, Math.min(fmax, force + manual));
        lastForce = control ? force : 0;
        prev = state;
        state = plant.step(state, lastForce, extraQ, constants);
        state = fall.step(state);
        acc -= dtMs;
        simSteps += 1;
      }
      frames += 1;
      if (waited) waitFrames += 1;
      if (now - rateAt > 2000) {
        rt = (simSteps * dtMs) / (now - rateAt);
        stall = frames ? waitFrames / frames : 0;
        fps = (frames * 1000) / (now - rateAt);
        simSteps = 0;
        frames = 0;
        waitFrames = 0;
        rateAt = now;
      }
      // Draw between the last two physics states (one step of display lag).
      const shown = blend(prev, state, acc / dtMs);
      const tips = plant.tipPositions(shown, constants);
      const visual = fall.visual();
      onFrame({
        state: shown,
        tips,
        pointer,
        force: lastForce,
        goalId,
        policyOn: driving && fall.inControl(),
        visual,
      });
    } catch (err) {
      console.warn(err);
    }
  }

  raf = requestAnimationFrame(frame);
  return {
    stop() {
      cancelAnimationFrame(raf);
    },
    getState() {
      return state;
    },
    setState(next) {
      state = next;
      prev = null;
      wasActing = false;
    },
    /**
     * Over the last ~2 s: sim speed vs wall clock, display fps, and the share
     * of frames where physics waited on the actor; plus the actor's own stats.
     */
    stats() {
      const actor = typeof getPolicy === "function" ? getPolicy() : policy;
      return { rt, fps, stall, ...(actor?.stats?.() ?? {}) };
    },
  };
}
