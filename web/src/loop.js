import { createFalloff } from "./falloff.js";

const STIFFNESS = 40;
const DAMPING = 4;
const PHYS_HZ = 120;

export function startLoop({
  plant,
  input,
  policy,
  constants,
  getGoal,
  getPolicyOn,
  getManualForce,
  getStarted,
  initialState,
  onFrame,
}) {
  let state = { ...(initialState || plant.hanging) };
  const fall = createFalloff(constants, plant.hanging);
  let last = performance.now();
  let acc = 0;
  const dtMs = (constants.dt ?? 1 / PHYS_HZ) * 1000;
  const fmax = constants.forceLimit ?? 20;
  let raf = 0;
  let lastForce = 0;

  function frame(now) {
    raf = requestAnimationFrame(frame);
    acc += Math.min(64, now - last);
    last = now;
    const pointer = input.pointer;
    const goalId = typeof getGoal === "function" ? getGoal() : plant.defaultGoal;
    const driving = typeof getPolicyOn === "function" ? getPolicyOn() : true;
    try {
      while (acc >= dtMs) {
        const started = typeof getStarted !== "function" || getStarted();
        const control = started && fall.inControl();
        let extraQ = null;
        if (control && pointer.active && pointer.body) {
          extraQ = plant.grabForces(state, pointer.world, pointer.body, STIFFNESS, DAMPING, constants);
        }
        const obs = plant.normalizeObs(plant.conditionedObs(plant.observe(state), goalId), constants);
        let force = 0;
        if (control && policy && driving) force = policy.act(obs);
        const manual = control && typeof getManualForce === "function" ? getManualForce() : 0;
        force = Math.max(-fmax, Math.min(fmax, force + manual));
        lastForce = control ? force : 0;
        state = plant.step(state, lastForce, extraQ, constants);
        state = fall.step(state);
        acc -= dtMs;
      }
      const tips = plant.tipPositions(state, constants);
      const visual = fall.visual();
      onFrame({
        state,
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
    },
  };
}
