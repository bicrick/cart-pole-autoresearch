import { grabForces, observe, normalizeObs, step, tipPositions, offTrack } from "./physics.js";
import { conditionedObs } from "./goals.js";

const STIFFNESS = 40;
const DAMPING = 4;
const PHYS_HZ = 120;

function respawnState() {
  // Mild near-hang / mid-track respawn after falling into the void.
  const hang = Math.random() < 0.5;
  const jitter = () => (Math.random() - 0.5) * 0.2;
  if (hang) {
    return {
      x: (Math.random() - 0.5) * 1.5,
      xd: 0,
      th1: Math.PI + jitter(),
      th1d: 0,
      th2: Math.PI + jitter(),
      th2d: 0,
    };
  }
  return { x: 0, xd: 0, th1: 0.05 + jitter(), th1d: 0, th2: -0.04 + jitter(), th2d: 0 };
}

export function startLoop({
  canvas,
  ctx,
  camera,
  input,
  policy,
  constants,
  getGoal,
  getPolicyOn,
  getManualForce,
  onFrame,
}) {
  let state = { x: 0, xd: 0, th1: 0.05, th1d: 0, th2: -0.04, th2d: 0 };
  let last = performance.now();
  let acc = 0;
  const dtMs = (constants.dt ?? 1 / PHYS_HZ) * 1000;
  const fmax = constants.forceLimit ?? 20;
  let raf = 0;
  let lastForce = 0;

  function frame(now) {
    acc += Math.min(64, now - last);
    last = now;
    const pointer = input.pointer;
    const goalId = typeof getGoal === "function" ? getGoal() : "UU";
    while (acc >= dtMs) {
      let extraQ = null;
      if (pointer.active && pointer.body) {
        extraQ = grabForces(state, pointer.world, pointer.body, STIFFNESS, DAMPING, constants);
      }
      const obs = normalizeObs(conditionedObs(observe(state), goalId), constants);
      const policyOn = typeof getPolicyOn === "function" ? getPolicyOn() : true;
      let force = policy && policyOn ? policy.act(obs) : 0;
      const manual = typeof getManualForce === "function" ? getManualForce() : 0;
      force = Math.max(-fmax, Math.min(fmax, force + manual));
      lastForce = force;
      state = step(state, force, extraQ, constants);
      if (offTrack(state, constants)) {
        state = respawnState();
        lastForce = 0;
      }
      acc -= dtMs;
    }
    const tips = tipPositions(state, constants);
    onFrame({ state, tips, pointer, force: lastForce, goalId });
    raf = requestAnimationFrame(frame);
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
