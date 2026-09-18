import { grabForces, observe, normalizeObs, step, tipPositions } from "./physics.js";

const STIFFNESS = 40;
const DAMPING = 4;
const PHYS_HZ = 120;

export function startLoop({ canvas, ctx, camera, input, policy, constants, onFrame }) {
  let state = { x: 0, xd: 0, th1: 0.05, th1d: 0, th2: -0.04, th2d: 0 };
  let last = performance.now();
  let acc = 0;
  const dtMs = (constants.dt ?? 1 / PHYS_HZ) * 1000;
  let raf = 0;

  function frame(now) {
    acc += Math.min(64, now - last);
    last = now;
    const pointer = input.pointer;
    while (acc >= dtMs) {
      let extraQ = null;
      if (pointer.active && pointer.body) {
        extraQ = grabForces(state, pointer.world, pointer.body, STIFFNESS, DAMPING, constants);
      }
      const obs = normalizeObs(observe(state), constants);
      const force = policy ? policy.act(obs) : 0;
      state = step(state, force, extraQ, constants);
      acc -= dtMs;
    }
    const tips = tipPositions(state, constants);
    onFrame(state, tips, pointer);
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
  };
}
