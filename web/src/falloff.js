export const HANGING = {
  x: 0,
  xd: 0,
  th1: Math.PI + 0.16,
  th1d: 0.28,
  th2: Math.PI - 0.1,
  th2d: -0.18,
};

export const HANGING_TRIPLE = {
  ...HANGING,
  th3: Math.PI + 0.08,
  th3d: 0.12,
};

export const UPRIGHT_TRIPLE = {
  x: 0,
  xd: 0,
  th1: 0,
  th1d: 0,
  th2: 0,
  th2d: 0,
  th3: 0,
  th3d: 0,
};

export const DOWN_TRIPLE = {
  x: 0,
  xd: 0,
  th1: Math.PI,
  th1d: 0,
  th2: Math.PI,
  th2d: 0,
  th3: Math.PI,
  th3d: 0,
};

export function createFalloff(constants, hanging = HANGING) {
  const track = constants.trackLimit ?? 2.4;
  const g = constants.gravity ?? 9.81;
  const dt = constants.dt ?? 1 / 120;
  let phase = "live";
  let y = 0;
  let yd = 0;
  let fade = 1;
  let goneT = 0;

  function visual() {
    return {
      y,
      alpha: fade,
      muted: phase === "falling" || phase === "gone",
    };
  }

  return {
    visual,
    inControl() {
      return phase === "live" || phase === "entering";
    },
    phase() {
      return phase;
    },
    step(state) {
      if (phase === "live") {
        if (Math.abs(state.x) > track) {
          phase = "falling";
          y = 0;
          yd = Math.max(0.4, Math.abs(state.xd) * 0.15);
        }
        return state;
      }

      if (phase === "falling") {
        yd += g * dt;
        y -= yd * dt;
        if (y < -3.4 || Math.abs(state.x) > track + 5) {
          phase = "gone";
          goneT = 0;
          fade = 0;
        }
        return state;
      }

      if (phase === "gone") {
        goneT += dt;
        fade = 0;
        if (goneT > 0.32) {
          phase = "entering";
          fade = 0;
          y = 0.1;
          yd = 0;
          return { ...hanging };
        }
        return state;
      }

      fade = Math.min(1, fade + dt / 0.5);
      y *= 0.9;
      if (fade >= 1 && Math.abs(y) < 0.004) {
        y = 0;
        phase = "live";
      }
      return state;
    },
  };
}
