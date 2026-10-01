/**
 * Show reel for the chromeless ?embed=1 cut: each pendulum swings up to a
 * target, holds, moves to a second target from where it is, then hands over to
 * the next plant. Every scene has a time limit so a missed swing-up never
 * stalls the loop.
 *
 *   createReel({ startPlant }) -> { plant, goal, tick(dt, atGoal) }
 *
 * `tick` returns null, or { plant, goal } when the reel moves on. `plant` is
 * only set when the plant changes.
 */
export const REEL = [
  { plant: "single", goals: ["U"] },
  { plant: "double", goals: ["UU", "UD"] },
  { plant: "triple", goals: ["UUU", "UDU"] },
  // UUUU swings up only some of the time; DDUU is reliable.
  { plant: "quad", goals: ["DDUU"] },
];

const HOLD_MS = 2200;
const LIMIT_MS = 14000;

export function createReel({ startPlant }) {
  let scene = Math.max(0, REEL.findIndex((s) => s.plant === startPlant));
  let step = 0;
  let held = 0;
  let elapsed = 0;

  function advance() {
    held = 0;
    elapsed = 0;
    const current = REEL[scene];
    if (step + 1 < current.goals.length) {
      step += 1;
      return { goal: current.goals[step] };
    }
    scene = (scene + 1) % REEL.length;
    step = 0;
    return { plant: REEL[scene].plant, goal: REEL[scene].goals[0] };
  }

  return {
    get plant() {
      return REEL[scene].plant;
    },
    get goal() {
      return REEL[scene].goals[step];
    },
    tick(dt, atGoal) {
      elapsed += dt;
      held = atGoal ? held + dt : 0;
      if (held > HOLD_MS || elapsed > LIMIT_MS) return advance();
      return null;
    },
  };
}
