/**
 * Goal parameter tables from /mppi/<plant>.json as typed arrays, plus what
 * the plant-agnostic controller needs: the plant layout (state keys, angle
 * slots, rollout kernel) and the named scalars it reads (targets, gate, dt).
 */
import { rolloutCost } from "./rollout.js";
import { rolloutCostDouble } from "./rollout-double.js";

export const LAYOUTS = {
  triple: {
    keys: ["x", "xd", "th1", "th1d", "th2", "th2d", "th3", "th3d"],
    angles: [2, 4, 6],
    targets: ["t1", "t2", "t3"],
    rollout: rolloutCost,
  },
  double: {
    keys: ["x", "xd", "th1", "th1d", "th2", "th2d"],
    angles: [2, 4],
    targets: ["t1", "t2"],
    rollout: rolloutCostDouble,
  },
};

export function layoutOf(spec) {
  return LAYOUTS[spec.plant ?? "triple"];
}

export function toParams(spec) {
  const layout = layoutOf(spec);
  const at = (name) => spec.fields.indexOf(name);
  const out = {};
  for (const [goal, g] of Object.entries(spec.goals)) {
    const p = Float64Array.from(g.p);
    out[goal] = {
      p,
      K: Float64Array.from(g.K),
      P: Float64Array.from(g.P),
      layout,
      rollout: layout.rollout,
      targets: layout.targets.map((t) => p[at(t)]),
      gateIn: p[at("gate_in")],
      gateOut: p[at("gate_out")],
      dt: p[at("dt")],
      fmax: spec.force_limit,
    };
  }
  return out;
}
