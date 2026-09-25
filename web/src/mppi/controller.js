/**
 * In-browser MPPI controller: port of train/mppi/mppi.py (`MPPI.optimize` /
 * `MPPI.act`) running its samples on an evaluator (WebGPU, Web Worker pool,
 * or in-thread). Plant-agnostic: the spec's layout (params.js) supplies the
 * state keys, angle slots and rollout kernel (triple or double).
 *
 * Every `knot` physics steps the controller switches to a new plan. The loop
 * asks `prepare()` before each step; it returns false while the plan for this
 * step is still in flight, so nothing is ever applied late. Without `step`
 * each plan is computed from the exact state at its boundary (like the Python
 * sim server); with `step` it is computed one knot ahead from the predicted
 * state, which is exact unless the user drags or shoves in between.
 *
 * `n` adapts so a replan fits `budgetMs` (default: one knot of wall time).
 */
import { LAYOUTS, toParams } from "./params.js";

const TWO_PI = 2 * Math.PI;
const LAMBDAS = Array.from({ length: 25 }, (_, i) => 10 ** (-4 + (4 * i) / 24));

function wrap(a) {
  const r = (a + Math.PI) % TWO_PI;
  return (r < 0 ? r + TWO_PI : r) - Math.PI;
}

export function stateArray(state, layout = LAYOUTS.triple) {
  return Float64Array.from(layout.keys, (k) => state[k]);
}

/** Gated target-LQR force (mppi.py make_anchor) from a state array. */
export function anchorForce(params, s) {
  const { K, targets, layout, gateIn, gateOut } = params;
  const e = Float64Array.from(s);
  let d = 0;
  layout.angles.forEach((i, j) => {
    e[i] = wrap(s[i] - targets[j]);
    d += 1 - Math.cos(e[i]);
  });
  const gate = Math.max(0, Math.min(1, (gateOut - d) / (gateOut - gateIn)));
  if (gate === 0) return 0;
  let fb = 0;
  for (let i = 0; i < e.length; i += 1) fb += K[i] * e[i];
  return -gate * fb;
}

/** Quiet at the target (mppi.py `MPPI.holding`): replans use hold_samples. */
export function holding(params, cfg, s) {
  if (!cfg.hold_samples) return false;
  const { targets, layout } = params;
  let d = 0;
  let spin = 0;
  layout.angles.forEach((i, j) => {
    d += 1 - Math.cos(s[i] - targets[j]);
    spin += s[i + 1] * s[i + 1];
  });
  return d < cfg.hold_d && spin < cfg.hold_spin;
}

/** ESS-targeted softmax weights (mppi.py `_weights`). */
export function mppiWeights(S, targetEss) {
  const n = S.length;
  let smin = Infinity;
  for (let i = 0; i < n; i += 1) if (S[i] < smin) smin = S[i];
  const d = new Float64Array(n);
  for (let i = 0; i < n; i += 1) d[i] = S[i] - smin;
  const sorted = Float64Array.from(d).sort();
  const mid = (n - 1) / 2;
  const med = 0.5 * (sorted[Math.floor(mid)] + sorted[Math.ceil(mid)]);
  const ref = Math.max(med, 1e-6);
  let best = null;
  let bestGap = Infinity;
  const w = new Float64Array(n);
  for (const lamScale of LAMBDAS) {
    const lam = ref * lamScale;
    let sum = 0;
    for (let i = 0; i < n; i += 1) {
      w[i] = Math.exp(-d[i] / lam);
      sum += w[i];
    }
    let sq = 0;
    for (let i = 0; i < n; i += 1) {
      w[i] /= sum;
      sq += w[i] * w[i];
    }
    const gap = Math.abs(1 / sq - targetEss);
    if (gap < bestGap) {
      bestGap = gap;
      best = Float64Array.from(w);
    }
  }
  return best;
}

export function createMppiController(spec, evaluator, opts = {}) {
  const cfg = spec.mppi;
  const goals = toParams(spec);
  const layout = Object.values(goals)[0].layout;
  const T = cfg.n_knots;
  const knot = cfg.knot;
  const fmax = spec.force_limit;
  const dt = Object.values(goals)[0].dt;
  const toArr = (state) => stateArray(state, layout);
  const budgetMs = opts.budgetMs ?? knot * dt * 1000 * 0.8;
  const maxN = opts.maxSamples ?? cfg.n_samples;
  const minN = opts.minSamples ?? 256;
  const adapt = opts.adapt ?? true;
  let n = opts.samples ?? maxN;
  let uff = new Float64Array(T);
  let tick = 0;
  // The page plant's `step` (physics-triple.js / physics.js) enables pipelined replanning.
  const step = opts.step ?? null;
  let planned = false;
  let job = null;
  let gen = 0;
  let goal = null;
  let seed = opts.seed ?? 1;
  let ms = 0;
  let lastN = n;

  async function optimize(g, s0, U) {
    const params = goals[g];
    const hold = holding(params, cfg, s0);
    const k = hold ? cfg.hold_samples : n;
    lastN = k;
    const { samples, costs } = await evaluator.evaluate(g, s0, U, k, (seed += 1));
    const w = mppiWeights(costs, cfg.ess);
    const mean = new Float64Array(T);
    let bi = 0;
    for (let i = 0; i < k; i += 1) {
      if (costs[i] < costs[bi]) bi = i;
      const wi = w[i];
      if (wi < 1e-12) continue;
      const off = i * T;
      for (let t = 0; t < T; t += 1) mean[t] += wi * samples[off + t];
    }
    const best = samples.slice(bi * T, bi * T + T);
    const cm = params.rollout(params.p, params.K, params.P, s0, 0, mean, 0, T);
    const cb = params.rollout(params.p, params.K, params.P, s0, 0, best, 0, T);
    // samples[0] is the unperturbed previous plan.
    if (costs[0] <= cm && costs[0] <= cb) return U;
    return cm <= cb ? mean : best;
  }

  function shifted(plan) {
    const out = new Float64Array(T);
    out.set(plan.subarray(1));
    return out;
  }

  /** State one knot ahead under the current plan (no user input). */
  function predict(state) {
    let s = { ...state };
    for (let i = 0; i < knot; i += 1) {
      const u = Math.max(-fmax, Math.min(fmax, anchorForce(goals[goal], toArr(s)) + uff[0]));
      s = step(s, u, null);
    }
    return toArr(s);
  }

  /** Start the replan that the knot boundary at `forTick` will use. */
  function launch(forTick, s0, U) {
    const mine = gen;
    const job = { forTick, plan: null };
    const t0 = performance.now();
    optimize(goal, s0, U).then((next) => {
      if (mine !== gen) return;
      job.plan = next;
      const took = performance.now() - t0;
      ms = ms ? 0.8 * ms + 0.2 * took : took;
      // Hold replans are cheap and say nothing about the swing-up budget.
      if (adapt && lastN === n) {
        if (ms > budgetMs && n > minN) n = Math.max(minN, Math.floor((n * 0.85) / 64) * 64);
        else if (ms < 0.6 * budgetMs && n < maxN) n = Math.min(maxN, Math.ceil((n * 1.1) / 64) * 64);
      }
    });
    return job;
  }

  return {
    reset() {
      gen += 1;
      uff = new Float64Array(T);
      tick = 0;
      planned = false;
      job = null;
    },
    /**
     * True when this step can run. At a knot boundary it swaps in the plan
     * computed for it; with `step` (pipelined) that plan was started one knot
     * earlier from the predicted state, so the sim only waits if a replan
     * takes longer than a knot.
     */
    prepare(state, goalId) {
      if (goalId !== goal) {
        goal = goalId;
        this.reset();
      }
      if (tick % knot !== 0 || planned) return true;
      if (!job || job.forTick !== tick) job = launch(tick, toArr(state), tick === 0 ? uff : shifted(uff));
      if (!job.plan) return false;
      uff = job.plan;
      planned = true;
      if (step) job = launch(tick + knot, predict(state), shifted(uff));
      return true;
    },
    act(_obs, state) {
      const s = toArr(state);
      const u = Math.max(-fmax, Math.min(fmax, anchorForce(goals[goal], s) + uff[0]));
      tick += 1;
      if (tick % knot === 0) planned = false;
      return u;
    },
    stats() {
      return { samples: lastN, ms, workers: evaluator.size, backend: evaluator.backend ?? "workers" };
    },
    /** One replan from `state` with `samples` samples (bench); resolves to wall ms. */
    async timeReplan(state, goalId, samples) {
      const keep = n;
      n = samples;
      const t0 = performance.now();
      await optimize(goalId, toArr(state), new Float64Array(T));
      n = keep;
      return performance.now() - t0;
    },
    plan() {
      return uff;
    },
  };
}
