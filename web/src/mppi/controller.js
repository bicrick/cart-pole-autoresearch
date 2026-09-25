/**
 * In-browser MPPI teacher for the cart-triple: port of train/mppi/mppi.py
 * (`MPPI.optimize` / `MPPI.act`) running its samples on an evaluator
 * (Web Worker pool, or in-thread).
 *
 * Every `knot` physics steps the controller replans from the exact current
 * state. The loop asks `prepare()` before each step; it returns false while a
 * replan is in flight, so physics waits for the plan exactly like the Python
 * sim server does, and nothing is ever applied late.
 *
 * `n` adapts so a replan fits `budgetMs` (default: one knot of wall time).
 */
import { rolloutCost } from "./rollout.js";
import { toParams } from "./params.js";

const TWO_PI = 2 * Math.PI;
const KEYS = ["x", "xd", "th1", "th1d", "th2", "th2d", "th3", "th3d"];
const LAMBDAS = Array.from({ length: 25 }, (_, i) => 10 ** (-4 + (4 * i) / 24));

function wrap(a) {
  const r = (a + Math.PI) % TWO_PI;
  return (r < 0 ? r + TWO_PI : r) - Math.PI;
}

export function stateArray(state) {
  return Float64Array.from(KEYS, (k) => state[k]);
}

/** Gated target-LQR force (mppi.py make_anchor) from a state array. */
export function anchorForce(params, s) {
  const { p, K } = params;
  const e1 = wrap(s[2] - p[14]);
  const e2 = wrap(s[4] - p[15]);
  const e3 = wrap(s[6] - p[16]);
  const d = (1 - Math.cos(e1)) + (1 - Math.cos(e2)) + (1 - Math.cos(e3));
  const gate = Math.max(0, Math.min(1, (p[34] - d) / (p[34] - p[33])));
  if (gate === 0) return 0;
  const fb = K[0] * s[0] + K[1] * s[1] + K[2] * e1 + K[3] * s[3] + K[4] * e2 + K[5] * s[5] + K[6] * e3 + K[7] * s[7];
  return -gate * fb;
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
  const goals = toParams(spec.goals);
  const T = cfg.n_knots;
  const knot = cfg.knot;
  const fmax = spec.force_limit;
  const dt = goals.UUU.p[12];
  const budgetMs = opts.budgetMs ?? knot * dt * 1000 * 0.8;
  const maxN = opts.maxSamples ?? cfg.n_samples;
  const minN = opts.minSamples ?? 256;
  const adapt = opts.adapt ?? true;
  let n = opts.samples ?? maxN;
  let uff = new Float64Array(T);
  let tick = 0;
  let planned = false;
  let pending = false;
  let gen = 0;
  let goal = null;
  let seed = opts.seed ?? 1;
  let ms = 0;

  async function optimize(g, s0, U) {
    const params = goals[g];
    const { samples, costs } = await evaluator.evaluate(g, s0, U, n, (seed += 1));
    const w = mppiWeights(costs, cfg.ess);
    const mean = new Float64Array(T);
    let bi = 0;
    for (let i = 0; i < n; i += 1) {
      if (costs[i] < costs[bi]) bi = i;
      const wi = w[i];
      if (wi < 1e-12) continue;
      const off = i * T;
      for (let t = 0; t < T; t += 1) mean[t] += wi * samples[off + t];
    }
    const best = samples.slice(bi * T, bi * T + T);
    const cm = rolloutCost(params.p, params.K, params.P, s0, 0, mean, 0, T);
    const cb = rolloutCost(params.p, params.K, params.P, s0, 0, best, 0, T);
    // samples[0] is the unperturbed previous plan.
    if (costs[0] <= cm && costs[0] <= cb) return U;
    return cm <= cb ? mean : best;
  }

  function startReplan(s) {
    const mine = gen;
    const U = tick === 0 ? uff : Float64Array.from({ length: T }, (_, t) => (t + 1 < T ? uff[t + 1] : 0));
    pending = true;
    const t0 = performance.now();
    optimize(goal, s, U).then((next) => {
      if (mine !== gen) return;
      uff = next;
      pending = false;
      planned = true;
      const took = performance.now() - t0;
      ms = ms ? 0.8 * ms + 0.2 * took : took;
      if (adapt) {
        if (ms > budgetMs && n > minN) n = Math.max(minN, Math.floor((n * 0.85) / 64) * 64);
        else if (ms < 0.6 * budgetMs && n < maxN) n = Math.min(maxN, Math.ceil((n * 1.1) / 64) * 64);
      }
    });
  }

  return {
    reset() {
      gen += 1;
      uff = new Float64Array(T);
      tick = 0;
      planned = false;
      pending = false;
    },
    /** True when this step can run; starts a replan at knot boundaries. */
    prepare(state, goalId) {
      if (goalId !== goal) {
        goal = goalId;
        this.reset();
      }
      if (tick % knot !== 0 || planned) return true;
      if (!pending) startReplan(stateArray(state));
      return false;
    },
    act(_obs, state) {
      const s = stateArray(state);
      const u = Math.max(-fmax, Math.min(fmax, anchorForce(goals[goal], s) + uff[0]));
      tick += 1;
      if (tick % knot === 0) planned = false;
      return u;
    },
    stats() {
      return { samples: n, ms, workers: evaluator.size };
    },
    plan() {
      return uff;
    },
  };
}
