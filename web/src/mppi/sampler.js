/**
 * One chunk of MPPI samples: colored noise around the current plan, clipped,
 * then scored with rolloutCost. Shared by the Web Workers and the in-thread
 * fallback. Noise matches train/mppi/mppi.py `_noise`: AR(1) over knots with
 * coefficient `beta`, per-sample sigma (the last `wide_frac` of samples use
 * `sigma_wide`), and sample 0 is the unperturbed plan.
 */
import { rolloutCost } from "./rollout.js";

/** xorshift128+ -> uniform (0,1); seeded per job so workers never share a stream. */
export function makeRng(seed) {
  let s0 = (seed ^ 0x9e3779b9) >>> 0 || 1;
  let s1 = (Math.imul(seed, 0x85ebca6b) ^ 0xc2b2ae35) >>> 0 || 2;
  let s2 = (Math.imul(seed ^ 0x27d4eb2f, 0x165667b1) >>> 0) || 3;
  let s3 = (seed * 2654435761) >>> 0 || 4;
  function next() {
    // xoshiro128** (32-bit), two draws per double.
    const r = Math.imul(rotl(Math.imul(s1, 5), 7), 9) >>> 0;
    const t = s1 << 9;
    s2 ^= s0;
    s3 ^= s1;
    s1 ^= s2;
    s0 ^= s3;
    s2 ^= t;
    s3 = rotl(s3, 11);
    return r;
  }
  for (let i = 0; i < 8; i += 1) next();
  let spare = null;
  return {
    uniform() {
      return (next() + 0.5) / 4294967296;
    },
    normal() {
      if (spare !== null) {
        const v = spare;
        spare = null;
        return v;
      }
      const u = (next() + 0.5) / 4294967296;
      const v = (next() + 0.5) / 4294967296;
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    },
  };
}

function rotl(x, k) {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

/**
 * Samples [i0, i1) of an n-sample replan. Returns { samples, costs } with
 * samples row-major [i1 - i0, nKnots]; `out` buffers are reused when big enough.
 */
export function sampleChunk(params, cfg, s0, U, n, i0, i1, seed, out = null) {
  const T = U.length;
  const m = i1 - i0;
  const samples = out?.samples?.length >= m * T ? out.samples.subarray(0, m * T) : new Float64Array(m * T);
  const costs = out?.costs?.length >= m ? out.costs.subarray(0, m) : new Float64Array(m);
  const rng = makeRng(seed);
  const beta = cfg.noise_beta;
  const scale = Math.sqrt(1 - beta * beta);
  const lim = 2 * params.p[13];
  const wideFrom = n - Math.round(cfg.wide_frac * n);
  for (let r = 0; r < m; r += 1) {
    const i = i0 + r;
    const off = r * T;
    if (i === 0) {
      for (let t = 0; t < T; t += 1) samples[off + t] = Math.max(-lim, Math.min(lim, U[t]));
    } else {
      const sig = i >= wideFrom ? cfg.sigma_wide : cfg.sigma;
      let e = rng.normal();
      for (let t = 0; t < T; t += 1) {
        if (t > 0) e = beta * e + scale * rng.normal();
        const v = U[t] + sig * e;
        samples[off + t] = v < -lim ? -lim : v > lim ? lim : v;
      }
    }
    costs[r] = rolloutCost(params.p, params.K, params.P, s0, 0, samples, off, T);
  }
  return { samples, costs };
}
