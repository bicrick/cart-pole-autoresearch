/**
 * `?bench`: time real replans on this device and print a table over the page.
 * Each row is one sample count from a mid-swing state; the budget is one knot
 * of wall time (what the live loop must fit to hold real time).
 */
import { rolloutCost } from "./rollout.js";
import { toParams } from "./params.js";
import { stateArray } from "./controller.js";

const SWING = { x: 0.3, xd: 1.2, th1: 2.1, th1d: 4.0, th2: 2.8, th2d: -3.0, th3: 3.6, th3d: 2.5 };
const COUNTS = [4096, 2048, 1024, 512, 256];
const REPS = 15;

function pct(sorted, q) {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

/**
 * GPU (f32) costs vs the JS f64 kernel on the same random plans: bulk relative
 * error, and whether the cheapest samples (what MPPI keeps) agree.
 */
async function gpuParity(evaluator, spec) {
  const T = spec.mppi.n_knots;
  const n = 1024;
  const params = toParams(spec.goals).UUU;
  const s0 = stateArray(SWING);
  const plans = new Float64Array(n * T);
  let seed = 12345;
  for (let i = 0; i < plans.length; i += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    plans[i] = ((seed / 4294967296) * 2 - 1) * 20;
  }
  const { costs } = await evaluator.score("UUU", s0, plans, n);
  const gpu = Array.from(costs);
  const cpu = Array.from({ length: n }, (_, i) => rolloutCost(params.p, params.K, params.P, s0, 0, plans, i * T, T));
  const rel = cpu.map((c, i) => Math.abs(c - gpu[i]) / Math.max(Math.abs(c), 1)).sort((a, b) => a - b);
  const top = (v) => new Set(v.map((c, i) => [c, i]).sort((a, b) => a[0] - b[0]).slice(0, 32).map((x) => x[1]));
  const tc = top(cpu);
  let overlap = 0;
  top(gpu).forEach((i) => {
    if (tc.has(i)) overlap += 1;
  });
  return `gpu vs js: rel err median ${rel[n >> 1].toExponential(1)}, p95 ${rel[Math.floor(n * 0.95)].toExponential(1)}, top-32 overlap ${overlap}/32`;
}

export async function runBench(actor, spec, print) {
  const cfg = spec.mppi;
  const dt = spec.goals.UUU.p[12];
  const budget = cfg.knot * dt * 1000;
  const stats = actor.stats();
  const lines = [
    `mppi bench · ${stats.backend === "webgpu" ? `webgpu (${actor.evaluator.adapter})` : `${stats.backend} · ${stats.workers} workers`} · ${navigator.hardwareConcurrency} cores`,
    `profile ${spec.profile ?? "default"} · replan every ${cfg.knot} steps · horizon ${cfg.n_knots} knots · budget ${budget.toFixed(1)} ms`,
    "",
    "samples   median ms   p95 ms   Msteps/s   fits",
  ];
  print(lines.join("\n") + "\n(running)");
  for (let w = 0; w < 3; w += 1) await actor.timeReplan(SWING, "UUU", 1024);
  let best = 0;
  for (const n of COUNTS) {
    const t = [];
    for (let r = 0; r < REPS; r += 1) t.push(await actor.timeReplan(SWING, "UUU", n));
    t.sort((a, b) => a - b);
    const med = pct(t, 0.5);
    const p95 = pct(t, 0.95);
    const steps = (n * cfg.n_knots * cfg.knot) / (cfg.rollout_sub ?? 1);
    const fits = p95 < 0.8 * budget;
    if (fits && n > best) best = n;
    lines.push(
      `${String(n).padStart(7)}   ${med.toFixed(1).padStart(9)}   ${p95.toFixed(1).padStart(6)}   ${(steps / med / 1e3)
        .toFixed(1)
        .padStart(8)}   ${fits ? "yes" : "no"}`,
    );
    print(lines.join("\n") + "\n(running)");
  }
  lines.push("", best ? `real time up to ~${best} samples on this device` : "no sample count fits the budget");
  if (actor.evaluator?.score) lines.push(await gpuParity(actor.evaluator, spec));
  print(lines.join("\n"));
  return lines.join("\n");
}
