/**
 * Build a plant's in-browser MPPI actor from /mppi/<plant>.json. One sample
 * evaluator per plant for the page, reused across plant switches.
 *   ?mppi=N      cap the sample count (default: the spec's n_samples)
 *   ?workers=N   worker count (default: see workerCount)
 *   ?backend=    webgpu (default when available) | workers | local
 */
import { createMppiController } from "./controller.js";
import { createLocalEvaluator, createWorkerPool } from "./pool.js";
import { createGpuEvaluator } from "./gpu.js";

const shared = new Map();

function query() {
  return new URLSearchParams(window.location.search);
}

function isMobile() {
  return navigator.userAgentData?.mobile ?? /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
}

/**
 * Equal chunks finish with the slowest worker, so skip the efficiency cores:
 * desktops keep one core for the page, phones (typically 2-4 big cores) keep
 * two and stop at 6.
 */
export function workerCount(hc = navigator.hardwareConcurrency || 4, mobile = isMobile()) {
  return mobile ? Math.max(1, Math.min(hc - 2, 6)) : Math.max(1, Math.min(hc - 1, 10));
}

/** The spec with one profile's MPPI config, and each goal's p patched to match. */
export function withProfile(spec, name) {
  const cfg = spec.profiles?.[name];
  if (!cfg) return spec;
  const idx = (f) => spec.fields.indexOf(f);
  const goals = {};
  for (const [goal, g] of Object.entries(spec.goals)) {
    const p = g.p.slice();
    p[idx("knot")] = cfg.knot;
    p[idx("sub")] = cfg.rollout_sub;
    p[idx("early_exit")] = cfg.early_exit ? 1 : 0;
    goals[goal] = { ...g, p };
  }
  return { ...spec, mppi: cfg, goals, profile: name };
}

/**
 * WebGPU when available (full profile; triple only, the double is light
 * enough for workers), else Web Workers (full on desktop, lite on phones),
 * else in-thread. ?profile=full|lite overrides the choice.
 */
async function createEvaluator(raw) {
  const q = query();
  const backend = q.get("backend");
  const pick = (fallback) => withProfile(raw, q.get("profile") || fallback);
  if (backend === "local" || typeof Worker !== "function") {
    const spec = pick("lite");
    return { spec, evaluator: createLocalEvaluator(spec) };
  }
  const gpuKernel = !raw.plant || raw.plant === "triple";
  if (backend !== "workers" && gpuKernel && navigator.gpu) {
    const spec = pick("full");
    try {
      return { spec, evaluator: await createGpuEvaluator(spec) };
    } catch (err) {
      console.warn("WebGPU evaluator unavailable, using workers:", err);
    }
  }
  const spec = pick(isMobile() ? "lite" : "full");
  return { spec, evaluator: createWorkerPool(spec, Number(q.get("workers")) || workerCount()) };
}

/** `step` is the page plant's step (physics-triple.js / physics.js), used to plan one knot ahead. */
export async function loadMppiActor(url, step) {
  if (!shared.has(url)) {
    shared.set(url, (async () => {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`mppi spec ${url}: ${res.status}`);
      return createEvaluator(await res.json());
    })());
  }
  const { spec, evaluator } = await shared.get(url);
  const cap = Number(query().get("mppi")) || undefined;
  // ?pipeline=0 plans from the exact boundary state instead (physics waits each knot).
  const pipelined = query().get("pipeline") !== "0";
  const actor = createMppiController(spec, evaluator, {
    maxSamples: cap,
    step: pipelined ? step : null,
  });
  actor.spec = spec;
  actor.evaluator = evaluator;
  return actor;
}
