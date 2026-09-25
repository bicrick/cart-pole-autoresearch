/**
 * Build the triple's in-browser MPPI actor: one Web Worker pool for the page
 * (reused across plant switches), in-thread fallback without Worker support.
 * `?mppi=N` caps the sample count (default: the teacher's 4096).
 */
import { createMppiController } from "./controller.js";
import { createLocalEvaluator, createWorkerPool } from "./pool.js";

let shared = null;

function workerCount() {
  const hc = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(hc - 1, 10));
}

export async function loadMppiActor(url) {
  if (!shared) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`mppi spec ${url}: ${res.status}`);
    const spec = await res.json();
    const evaluator = typeof Worker === "function" ? createWorkerPool(spec, workerCount()) : createLocalEvaluator(spec);
    shared = { spec, evaluator };
  }
  const cap = Number(new URLSearchParams(window.location.search).get("mppi")) || undefined;
  return createMppiController(shared.spec, shared.evaluator, { maxSamples: cap });
}
