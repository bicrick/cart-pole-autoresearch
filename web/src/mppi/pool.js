/**
 * Sample evaluators for the MPPI controller. All expose
 *   evaluate(goal, s0, U, n, seed) -> Promise<{samples: Float64Array[n*T], costs: Float64Array[n]}>
 * The returned arrays are only valid until the next evaluate() call.
 * The worker pool splits the n samples into one contiguous chunk per worker.
 */
import { sampleChunk } from "./sampler.js";
import { toParams } from "./params.js";

/** What a worker needs to build its goal tables (params.js toParams). */
function plantSpec(spec) {
  return { plant: spec.plant, fields: spec.fields, goals: spec.goals, force_limit: spec.force_limit, mppi: spec.mppi };
}

function growable() {
  let samples = new Float64Array(0);
  let costs = new Float64Array(0);
  return (n, T) => {
    if (samples.length < n * T) samples = new Float64Array(n * T);
    if (costs.length < n) costs = new Float64Array(n);
    return { samples: samples.subarray(0, n * T), costs: costs.subarray(0, n) };
  };
}

/** Everything on the calling thread (node tests, or no Worker support). */
export function createLocalEvaluator(spec) {
  const goals = toParams(spec);
  const buffers = growable();
  return {
    size: 1,
    backend: "local",
    async evaluate(goal, s0, U, n, seed) {
      return sampleChunk(goals[goal], spec.mppi, s0, U, n, 0, n, seed, buffers(n, U.length));
    },
    terminate() {},
  };
}

export function createWorkerPool(spec, size) {
  const workers = [];
  // Per-worker chunk buffers, returned by the worker with each result.
  const spare = [];
  for (let i = 0; i < size; i += 1) {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    w.postMessage({ type: "goals", spec: plantSpec(spec) });
    workers.push(w);
    spare.push(null);
  }
  const output = growable();
  let jobId = 0;
  return {
    size,
    backend: "workers",
    evaluate(goal, s0, U, n, seed) {
      const id = (jobId += 1);
      const T = U.length;
      const per = Math.ceil(n / size);
      const out = output(n, T);
      const jobs = [];
      for (let k = 0; k < size; k += 1) {
        const i0 = k * per;
        const i1 = Math.min(n, i0 + per);
        if (i0 >= i1) break;
        jobs.push(
          new Promise((resolve) => {
            const w = workers[k];
            const onMsg = (event) => {
              const d = event.data;
              if (d.id !== id) return;
              w.removeEventListener("message", onMsg);
              out.samples.set(new Float64Array(d.samples, 0, d.m * T), d.i0 * T);
              out.costs.set(new Float64Array(d.costs, 0, d.m), d.i0);
              spare[k] = { samples: d.samples, costs: d.costs };
              resolve();
            };
            w.addEventListener("message", onMsg);
            const job = { type: "job", id, goal, s0, U, n, i0, i1, seed: (seed * 7919 + k * 104729) >>> 0 };
            const buf = spare[k];
            spare[k] = null;
            if (buf && buf.samples.byteLength >= (i1 - i0) * T * 8) {
              w.postMessage({ ...job, samples: buf.samples, costs: buf.costs }, [buf.samples, buf.costs]);
            } else {
              w.postMessage(job);
            }
          }),
        );
      }
      return Promise.all(jobs).then(() => out);
    },
    terminate() {
      workers.forEach((w) => w.terminate());
    },
  };
}
