/**
 * Sample evaluators for the MPPI controller. Both expose
 *   evaluate(goal, s0, U, n, seed) -> Promise<{samples: Float64Array[n*T], costs: Float64Array[n]}>
 * The worker pool splits the n samples into one contiguous chunk per worker.
 */
import { sampleChunk } from "./sampler.js";
import { toParams } from "./params.js";

function assemble(n, T, parts) {
  const samples = new Float64Array(n * T);
  const costs = new Float64Array(n);
  for (const part of parts) {
    samples.set(part.samples, part.i0 * T);
    costs.set(part.costs, part.i0);
  }
  return { samples, costs };
}

/** Everything on the calling thread (node tests, or no Worker support). */
export function createLocalEvaluator(spec) {
  const goals = toParams(spec.goals);
  return {
    size: 1,
    async evaluate(goal, s0, U, n, seed) {
      const part = sampleChunk(goals[goal], spec.mppi, s0, U, n, 0, n, seed);
      return { samples: part.samples, costs: part.costs };
    },
    terminate() {},
  };
}

export function createWorkerPool(spec, size) {
  const workers = [];
  for (let i = 0; i < size; i += 1) {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    w.postMessage({ type: "goals", goals: spec.goals, cfg: spec.mppi });
    workers.push(w);
  }
  let jobId = 0;
  return {
    size,
    evaluate(goal, s0, U, n, seed) {
      const id = (jobId += 1);
      const T = U.length;
      const per = Math.ceil(n / size);
      const jobs = [];
      for (let k = 0; k < size; k += 1) {
        const i0 = k * per;
        const i1 = Math.min(n, i0 + per);
        if (i0 >= i1) break;
        jobs.push(
          new Promise((resolve) => {
            const w = workers[k];
            const onMsg = (event) => {
              if (event.data.id !== id) return;
              w.removeEventListener("message", onMsg);
              resolve(event.data);
            };
            w.addEventListener("message", onMsg);
            w.postMessage({ type: "job", id, goal, s0, U, n, i0, i1, seed: (seed * 7919 + k * 104729) >>> 0 });
          }),
        );
      }
      return Promise.all(jobs).then((parts) => assemble(n, T, parts));
    },
    terminate() {
      workers.forEach((w) => w.terminate());
    },
  };
}
