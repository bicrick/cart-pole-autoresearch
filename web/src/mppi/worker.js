/**
 * MPPI rollout worker. Messages:
 *   {type: "goals", spec: {plant, fields, goals, force_limit, mppi}}   once, at startup
 *   {type: "job", id, goal, s0, U, n, i0, i1, seed, samples?, costs?}
 *     -> {id, i0, m, samples, costs}
 * `samples` / `costs` are ArrayBuffers handed back and forth so no replan allocates.
 */
import { sampleChunk } from "./sampler.js";
import { toParams } from "./params.js";

let goals = {};
let cfg = null;

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "goals") {
    goals = toParams(msg.spec);
    cfg = msg.spec.mppi;
    return;
  }
  if (msg.type === "job") {
    const out = msg.samples ? { samples: new Float64Array(msg.samples), costs: new Float64Array(msg.costs) } : null;
    const part = sampleChunk(goals[msg.goal], cfg, msg.s0, msg.U, msg.n, msg.i0, msg.i1, msg.seed, out);
    const samples = part.samples.buffer;
    const costs = part.costs.buffer;
    self.postMessage({ id: msg.id, i0: msg.i0, m: msg.i1 - msg.i0, samples, costs }, [samples, costs]);
  }
};
