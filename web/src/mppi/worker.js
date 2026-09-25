/**
 * MPPI rollout worker. Messages:
 *   {type: "goals", goals: {GOAL: {p, K, P}}, cfg}   once, at startup
 *   {type: "job", id, goal, s0, U, n, i0, i1, seed}  -> {id, i0, samples, costs}
 */
import { sampleChunk } from "./sampler.js";
import { toParams } from "./params.js";

let goals = {};
let cfg = null;

self.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "goals") {
    goals = toParams(msg.goals);
    cfg = msg.cfg;
    return;
  }
  if (msg.type === "job") {
    const { samples, costs } = sampleChunk(goals[msg.goal], cfg, msg.s0, msg.U, msg.n, msg.i0, msg.i1, msg.seed);
    self.postMessage({ id: msg.id, i0: msg.i0, samples, costs }, [samples.buffer, costs.buffer]);
  }
};
