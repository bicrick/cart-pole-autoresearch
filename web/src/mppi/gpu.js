/**
 * WebGPU sample evaluator (same interface as pool.js):
 *   evaluate(goal, s0, U, n, seed) -> Promise<{samples: Float32Array[n*T], costs: Float32Array[n]}>
 * One compute dispatch generates and scores every sample; costs and samples are
 * read back for the (cheap) weighting on the CPU. Calls are serialized, and the
 * returned arrays are valid until the next call.
 */
import { WGSL, PRM_K, PRM_P, PRM_S0, PRM_LEN } from "./gpu-kernel.js";
import { toParams } from "./params.js";

const WG = 64;

export async function createGpuEvaluator(spec) {
  if (!navigator.gpu) throw new Error("WebGPU unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: WGSL });
  const info = await module.getCompilationInfo?.();
  const errors = info?.messages?.filter((m) => m.type === "error") ?? [];
  if (errors.length) throw new Error(`WGSL: ${errors.map((m) => `${m.lineNum}: ${m.message}`).join("; ")}`);
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });

  const goals = toParams(spec);
  const cfg = spec.mppi;
  const T = cfg.n_knots;
  const lim = 2 * spec.force_limit;
  const uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const prmBuf = device.createBuffer({ size: PRM_LEN * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const uBuf = device.createBuffer({ size: T * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const prm = new Float32Array(PRM_LEN);
  const cfgBytes = new ArrayBuffer(48);
  const cfgU32 = new Uint32Array(cfgBytes);
  const cfgF32 = new Float32Array(cfgBytes);

  let cap = 0;
  let samplesBuf, costsBuf, readSamples, readCosts, bindGroup;
  function ensure(n) {
    if (n <= cap) return;
    cap = Math.ceil(n / 1024) * 1024;
    [samplesBuf, costsBuf, readSamples, readCosts].forEach((b) => b?.destroy());
    const st = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    samplesBuf = device.createBuffer({ size: cap * T * 4, usage: st });
    costsBuf = device.createBuffer({ size: cap * 4, usage: st });
    const rd = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;
    readSamples = device.createBuffer({ size: cap * T * 4, usage: rd });
    readCosts = device.createBuffer({ size: cap * 4, usage: rd });
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: prmBuf } },
        { binding: 2, resource: { buffer: uBuf } },
        { binding: 3, resource: { buffer: samplesBuf } },
        { binding: 4, resource: { buffer: costsBuf } },
      ],
    });
  }

  let outSamples = new Float32Array(0);
  let outCosts = new Float32Array(0);
  let queue = Promise.resolve();

  async function run(goal, s0, U, n, seed, plans) {
    ensure(n);
    const g = goals[goal];
    prm.set(g.p, 0);
    prm.set(g.K, PRM_K);
    prm.set(g.P, PRM_P);
    prm.set(s0, PRM_S0);
    device.queue.writeBuffer(prmBuf, 0, prm);
    device.queue.writeBuffer(uBuf, 0, Float32Array.from(U));
    cfgU32[0] = n;
    cfgU32[1] = T;
    cfgU32[2] = plans ? 1 : 0;
    cfgU32[3] = seed >>> 0;
    cfgF32[4] = cfg.sigma;
    cfgF32[5] = cfg.sigma_wide;
    cfgU32[6] = n - Math.round(cfg.wide_frac * n);
    cfgF32[7] = cfg.noise_beta;
    cfgF32[8] = lim;
    device.queue.writeBuffer(uniform, 0, cfgBytes);
    if (plans) device.queue.writeBuffer(samplesBuf, 0, Float32Array.from(plans));

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / WG));
    pass.end();
    enc.copyBufferToBuffer(costsBuf, 0, readCosts, 0, n * 4);
    enc.copyBufferToBuffer(samplesBuf, 0, readSamples, 0, n * T * 4);
    device.queue.submit([enc.finish()]);

    await Promise.all([readCosts.mapAsync(GPUMapMode.READ, 0, n * 4), readSamples.mapAsync(GPUMapMode.READ, 0, n * T * 4)]);
    if (outCosts.length < n) outCosts = new Float32Array(cap);
    if (outSamples.length < n * T) outSamples = new Float32Array(cap * T);
    const costs = outCosts.subarray(0, n);
    const samples = outSamples.subarray(0, n * T);
    costs.set(new Float32Array(readCosts.getMappedRange(0, n * 4)));
    samples.set(new Float32Array(readSamples.getMappedRange(0, n * T * 4)));
    readCosts.unmap();
    readSamples.unmap();
    return { samples, costs };
  }

  function serialized(fn) {
    const p = queue.then(fn, fn);
    queue = p.catch(() => {});
    return p;
  }

  return {
    size: 1,
    backend: "webgpu",
    adapter: adapter.info?.description || adapter.info?.vendor || "gpu",
    evaluate(goal, s0, U, n, seed) {
      return serialized(() => run(goal, s0, U, n, seed, null));
    },
    /** Score given plans (row-major [n, T]) from s0: parity tests. */
    score(goal, s0, plans, n) {
      return serialized(() => run(goal, s0, new Float32Array(T), n, 0, plans));
    },
    terminate() {
      device.destroy();
    },
  };
}
