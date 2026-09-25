function matvec(weight, bias, x) {
  const out = new Array(weight.length);
  for (let i = 0; i < weight.length; i += 1) {
    let sum = bias[i];
    const row = weight[i];
    for (let j = 0; j < row.length; j += 1) {
      sum += row[j] * x[j];
    }
    out[i] = sum;
  }
  return out;
}

function tanhVec(x) {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    out[i] = Math.tanh(x[i]);
  }
  return out;
}

function reluVec(x) {
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    const v = x[i];
    out[i] = v > 0 ? v : 0;
  }
  return out;
}

/** Gated target-LQR force from the 11-D triple obs (x, xd, sin/cos x3, w x3). */
export function anchorForce(anchor, obs) {
  const { K, target, gate_in: lo, gate_out: hi } = anchor;
  const e = [0, 1, 2].map((j) => {
    const s = obs[2 + 2 * j];
    const c = obs[3 + 2 * j];
    const ct = Math.cos(target[j]);
    const st = Math.sin(target[j]);
    return Math.atan2(s * ct - c * st, c * ct + s * st);
  });
  const d = e.reduce((acc, v) => acc + (1 - Math.cos(v)), 0);
  const gate = Math.max(0, Math.min(1, (hi - d) / (hi - lo)));
  if (gate === 0) return 0;
  const fb =
    K[0] * obs[0] + K[1] * obs[1] + K[2] * e[0] + K[3] * obs[8] +
    K[4] * e[1] + K[5] * obs[9] + K[6] * e[2] + K[7] * obs[10];
  return -gate * fb;
}

export function createPolicy(spec) {
  const layers = spec.layers;
  const forceLimit = spec.force_limit ?? spec.physics?.forceLimit ?? 20;
  return {
    spec,
    forceLimit,
    act(obs) {
      let h = obs;
      for (const layer of layers) {
        if (layer.type === "linear") {
          h = matvec(layer.weight, layer.bias, h);
        } else if (layer.type === "tanh") {
          h = tanhVec(h);
        } else if (layer.type === "relu") {
          h = reluVec(h);
        }
      }
      if (spec.anchor) {
        const u = anchorForce(spec.anchor, obs) + Math.tanh(h[0]) * spec.residual_scale;
        return Math.max(-forceLimit, Math.min(forceLimit, u));
      }
      return Math.tanh(h[0]) * forceLimit;
    },
  };
}

export async function loadPolicy(url = "/policy.json") {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load policy: ${res.status}`);
  }
  return createPolicy(await res.json());
}
