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
        }
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
