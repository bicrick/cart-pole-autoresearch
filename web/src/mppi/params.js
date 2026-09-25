/** Goal parameter tables from /mppi/triple.json as typed arrays. */
export function toParams(goals) {
  const out = {};
  for (const [goal, g] of Object.entries(goals)) {
    out[goal] = { p: Float64Array.from(g.p), K: Float64Array.from(g.K), P: Float64Array.from(g.P) };
  }
  return out;
}
