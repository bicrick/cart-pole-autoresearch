function wrapAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

function stateVector(state) {
  return [state.x, state.xd, state.th1, state.th1d, state.th2, state.th2d, state.th3, state.th3d];
}

function angleFromHang(th) {
  return Math.abs(wrapAngle(th - Math.PI));
}

export function nearHang(state) {
  if (!state || state.th3 == null) return false;
  const slow = [state.xd, state.th1d, state.th2d, state.th3d].every((w) => Math.abs(w) < 0.5);
  return (
    Math.abs(state.x) < 0.3 &&
    angleFromHang(state.th1) < 0.4 &&
    angleFromHang(state.th2) < 0.4 &&
    angleFromHang(state.th3) < 0.4 &&
    slow
  );
}

/** Links still down, even if the hang hold has started to drift. */
export function hangingLinks(state) {
  if (!state || state.th3 == null) return false;
  return (
    Math.abs(state.x) < 1 &&
    angleFromHang(state.th1) < 0.6 &&
    angleFromHang(state.th2) < 0.6 &&
    angleFromHang(state.th3) < 0.6
  );
}

export function createSwingTrack(spec) {
  const refs = spec.refs;
  const forces = spec.forces;
  const gains = spec.gains;
  const limit = spec.force_limit ?? 40;
  let k = 0;
  let latched = false;

  return {
    get latched() {
      return latched;
    },
    reset() {
      k = 0;
      latched = false;
    },
    force(state) {
      if (latched || k >= refs.length) {
        latched = true;
        return null;
      }
      const ref = refs[k];
      const z = stateVector(state);
      const err = z.map((value, i) => (i === 2 || i === 4 || i === 6 ? wrapAngle(value - ref[i]) : value - ref[i]));
      let miss = 0;
      for (let i = 0; i < err.length; i += 1) miss = Math.max(miss, Math.abs(err[i]));
      if (k > 8 && miss > 1.5) {
        latched = true;
        return null;
      }
      const gain = gains[k];
      let u = forces[k];
      for (let i = 0; i < gain.length; i += 1) u -= gain[i] * err[i];
      u = Math.max(-limit, Math.min(limit, u));
      k += 1;
      const ang = Math.max(Math.abs(wrapAngle(state.th1)), Math.abs(wrapAngle(state.th2)), Math.abs(wrapAngle(state.th3)));
      const rate = Math.max(Math.abs(state.th1d), Math.abs(state.th2d), Math.abs(state.th3d));
      if (ang <= 0.03 && rate <= 0.01) {
        latched = true;
        return null;
      }
      return u;
    },
  };
}
