/**
 * WGSL for the n-link sample evaluator, generated per link count: the math of
 * rollout-nlink.js / train/mppi/nlink/kernel_body.py in f32, with the mass
 * matrix, Cholesky solve and energy unrolled into scalars (no local arrays in
 * the step loop). Bindings, noise and the entry point are gpu-kernel.js's.
 *
 * prm layout: p (NS scalars + 5 link rows), then K [dim], P [dim*dim] row-major, s0 [dim].
 */
import { WGSL_COMMON, WGSL_MAIN } from "./gpu-kernel.js";
import { NS, ROW_M, ROW_L, ROW_C, ROW_S, ROW_T } from "./rollout-nlink.js";

const range = (n) => Array.from({ length: n }, (_, i) => i);

export function nlinkKernel(n) {
  const D = n + 1;
  const dim = 2 + 2 * n;
  const K = NS + 5 * n;
  const P = K + dim;
  const S0 = P + dim * dim;
  const LEN = S0 + dim;
  const I = range(n);
  const link = (row, i) => `prm[${NS + row * n + i}]`;
  const sum = (terms, zero = "0.0") => (terms.length ? terms.join(" + ") : zero);
  const Sm = (i, k) => `S${Math.max(i, k)}`;

  /** Declares `${out}` = pole KE + PE (cart frame) from the current th/w/sn/cs. */
  const energy = (out) => {
    const lines = [];
    I.forEach((i) => {
      const pv = i === 0 ? "0.0" : `${out}vx${i - 1}`;
      const pw = i === 0 ? "0.0" : `${out}vy${i - 1}`;
      const py = i === 0 ? "0.0" : `${out}y${i - 1}`;
      const pe = i === 0 ? "0.0" : `${out}e${i - 1}`;
      lines.push(
        `let ${out}vx${i} = ${pv} + l${i} * cs${i} * w${i};`,
        `let ${out}vy${i} = ${pw} - l${i} * sn${i} * w${i};`,
        `let ${out}y${i} = ${py} + l${i} * cs${i};`,
        `let ${out}e${i} = ${pe} + m${i} * (0.5 * (${out}vx${i} * ${out}vx${i} + ${out}vy${i} * ${out}vy${i}) + g * ${out}y${i});`,
      );
    });
    lines.push(`let ${out} = ${out}e${n - 1};`);
    return lines.join("\n      ");
  };

  const angle = () => sum(I.map((i) => `(1.0 - (cs${i} * tc${i} + sn${i} * ts${i}))`));

  const mass = [];
  mass.push(`let a0_0 = M + msum + 1e-8;`);
  I.forEach((i) => {
    mass.push(`let a${1 + i}_0 = S${i} * l${i} * cs${i};`);
    range(i + 1).forEach((k) => {
      const jit = k === i ? " + 1e-8" : "";
      mass.push(`let a${1 + i}_${1 + k} = ${Sm(i, k)} * l${i} * l${k} * (cs${i} * cs${k} + sn${i} * sn${k})${jit};`);
    });
  });
  mass.push(`let h0 = u - b * xd${I.map((i) => ` + S${i} * l${i} * sn${i} * w${i} * w${i}`).join("")};`);
  I.forEach((i) => {
    const coup = I.filter((k) => k !== i)
      .map((k) => ` - ${Sm(i, k)} * l${i} * l${k} * (sn${i} * cs${k} - cs${i} * sn${k}) * w${k} * w${k}`)
      .join("");
    mass.push(`let h${1 + i} = -c${i} * w${i} + S${i} * g * l${i} * sn${i}${coup};`);
  });

  const chol = [];
  range(D).forEach((j) => {
    const sq = range(j).map((k) => ` - L${j}_${k} * L${j}_${k}`).join("");
    chol.push(`let L${j}_${j} = sqrt(a${j}_${j}${sq});`);
    range(D).filter((i) => i > j).forEach((i) => {
      const dots = range(j).map((k) => ` - L${i}_${k} * L${j}_${k}`).join("");
      chol.push(`let L${i}_${j} = (a${i}_${j}${dots}) / L${j}_${j};`);
    });
  });
  range(D).forEach((i) => {
    chol.push(`let y${i} = (h${i}${range(i).map((k) => ` - L${i}_${k} * y${k}`).join("")}) / L${i}_${i};`);
  });
  range(D).reverse().forEach((i) => {
    const back = range(D).filter((k) => k > i).map((k) => ` - L${k}_${i} * z${k}`).join("");
    chol.push(`let z${i} = (y${i}${back}) / L${i}_${i};`);
  });

  const code = /* wgsl */ `
fn rollout(off: u32, T: u32) -> f32 {
  let M = prm[0]; let g = prm[1]; let b = prm[2]; let dt = prm[3]; let fmax = prm[4]; let eT = prm[5];
  let wAngle = prm[6]; let wVel = prm[7]; let wVelNear = prm[8]; let nearScale = prm[9];
  let wX = prm[10]; let wXd = prm[11]; let xSoft = prm[12]; let wBarrier = prm[13];
  let xDead = prm[14]; let wDead = prm[15]; let wEnergy = prm[16]; let wU = prm[17]; let wDu = prm[18];
  let wTl = prm[19]; let wTe = prm[20]; let gIn = prm[21]; let gOut = prm[22]; let anchorOn = prm[23];
  let knot = u32(prm[24]);
  let early = prm[25] > 0.0;
  let kdt = prm[24] * dt;
  let total = T * knot;
  ${I.map((i) => `let m${i} = ${link(ROW_M, i)}; let l${i} = ${link(ROW_L, i)}; let c${i} = ${link(ROW_C, i)}; let S${i} = ${link(ROW_S, i)};
  let tg${i} = ${link(ROW_T, i)}; let tc${i} = cos(tg${i}); let ts${i} = sin(tg${i});`).join("\n  ")}
  let msum = ${I.map((i) => `m${i}`).join(" + ")};

  var x = prm[${S0}]; var xd = prm[${S0 + 1}];
  ${I.map((i) => `var th${i} = prm[${S0 + 2 + 2 * i}]; var w${i} = prm[${S0 + 3 + 2 * i}]; var sn${i} = sin(th${i}); var cs${i} = cos(th${i});`).join("\n  ")}
  var acc: f32 = 0.0;
  var done: u32 = 0u;

  for (var j: u32 = 0u; j < T; j = j + 1u) {
    let r = samples[off + j];
    for (var q: u32 = 0u; q < knot; q = q + 1u) {
      var u = r;
      if (anchorOn > 0.0) {
        let d = ${angle()};
        let gate = clamp((gOut - d) / (gOut - gIn), 0.0, 1.0);
        if (gate > 0.0) {
          let fb = prm[${K}] * x + prm[${K + 1}] * xd${I.map((i) => ` + prm[${K + 2 + 2 * i}] * wrapA(th${i} - tg${i}) + prm[${K + 3 + 2 * i}] * w${i}`).join("")};
          u = r - gate * fb;
        }
      }
      u = clamp(fin(u), -fmax, fmax);
      xd = clamp(xd, -MAX_CART_VEL, MAX_CART_VEL);
      ${I.map((i) => `w${i} = clamp(w${i}, -MAX_ANG_VEL, MAX_ANG_VEL);`).join(" ")}
      ${mass.join("\n      ")}
      ${chol.join("\n      ")}
      xd = clamp(xd + clamp(fin(z0), -MAX_ACC, MAX_ACC) * dt, -MAX_CART_VEL, MAX_CART_VEL);
      ${I.map((i) => `w${i} = clamp(w${i} + clamp(fin(z${1 + i}), -MAX_ACC, MAX_ACC) * dt, -MAX_ANG_VEL, MAX_ANG_VEL);`).join("\n      ")}
      x = x + xd * dt;
      ${I.map((i) => `th${i} = th${i} + w${i} * dt; sn${i} = sin(th${i}); cs${i} = cos(th${i});`).join("\n      ")}

      let ang = ${angle()};
      let spin = ${sum(I.map((i) => `w${i} * w${i}`))};
      let near = exp(-ang / nearScale);
      let ax = abs(x);
      let over = max(ax - xSoft, 0.0);
      let dead = select(0.0, 1.0, ax > xDead);
      ${energy("pe")}
      let en = pe - eT;
      acc = acc + (wAngle * ang + (wVel + wVelNear * near) * spin + wX * x * x + wXd * xd * xd
        + wBarrier * over * over + wDead * dead + wEnergy * (1.0 - near) * en * en + wU * u * u) * dt;
      done = done + 1u;
      if (early && dead > 0.0) {
        return acc + wDead * dt * f32(total - done) + duCost(off, T, wDu, kdt);
      }
    }
  }

  let e = array<f32, ${dim}>(x, xd${I.map((i) => `, wrapA(th${i} - tg${i}), w${i}`).join("")});
  var quad: f32 = 0.0;
  for (var i: u32 = 0u; i < ${dim}u; i = i + 1u) {
    var row: f32 = 0.0;
    for (var k: u32 = 0u; k < ${dim}u; k = k + 1u) {
      row = row + prm[${P}u + i * ${dim}u + k] * e[k];
    }
    quad = quad + e[i] * row;
  }
  let angT = ${angle()};
  let nearT = exp(-angT / nearScale);
  ${energy("peT")}
  let enT = peT - eT;
  acc = acc + nearT * wTl * quad + (1.0 - nearT) * wTe * enT * enT;
  return acc + duCost(off, T, wDu, kdt);
}
`;
  return { code: WGSL_COMMON + code + WGSL_MAIN, K, P, S0, LEN };
}
