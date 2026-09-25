// Canvas colors for the light and dark palettes in style.css (html.is-dark).
const LIGHT = {
  ink: "#21232d",
  body: "#2a2d38",
  muted: "#8b8f99",
  mutedBody: "#9aa0aa",
  link: "#2563eb",
  rail: "#e8e8e6",
  ghost: "rgba(90, 94, 107, 0.38)",
};
const DARK = {
  ink: "#ecece6",
  body: "#c6c7cc",
  muted: "#6d717c",
  mutedBody: "#5d6069",
  link: "#6b9eff",
  rail: "#2c2e36",
  ghost: "rgba(139, 142, 151, 0.4)",
};

function palette() {
  return document.documentElement.classList.contains("is-dark") ? DARK : LIGHT;
}

export function createCamera() {
  const camera = {
    x: 0,
    scale: 180,
    railY: 0.56,
    tips: { cart: { x: 0, y: 0 }, poles: [] },
    worldToScreen(wx, wy, canvas) {
      return {
        x: canvas.width * 0.5 + (wx - camera.x) * camera.scale,
        y: canvas.height * camera.railY - wy * camera.scale,
      };
    },
    screenToWorld(sx, sy) {
      const canvas = camera.canvas;
      return {
        x: camera.x + (sx - canvas.width * 0.5) / camera.scale,
        y: (canvas.height * camera.railY - sy) / camera.scale,
      };
    },
  };
  return camera;
}

function sizeCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const w = Math.max(1, Math.floor(cssW * dpr));
  const h = Math.max(1, Math.floor(cssH * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return dpr;
}

function polesOf(tips) {
  if (tips?.poles?.length) return tips.poles;
  return [tips?.lower, tips?.mid, tips?.upper, tips?.tip].filter(Boolean);
}

function poleReach(constants) {
  let reach = (constants.poleLength1 ?? 0.5) + (constants.poleLength2 ?? 0.5);
  for (let i = 3; constants[`poleLength${i}`] != null; i += 1) reach += constants[`poleLength${i}`];
  return reach;
}

function drawTrack(ctx, camera, canvas, constants, dpr) {
  const track = constants.trackLimit ?? 2.4;
  const left = camera.worldToScreen(-track, 0, canvas);
  const right = camera.worldToScreen(track, 0, canvas);
  ctx.strokeStyle = palette().rail;
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  ctx.moveTo(left.x, left.y);
  ctx.lineTo(right.x, right.y);
  ctx.stroke();
  if (!constants.trackWalls) return;
  const post = 18 * dpr;
  ctx.strokeStyle = palette().ink;
  ctx.lineWidth = 3 * dpr;
  ctx.beginPath();
  ctx.moveTo(left.x, left.y - post);
  ctx.lineTo(left.x, left.y + post);
  ctx.moveTo(right.x, right.y - post);
  ctx.lineTo(right.x, right.y + post);
  ctx.stroke();
}

function drawGhost(ctx, camera, canvas, state, goalId, constants, dpr, ghostTips) {
  if (typeof ghostTips !== "function") return;
  const ghost = ghostTips(state.x, goalId, constants);
  const joints = polesOf(ghost);
  const g0 = camera.worldToScreen(ghost.cart.x, 0, canvas);
  ctx.strokeStyle = palette().ghost;
  ctx.lineWidth = 3 * dpr;
  ctx.lineCap = "butt";
  ctx.lineJoin = "bevel";
  ctx.beginPath();
  ctx.moveTo(g0.x, g0.y);
  for (const joint of joints) {
    const p = camera.worldToScreen(joint.x, joint.y, canvas);
    ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.fillStyle = palette().ghost;
  joints.forEach((joint, i) => {
    const p = camera.worldToScreen(joint.x, joint.y, canvas);
    square(ctx, p.x, p.y, (4 + i) * dpr);
  });
}

/** Filled square centred on (x, y) with half-side h: joints and wheels are square. */
function square(ctx, x, y, h) {
  ctx.fillRect(x - h, y - h, 2 * h, 2 * h);
}

export function draw(canvas, ctx, state, tips, camera, pointer, constants, goalId, policyOn = true, visual = null, ghostTips = null) {
  const dpr = sizeCanvas(canvas);
  camera.canvas = canvas;
  const track = constants.trackLimit ?? 2.4;
  const reach = poleReach(constants);
  const tip = 0.08;
  const pad = 0.16;
  // Landscape shows the whole track. Portrait (phones) zooms to the pendulum
  // and follows the cart, never panning past the track ends.
  const full = track + reach * 0.45;
  const portrait = canvas.height > canvas.width * 1.1;
  const tight = portrait;
  camera.railY = portrait ? 0.48 : 0.56;
  let halfSpan = tight ? Math.min(full, reach + 0.2) : full;
  if (camera.embed) {
    // Fixed wide shot: the whole track stays in frame and the cart moves through it.
    camera.railY = 0.5;
    camera.x = 0;
    const margin = 0.38;
    const scaleX = canvas.width / (2 * (track + 0.55));
    const scaleY = (canvas.height * 0.5) / (reach + margin);
    camera.scale = Math.min(scaleX, scaleY);
  } else {
    const room = Math.max(0, track + 0.3 - halfSpan);
    const target = Math.max(-room, Math.min(room, tips.cart.x));
    camera.x = room > 0 ? camera.x + (target - camera.x) * 0.15 : 0;
    const scaleX = canvas.width / (2 * halfSpan);
    const up = (camera.railY * canvas.height) / (reach + tip + pad);
    const down = ((1 - camera.railY) * canvas.height) / (reach + tip + pad);
    camera.scale = Math.min(scaleX, up, down);
  }
  camera.tips = tips;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  drawTrack(ctx, camera, canvas, constants, dpr);
  if (policyOn && !visual?.muted) drawGhost(ctx, camera, canvas, state, goalId, constants, dpr, ghostTips);

  const lift = visual?.y ?? 0;
  const cart = camera.worldToScreen(tips.cart.x, lift, canvas);
  const joints = polesOf(tips).map((joint) => ({
    ...joint,
    screen: camera.worldToScreen(joint.x, joint.y + lift, canvas),
  }));
  const pal = palette();
  const ink = policyOn && !visual?.muted ? pal.ink : pal.muted;
  const body = policyOn && !visual?.muted ? pal.body : pal.mutedBody;
  ctx.save();
  ctx.globalAlpha = visual?.alpha ?? 1;

  ctx.strokeStyle = ink;
  ctx.lineWidth = 5 * dpr;
  ctx.lineCap = "butt";
  ctx.lineJoin = "bevel";
  ctx.beginPath();
  ctx.moveTo(cart.x, cart.y);
  for (const joint of joints) ctx.lineTo(joint.screen.x, joint.screen.y);
  ctx.stroke();

  const cartW = 0.28 * camera.scale;
  const cartH = 0.12 * camera.scale;
  const cartX = cart.x - cartW * 0.5;
  const cartY = cart.y - cartH * 0.35;
  const wheelR = 0.03 * camera.scale;
  ctx.fillStyle = body;
  ctx.fillRect(cartX, cartY, cartW, cartH);
  ctx.fillStyle = ink;
  square(ctx, cart.x - cartW * 0.28, cart.y + cartH * 0.55, wheelR);
  square(ctx, cart.x + cartW * 0.28, cart.y + cartH * 0.55, wheelR);

  joints.forEach((joint, i) => {
    const h = (0.038 + i * 0.008) * camera.scale;
    ctx.fillStyle = pointer.body === joint.body ? pal.link : ink;
    square(ctx, joint.screen.x, joint.screen.y, h);
  });

  if (pointer.body === "cart" && !visual?.muted) {
    ctx.strokeStyle = pal.link;
    ctx.lineWidth = 2.5 * dpr;
    ctx.strokeRect(cart.x - cartW * 0.5, cart.y - cartH * 0.35, cartW, cartH);
  }

  if (pointer.active && pointer.body && !visual?.muted) {
    const grab = camera.worldToScreen(pointer.world.x, pointer.world.y, canvas);
    ctx.strokeStyle = pal.link;
    ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath();
    const from = pointer.body === "cart" ? cart : joints.find((j) => j.body === pointer.body)?.screen ?? cart;
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(grab.x, grab.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}
