const INK = "#21232d";
const INK_SOFT = "#2a2d38";
const MUTED = "#8b8f99";
const MUTED_SOFT = "#9aa0aa";
const LINK = "#2563eb";
const RAIL = "#e8e8e6";
const GHOST = "rgba(90, 94, 107, 0.38)";

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
  return (constants.poleLength1 ?? 0.5) + (constants.poleLength2 ?? 0.5) + (constants.poleLength3 ?? 0);
}

function drawTrack(ctx, camera, canvas, constants, dpr) {
  const track = constants.trackLimit ?? 2.4;
  const left = camera.worldToScreen(-track, 0, canvas);
  const right = camera.worldToScreen(track, 0, canvas);
  ctx.strokeStyle = RAIL;
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  ctx.moveTo(left.x, left.y);
  ctx.lineTo(right.x, right.y);
  ctx.stroke();
  if (!constants.trackWalls) return;
  const post = 18 * dpr;
  ctx.strokeStyle = INK;
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
  ctx.strokeStyle = GHOST;
  ctx.lineWidth = 3 * dpr;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(g0.x, g0.y);
  for (const joint of joints) {
    const p = camera.worldToScreen(joint.x, joint.y, canvas);
    ctx.lineTo(p.x, p.y);
  }
  ctx.stroke();
  ctx.fillStyle = GHOST;
  ctx.beginPath();
  joints.forEach((joint, i) => {
    const p = camera.worldToScreen(joint.x, joint.y, canvas);
    ctx.moveTo(p.x + (4 + i) * dpr, p.y);
    ctx.arc(p.x, p.y, (4 + i) * dpr, 0, Math.PI * 2);
  });
  ctx.fill();
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
  camera.railY = portrait ? 0.48 : 0.56;
  const halfSpan = portrait ? Math.min(full, reach + 0.2) : full;
  const room = Math.max(0, track + 0.3 - halfSpan);
  const target = Math.max(-room, Math.min(room, tips.cart.x));
  camera.x = room > 0 ? camera.x + (target - camera.x) * 0.15 : 0;
  const scaleX = canvas.width / (2 * halfSpan);
  const up = (camera.railY * canvas.height) / (reach + tip + pad);
  const down = ((1 - camera.railY) * canvas.height) / (reach + tip + pad);
  camera.scale = Math.min(scaleX, up, down);
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
  const ink = policyOn && !visual?.muted ? INK : MUTED;
  const body = policyOn && !visual?.muted ? INK_SOFT : MUTED_SOFT;
  ctx.save();
  ctx.globalAlpha = visual?.alpha ?? 1;

  ctx.strokeStyle = ink;
  ctx.lineWidth = 5 * dpr;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
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
  ctx.beginPath();
  ctx.fillStyle = ink;
  ctx.arc(cart.x - cartW * 0.28, cart.y + cartH * 0.55, wheelR, 0, Math.PI * 2);
  ctx.arc(cart.x + cartW * 0.28, cart.y + cartH * 0.55, wheelR, 0, Math.PI * 2);
  ctx.fill();

  joints.forEach((joint, i) => {
    const r = (0.045 + i * 0.01) * camera.scale;
    ctx.fillStyle = pointer.body === joint.body ? LINK : ink;
    ctx.beginPath();
    ctx.arc(joint.screen.x, joint.screen.y, r, 0, Math.PI * 2);
    ctx.fill();
  });

  if (pointer.body === "cart" && !visual?.muted) {
    ctx.strokeStyle = LINK;
    ctx.lineWidth = 2.5 * dpr;
    ctx.strokeRect(cart.x - cartW * 0.5, cart.y - cartH * 0.35, cartW, cartH);
  }

  if (pointer.active && pointer.body && !visual?.muted) {
    const grab = camera.worldToScreen(pointer.world.x, pointer.world.y, canvas);
    ctx.strokeStyle = LINK;
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
