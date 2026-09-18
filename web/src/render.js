import { ghostTips } from "./goals.js";

const PAPER = "#fbfbfa";
const INK = "#21232d";
const INK_SOFT = "#2a2d38";
const LINK = "#2563eb";
const RAIL = "#e8e8e6";
const GHOST = "rgba(90, 94, 107, 0.38)";

export function createCamera() {
  const camera = {
    x: 0,
    scale: 180,
    tips: { cart: { x: 0, y: 0 }, lower: { x: 0, y: 0 }, upper: { x: 0, y: 0 } },
    worldToScreen(wx, wy, canvas) {
      return {
        x: canvas.width * 0.5 + (wx - camera.x) * camera.scale,
        y: canvas.height * 0.62 - wy * camera.scale,
      };
    },
    screenToWorld(sx, sy) {
      const canvas = camera.canvas;
      return {
        x: camera.x + (sx - canvas.width * 0.5) / camera.scale,
        y: (canvas.height * 0.62 - sy) / camera.scale,
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
  ctx.strokeStyle = "rgba(33, 35, 45, 0.35)";
  ctx.lineWidth = 2.5 * dpr;
  ctx.beginPath();
  ctx.moveTo(left.x, left.y - 12 * dpr);
  ctx.lineTo(left.x, left.y + 12 * dpr);
  ctx.moveTo(right.x, right.y - 12 * dpr);
  ctx.lineTo(right.x, right.y + 12 * dpr);
  ctx.stroke();
}

function drawGhost(ctx, camera, canvas, state, goalId, constants, dpr) {
  const ghost = ghostTips(state.x, goalId, constants);
  const g0 = camera.worldToScreen(ghost.cart.x, 0, canvas);
  const g1 = camera.worldToScreen(ghost.lower.x, ghost.lower.y, canvas);
  const g2 = camera.worldToScreen(ghost.upper.x, ghost.upper.y, canvas);
  ctx.strokeStyle = GHOST;
  ctx.lineWidth = 3 * dpr;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(g0.x, g0.y);
  ctx.lineTo(g1.x, g1.y);
  ctx.lineTo(g2.x, g2.y);
  ctx.stroke();
  ctx.fillStyle = GHOST;
  ctx.beginPath();
  ctx.arc(g1.x, g1.y, 4 * dpr, 0, Math.PI * 2);
  ctx.arc(g2.x, g2.y, 5 * dpr, 0, Math.PI * 2);
  ctx.fill();
}

export function draw(canvas, ctx, state, tips, camera, pointer, constants, goalId) {
  const dpr = sizeCanvas(canvas);
  camera.canvas = canvas;
  camera.x = 0;
  const track = constants.trackLimit ?? 2.4;
  const reach = (constants.poleLength1 ?? 0.5) + (constants.poleLength2 ?? 0.5);
  const halfSpan = track + reach * 0.55;
  const scaleX = canvas.width / (2 * halfSpan);
  const scaleY = canvas.height / 2.7;
  camera.scale = Math.min(scaleX, scaleY);
  camera.tips = tips;

  ctx.fillStyle = PAPER;
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  drawTrack(ctx, camera, canvas, constants, dpr);
  drawGhost(ctx, camera, canvas, state, goalId, constants, dpr);

  const cart = camera.worldToScreen(tips.cart.x, 0, canvas);
  const p1 = camera.worldToScreen(tips.lower.x, tips.lower.y, canvas);
  const p2 = camera.worldToScreen(tips.upper.x, tips.upper.y, canvas);

  ctx.strokeStyle = INK;
  ctx.lineWidth = 5 * dpr;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(cart.x, cart.y);
  ctx.lineTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.stroke();

  const cartW = 0.28 * camera.scale;
  const cartH = 0.12 * camera.scale;
  ctx.fillStyle = INK_SOFT;
  ctx.fillRect(cart.x - cartW * 0.5, cart.y - cartH * 0.35, cartW, cartH);
  ctx.beginPath();
  ctx.fillStyle = INK;
  ctx.arc(cart.x - cartW * 0.28, cart.y + cartH * 0.55, 0.03 * camera.scale, 0, Math.PI * 2);
  ctx.arc(cart.x + cartW * 0.28, cart.y + cartH * 0.55, 0.03 * camera.scale, 0, Math.PI * 2);
  ctx.fill();

  const r1 = 0.045 * camera.scale;
  const r2 = 0.055 * camera.scale;
  ctx.fillStyle = pointer.body === "lower" ? LINK : INK;
  ctx.beginPath();
  ctx.arc(p1.x, p1.y, r1, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = pointer.body === "upper" ? LINK : INK;
  ctx.beginPath();
  ctx.arc(p2.x, p2.y, r2, 0, Math.PI * 2);
  ctx.fill();
  if (pointer.body === "cart") {
    ctx.strokeStyle = LINK;
    ctx.lineWidth = 2.5 * dpr;
    ctx.strokeRect(cart.x - cartW * 0.5, cart.y - cartH * 0.35, cartW, cartH);
  }

  if (pointer.active && pointer.body) {
    const grab = camera.worldToScreen(pointer.world.x, pointer.world.y, canvas);
    ctx.strokeStyle = LINK;
    ctx.lineWidth = 1.5 * dpr;
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath();
    const from = pointer.body === "cart" ? cart : pointer.body === "lower" ? p1 : p2;
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(grab.x, grab.y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}
