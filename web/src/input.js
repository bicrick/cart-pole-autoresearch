const HIT = {
  cart: 0.14,
  lower: 0.1,
  upper: 0.1,
};

export function createInput(canvas, camera) {
  const pointer = {
    active: false,
    id: null,
    body: null,
    world: { x: 0, y: 0 },
  };

  function eventToWorld(event) {
    const rect = canvas.getBoundingClientRect();
    const sx = ((event.clientX - rect.left) / rect.width) * canvas.width;
    const sy = ((event.clientY - rect.top) / rect.height) * canvas.height;
    return camera.screenToWorld(sx, sy);
  }

  function pickBody(world, tips) {
    const dCart = Math.hypot(world.x - tips.cart.x, world.y - tips.cart.y);
    const dLow = Math.hypot(world.x - tips.lower.x, world.y - tips.lower.y);
    const dUp = Math.hypot(world.x - tips.upper.x, world.y - tips.upper.y);
    const hits = [
      { body: "cart", d: dCart, r: HIT.cart },
      { body: "lower", d: dLow, r: HIT.lower },
      { body: "upper", d: dUp, r: HIT.upper },
    ].filter((h) => h.d <= h.r);
    if (!hits.length) {
      // Closest body within a generous grab radius so a thumb still works.
      const nearest = [
        { body: "cart", d: dCart },
        { body: "lower", d: dLow },
        { body: "upper", d: dUp },
      ].sort((a, b) => a.d - b.d)[0];
      return nearest.d < 0.28 ? nearest.body : null;
    }
    hits.sort((a, b) => a.d - b.d);
    return hits[0].body;
  }

  function onDown(event) {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    pointer.active = true;
    pointer.id = event.pointerId;
    pointer.world = eventToWorld(event);
    pointer.body = pickBody(pointer.world, camera.tips);
  }

  function onMove(event) {
    if (!pointer.active || event.pointerId !== pointer.id) return;
    pointer.world = eventToWorld(event);
  }

  function onUp(event) {
    if (event.pointerId !== pointer.id) return;
    pointer.active = false;
    pointer.id = null;
    pointer.body = null;
  }

  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  return {
    pointer,
    destroy() {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
    },
  };
}
