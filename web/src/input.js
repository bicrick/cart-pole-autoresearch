const HIT = {
  cart: 0.14,
  lower: 0.1,
  mid: 0.1,
  upper: 0.1,
  tip: 0.1,
};

function jointsOf(tips) {
  if (tips?.poles?.length) return tips.poles;
  return [tips?.lower, tips?.mid, tips?.upper, tips?.tip].filter(Boolean);
}

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
    const joints = jointsOf(tips);
    const candidates = [
      { body: "cart", d: Math.hypot(world.x - tips.cart.x, world.y - tips.cart.y), r: HIT.cart },
      ...joints.map((joint) => ({
        body: joint.body,
        d: Math.hypot(world.x - joint.x, world.y - joint.y),
        r: HIT[joint.body] ?? 0.1,
      })),
    ];
    const hits = candidates.filter((h) => h.d <= h.r);
    if (!hits.length) {
      const nearest = candidates.sort((a, b) => a.d - b.d)[0];
      return nearest && nearest.d < 0.28 ? nearest.body : null;
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

export function createKeys({ onGoal, onTogglePolicy, onCycleGoal, getGoals } = {}) {
  const held = new Set();

  function onDown(event) {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const key = event.key;
    if (key === "Tab") {
      event.preventDefault();
      onCycleGoal?.(event.shiftKey);
      return;
    }
    if (key >= "1" && key <= "8") {
      const goals = typeof getGoals === "function" ? getGoals() : ["UU", "UD", "DU", "DD"];
      const goal = goals[Number(key) - 1];
      if (goal) onGoal?.(goal);
      return;
    }
    if (key === "p" || key === "P") {
      onTogglePolicy?.();
      return;
    }
    if (key === "a" || key === "A" || key === "ArrowLeft") held.add("left");
    if (key === "d" || key === "D" || key === "ArrowRight") held.add("right");
  }

  function onUp(event) {
    const key = event.key;
    if (key === "a" || key === "A" || key === "ArrowLeft") held.delete("left");
    if (key === "d" || key === "D" || key === "ArrowRight") held.delete("right");
  }

  window.addEventListener("keydown", onDown);
  window.addEventListener("keyup", onUp);
  window.addEventListener("blur", () => held.clear());

  return {
    manualForce(limit) {
      let force = 0;
      if (held.has("left")) force -= limit;
      if (held.has("right")) force += limit;
      return force;
    },
    destroy() {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    },
  };
}
