/**
 * Top-left plant carousel: ‹ label › and one square pip per plant. The label
 * slides out and back in the direction of travel; pips jump straight to a
 * plant. Clicking the label steps forward, as the old toggle did.
 *
 *   createPlantSwitch(root, { ids, labelOf, onSelect }) -> { set(id, dir) }
 */
const SLIDE_PX = 12;
const OUT_MS = 130;
const IN_MS = 190;

function reducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

export function createPlantSwitch(root, { ids, labelOf, onSelect }) {
  const label = root.querySelector("[data-plant-label]");
  const pips = root.querySelector("[data-plant-pips]");
  const frame = label.parentElement;
  let current = null;
  let token = 0;

  /** Size the label's frame to the widest label so the › chevron never moves. */
  function fit() {
    const keep = label.textContent;
    let widest = 0;
    for (const id of ids) {
      label.textContent = labelOf(id);
      widest = Math.max(widest, label.getBoundingClientRect().width);
    }
    label.textContent = keep;
    frame.style.minWidth = `${Math.ceil(widest)}px`;
  }
  fit();
  document.fonts?.ready.then(fit);

  const buttons = ids.map((id) => {
    const pip = document.createElement("button");
    pip.type = "button";
    pip.className = "plant-pip";
    pip.dataset.plant = id;
    pip.setAttribute("aria-label", labelOf(id));
    pip.title = labelOf(id);
    pip.addEventListener("click", (event) => {
      event.stopPropagation();
      if (id !== current) onSelect(id, Math.sign(ids.indexOf(id) - ids.indexOf(current)) || 1);
    });
    pips.append(pip);
    return pip;
  });

  const step = (dir) => {
    const n = ids.length;
    onSelect(ids[(ids.indexOf(current) + dir + n) % n], dir);
  };
  root.querySelectorAll("[data-dir]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      step(Number(btn.dataset.dir));
    });
  });
  label.addEventListener("click", (event) => {
    event.stopPropagation();
    step(1);
  });

  async function slide(text, dir) {
    const mine = (token += 1);
    if (reducedMotion() || !label.animate || current === null) {
      label.textContent = text;
      return;
    }
    const off = `translateX(${-dir * SLIDE_PX}px)`;
    const on = `translateX(${dir * SLIDE_PX}px)`;
    try {
      await label.animate([{ transform: "none", opacity: 1 }, { transform: off, opacity: 0 }], {
        duration: OUT_MS,
        easing: "ease-in",
        fill: "forwards",
      }).finished;
    } catch {
      return;
    }
    if (mine !== token) return;
    label.textContent = text;
    label.animate([{ transform: on, opacity: 0 }, { transform: "none", opacity: 1 }], {
      duration: IN_MS,
      easing: "cubic-bezier(0.2, 0.8, 0.2, 1)",
      fill: "forwards",
    });
  }

  return {
    set(id, dir = 1) {
      if (id === current) return;
      slide(labelOf(id), dir);
      current = id;
      buttons.forEach((pip) => {
        const on = pip.dataset.plant === id;
        pip.classList.toggle("is-active", on);
        pip.setAttribute("aria-pressed", on ? "true" : "false");
      });
    },
  };
}
