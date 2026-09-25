/**
 * Theme toggle, ported from ~/Desktop/personal-website (components/ThemeToggle.js):
 * "#" in light mode, a dot in dark mode; `html.is-dark` swaps the palette in
 * style.css, the choice persists in localStorage, and the browser chrome color
 * follows --paper. Fires "themechange" on window so the canvas can re-read colors.
 */
const STORAGE = "theme";
const DARK_MARK = "#";

function readDark() {
  try {
    return window.localStorage.getItem(STORAGE) === "dark";
  } catch {
    return false;
  }
}

function persist(dark) {
  try {
    window.localStorage.setItem(STORAGE, dark ? "dark" : "light");
  } catch {
    /* ignore quota / private mode */
  }
}

function applyChromeColor() {
  const meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) return;
  const paper = getComputedStyle(document.documentElement).getPropertyValue("--paper").trim();
  if (paper) meta.content = paper;
}

function apply(button, dark) {
  document.documentElement.classList.toggle("is-dark", dark);
  applyChromeColor();
  button.classList.toggle("is-dot", dark);
  button.textContent = dark ? "" : DARK_MARK;
  button.setAttribute("aria-pressed", dark ? "true" : "false");
  button.setAttribute("aria-label", dark ? "light mode" : "dark mode");
  window.dispatchEvent(new Event("themechange"));
}

/** Embed frames take a theme from the query string and do not write localStorage. */
export function applyEmbedTheme(dark) {
  document.documentElement.classList.toggle("is-dark", Boolean(dark));
  applyChromeColor();
  window.dispatchEvent(new Event("themechange"));
}

export function bindTheme(button) {
  let dark = readDark();
  apply(button, dark);
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    dark = !dark;
    apply(button, dark);
    persist(dark);
  });
}
