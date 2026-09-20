const STORAGE = "theme";
const DARK_MARK = "*";
const LIGHT_MARK = "o";

export function bindTheme(page, button) {
  const dark = window.localStorage.getItem(STORAGE) === "dark";
  apply(page, button, dark);
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    apply(page, button, !page.classList.contains("is-dark"));
  });
}

function apply(page, button, dark) {
  page.classList.toggle("is-dark", dark);
  document.documentElement.classList.toggle("is-dark", dark);
  button.textContent = dark ? LIGHT_MARK : DARK_MARK;
  button.setAttribute("aria-pressed", dark ? "true" : "false");
  button.setAttribute("aria-label", dark ? "light mode" : "dark mode");
  try {
    window.localStorage.setItem(STORAGE, dark ? "dark" : "light");
  } catch {
    /* ignore quota / private mode */
  }
}
