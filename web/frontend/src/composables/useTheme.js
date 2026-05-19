// Theme switcher — three modes: 'system' (follow OS), 'light', 'dark'.
//
// Manual overrides set data-theme on <html> so tokens.css can resolve.
// Listeners outside Vue's reactivity (e.g. canvas drawing) can subscribe
// to the custom 'theme-changed' event dispatched on window.

import { ref, watch } from "vue";

const STORAGE_KEY = "ent-cv-theme";
const MODES = ["system", "light", "dark"];

const stored = typeof localStorage !== "undefined" ? localStorage.getItem(STORAGE_KEY) : null;
const initial = MODES.includes(stored) ? stored : "system";

const themeMode = ref(initial);

function systemPrefersLight() {
  return typeof window !== "undefined"
    && window.matchMedia?.("(prefers-color-scheme: light)").matches === true;
}

function effectiveTheme() {
  if (themeMode.value === "system") {
    return systemPrefersLight() ? "light" : "dark";
  }
  return themeMode.value;
}

function applyTheme() {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (themeMode.value === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", themeMode.value);
  }
  window.dispatchEvent(new CustomEvent("theme-changed", {
    detail: { mode: themeMode.value, effective: effectiveTheme() },
  }));
}

// Persist and re-apply when the user toggles.
watch(themeMode, (mode) => {
  if (typeof localStorage !== "undefined") {
    if (mode === "system") localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, mode);
  }
  applyTheme();
});

// When in 'system' mode, re-apply if the OS preference changes.
if (typeof window !== "undefined" && window.matchMedia) {
  const mql = window.matchMedia("(prefers-color-scheme: light)");
  mql.addEventListener?.("change", () => {
    if (themeMode.value === "system") applyTheme();
  });
}

// Apply once at module load so the very first paint matches storage.
applyTheme();

export function useTheme() {
  function cycle() {
    const i = MODES.indexOf(themeMode.value);
    themeMode.value = MODES[(i + 1) % MODES.length];
  }
  return { themeMode, cycle, effectiveTheme };
}
