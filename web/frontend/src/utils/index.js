export const CLASS_COLORS = [
  "#ff6b6b", "#4ecdc4", "#45b7d1", "#f7dc6f", "#bb8fce",
  "#82e0aa", "#f0b27a", "#85c1e9", "#f1948a",
];

export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 100);
  return `${m}:${String(s).padStart(2, "0")}.${String(ms).padStart(2, "0")}`;
}

