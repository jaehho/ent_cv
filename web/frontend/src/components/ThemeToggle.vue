<template>
  <button
    class="theme-toggle"
    :title="`Theme: ${themeMode} (click to cycle)`"
    :aria-label="`Theme: ${themeMode}, click to cycle`"
    @click="cycle"
  >
    <Monitor v-if="themeMode === 'system'" :size="14" :stroke-width="2" />
    <Sun v-else-if="themeMode === 'light'" :size="14" :stroke-width="2" />
    <Moon v-else :size="14" :stroke-width="2" />
  </button>
</template>

<script setup>
import { Monitor, Sun, Moon } from "lucide-vue-next";
import { useTheme } from "../composables/useTheme.js";

const { themeMode, cycle } = useTheme();
</script>

<style scoped>
.theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 28px;
  padding: 0;
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-1);
  color: var(--text-dim);
  background: transparent;
  cursor: pointer;
  transition: color .15s, border-color .15s;
}
.theme-toggle:hover {
  color: var(--text);
  border-color: var(--accent-border);
}
</style>
