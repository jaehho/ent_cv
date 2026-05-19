<template>
  <div class="class-panel">
    <div class="class-panel-header">
      <span class="section-label">Classes</span>
      <div class="custom-dropdown">
        <span class="section-label dropdown-prefix">Sort:</span>
        <div class="dropdown-trigger" @click="showSortDropdown = !showSortDropdown">
          <span class="dropdown-value">{{ classSortMode }}</span>
          <span class="dropdown-chevron">▾</span>
        </div>
        <div v-if="showSortDropdown" class="dropdown-menu">
          <div
            v-for="mode in SORT_MODES"
            :key="mode.value"
            class="dropdown-item"
            @click="onSortPick(mode.value)"
          >{{ mode.label }}</div>
        </div>
      </div>
    </div>

    <div
      v-for="(item, displayIdx) in displayedClasses"
      :key="item.idx"
      class="class-row"
      :class="{ 'class-row--dragging': draggingClassIdx === displayIdx }"
      :style="rowStyle(item, displayIdx)"
      :draggable="classSortMode === 'custom'"
      @click="emit('toggle-class', item.idx)"
      @dblclick.stop="emit('dblclick-class', item.idx)"
      @dragstart="emit('drag-start', $event, displayIdx)"
      @dragover.prevent="emit('drag-over', $event, displayIdx)"
      @drop="emit('drop', $event, displayIdx)"
      @dragend="emit('drag-end')"
    >
      <div v-if="classSortMode === 'custom'" class="drag-handle" @mousedown.stop>⋮⋮</div>

      <div
        class="class-dot"
        :style="{
          background: classStats[item.idx]?.count === 0
            ? 'var(--border-strong)'
            : CLASS_COLORS[item.idx % CLASS_COLORS.length]
        }"
      />

      <div class="class-body">
        <div class="class-name">{{ item.cls }}</div>
        <div class="class-stat-row">
          <div v-if="classStats[item.idx]?.count === 0" class="class-stat class-stat--empty">
            no detections
          </div>
          <div v-else-if="classStats[item.idx]" class="class-stat">
            {{ classStats[item.idx].pct.toFixed(0) }}% frames
          </div>
          <div
            v-if="filteredSummary?.onset_count?.[item.cls] != null"
            class="class-extra class-extra--onset"
          >
            {{ filteredSummary.onset_count[item.cls] }}× onset
          </div>
          <div
            v-if="filterMode === 'filtered' && filteredSummary?.class_time_sec?.[item.cls] != null"
            class="class-extra class-extra--time"
          >
            {{ filteredSummary.class_time_sec[item.cls].toFixed(1) }}s
          </div>
        </div>

        <div v-if="enabledClasses.has(item.idx) && classStats[item.idx]" class="sparkline-wrap">
          <svg width="100%" height="24" viewBox="0 0 40 24" preserveAspectRatio="none">
            <rect
              v-for="(p, i) in sparklines[item.idx]"
              :key="i"
              :x="i * 2"
              :y="24 - p * 24"
              :height="p * 24"
              width="1.5"
              :fill="CLASS_COLORS[item.idx % CLASS_COLORS.length]"
              opacity="0.7"
            />
          </svg>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import { CLASS_COLORS } from "../utils/index.js";

const SORT_MODES = [
  { value: "default",      label: "Default" },
  { value: "frequency",    label: "Frequency" },
  { value: "alphabetical", label: "Alphabetical" },
  { value: "custom",       label: "Custom" },
];

const props = defineProps({
  displayedClasses:  { type: Array,   required: true },
  enabledClasses:    { type: Object,  required: true },  // Set<number>
  classStats:        { type: Array,   required: true },
  sparklines:        { type: Object,  required: true },
  filterMode:        { type: String,  default: "raw" },
  filteredSummary:   { type: Object,  default: null },
  classSortMode:     { type: String,  required: true },
  draggingClassIdx:  { type: Number,  default: null },
});
const emit = defineEmits([
  "update:classSortMode",
  "toggle-class",
  "dblclick-class",
  "drag-start",
  "drag-over",
  "drop",
  "drag-end",
]);

const showSortDropdown = ref(false);

function onSortPick(mode) {
  emit("update:classSortMode", mode);
  showSortDropdown.value = false;
}

function rowStyle(item) {
  return {
    background: props.enabledClasses.has(item.idx) ? "var(--accent-soft)" : "transparent",
    opacity:    props.classStats[item.idx]?.count === 0
                  ? 0.25
                  : (props.enabledClasses.has(item.idx) ? 1 : 0.35),
    cursor:     props.classSortMode === "custom" ? "move" : "pointer",
    pointerEvents: props.classStats[item.idx]?.count === 0 ? "none" : undefined,
  };
}

function handleClickOutside(e) {
  if (showSortDropdown.value && !e.target.closest(".custom-dropdown")) {
    showSortDropdown.value = false;
  }
}

onMounted(() => window.addEventListener("click", handleClickOutside));
onUnmounted(() => window.removeEventListener("click", handleClickOutside));
</script>

<style scoped>
.class-panel {
  flex: 1; min-height: 0; overflow-y: auto; padding: 10px 14px;
}
.class-panel-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 10px;
}
.section-label {
  font-size: 12px; color: var(--text-faint); letter-spacing: 2px;
  text-transform: uppercase; display: block; margin: 0;
}
.dropdown-prefix { margin-right: 4px; }

.custom-dropdown { position: relative; display: inline-flex; align-items: center; }
.dropdown-trigger {
  display: inline-flex; align-items: center; cursor: pointer; user-select: none;
}
.dropdown-value {
  font-size: 12px; font-family: var(--font-mono);
  letter-spacing: 2px; text-transform: uppercase;
  color: var(--text-faint); transition: color .15s;
}
.dropdown-trigger:hover .dropdown-value { color: var(--text-dim); }
.dropdown-chevron { margin-left: 4px; font-size: 10px; color: var(--text-faint); }
.dropdown-menu {
  position: absolute; top: 100%; right: 0; margin-top: 4px;
  background: var(--bg-2); border: 1px solid var(--border-strong);
  border-radius: 4px; padding: 4px 0; z-index: 1000; min-width: 120px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.dropdown-item {
  padding: 6px 12px; font-size: 12px; font-family: var(--font-mono);
  color: var(--text-dim); cursor: pointer; white-space: nowrap;
  transition: background .15s;
}
.dropdown-item:hover { background: var(--border); color: var(--text); }

.class-row {
  display: flex; align-items: center; gap: 8px; padding: 7px 8px;
  margin-bottom: 2px; border-radius: 6px;
  transition: all 0.15s;
}
.class-row--dragging { opacity: 0.5; }
.drag-handle {
  margin-right: 6px;
  color: var(--text-faint); cursor: grab; user-select: none;
}
.class-dot {
  width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0;
}
.class-body { flex: 1; min-width: 0; }
.class-name {
  font-size: 14px; font-weight: 500;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.class-stat-row {
  display: flex; gap: 6px; align-items: baseline; flex-wrap: wrap;
}
.class-stat { font-size: 11px; color: var(--text-faint); margin-top: 1px; }
.class-stat--empty { font-style: italic; }
.class-extra { font-size: 11px; }
.class-extra--onset { color: var(--text-dim); }
.class-extra--time  { color: var(--accent); }

.sparkline-wrap { width: 100%; height: 24px; margin-top: 6px; }
</style>
