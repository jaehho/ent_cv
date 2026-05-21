<template>
  <div class="section">
    <div class="section-label">Playback</div>

    <!-- Transport buttons -->
    <div class="transport">
      <button class="btn btn-icon" @click="emit('seek-prev')" aria-label="Previous">
        <SkipBack :size="16" :stroke-width="2" />
      </button>
      <button
        class="btn btn-play btn-icon"
        :class="isPlaying ? 'btn-play--pause' : 'btn-play--go'"
        :aria-label="isPlaying ? 'Pause' : 'Play'"
        @click="emit('toggle-play')"
      >
        <Pause v-if="isPlaying" :size="18" :stroke-width="2" />
        <Play  v-else           :size="18" :stroke-width="2" />
      </button>
      <button class="btn btn-icon" @click="emit('seek-next')" aria-label="Next">
        <SkipForward :size="16" :stroke-width="2" />
      </button>
    </div>

    <!-- Rate -->
    <div class="rates">
      <button
        v-for="r in RATES"
        :key="r"
        class="btn btn-rate"
        :class="{ 'btn-rate--active': playbackRate === r }"
        @click="emit('set-rate', r)"
      >{{ r }}x</button>
    </div>

    <!-- View mode: raw vs filtered detections. Filtered button is disabled
         when no filtered_detections.json was loaded for this case. -->
    <div class="view-block">
      <div class="view-label">View</div>
      <div class="view-toggle">
        <button
          class="mode-btn view-btn"
          :class="{ 'mode-btn--active': viewMode === 'raw' }"
          @click="emit('update:viewMode', 'raw')"
        >Raw</button>
        <button
          class="mode-btn view-btn"
          :class="{ 'mode-btn--active': viewMode === 'filtered' }"
          :disabled="!filteredAvailable"
          :title="filteredAvailable ? 'Show filtered detections (with in-use flags)' : 'No filtered_detections.json on disk — run postprocess for this case'"
          @click="emit('update:viewMode', 'filtered')"
        >Filtered</button>
      </div>
    </div>

    <!-- Jump filter -->
    <div class="jump-block">
      <div class="jump-label">
        Jump
        <span v-if="jumpFilter !== 'none' && jumpFrameCount !== null" class="jump-count jump-count--ok">
          {{ jumpFrameCount }} frames
        </span>
        <span v-else-if="jumpFilter === 'changed' && !changedFramesAvailable" class="jump-count jump-count--empty">
          run filter first
        </span>
      </div>
      <div class="jump-filters">
        <button
          v-for="jf in JUMP_FILTERS"
          :key="jf.value"
          class="mode-btn jump-btn"
          :class="{ 'mode-btn--active': jumpFilter === jf.value }"
          @click="emit('update:jumpFilter', jf.value)"
        >{{ jf.label }}</button>
      </div>

      <!-- Class picker, only when filtering by class -->
      <div v-if="jumpFilter === 'class'" class="class-pills">
        <button
          v-for="item in displayedClasses"
          :key="item.idx"
          class="class-pill"
          :style="pillStyle(item.idx)"
          @click="emit('toggle-jump-class', item.idx)"
        >{{ item.cls }}</button>
      </div>
    </div>
  </div>

  <!-- Time / frame readout -->
  <div class="time-display">
    <div class="time-value">
      <input
        type="text"
        :value="formatTime(currentTime)"
        class="time-input"
        title="Enter time: M:SS, H:MM:SS, or bare number (minutes, e.g. 30 → 30:00, 30.5 → 30:30)"
        @keydown.enter="e => { handleTimeInput(e.target.value); e.target.blur(); }"
        @blur="e => handleTimeInput(e.target.value)"
      />
      <div v-if="timeInputError" class="time-input-error">{{ timeInputError }}</div>
    </div>
    <div class="time-sub">
      Frame
      <input
        type="number"
        :value="currentFrame + 1"
        :min="1"
        :max="totalFrames"
        class="frame-input"
        @keydown.enter="e => handleFrameInput(e.target.value)"
        @blur="e => handleFrameInput(e.target.value)"
      />
      / {{ totalFrames }}
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { Play, Pause, SkipBack, SkipForward } from "lucide-vue-next";
import { CLASS_COLORS, formatTime } from "../utils/index.js";

const RATES = [0.25, 0.5, 1, 2, 4];
const JUMP_FILTERS = [
  { value: "none",    label: "Off" },
  { value: "any",     label: "Any" },
  { value: "class",   label: "Class" },
  { value: "onset",   label: "Onset" },
  { value: "offset",  label: "Offset" },
  { value: "changed", label: "Changed" },
];

const props = defineProps({
  isPlaying:              { type: Boolean, required: true },
  playbackRate:           { type: Number,  required: true },
  currentFrame:           { type: Number,  required: true },
  currentTime:            { type: Number,  required: true },
  fps:                    { type: Number,  default: null },
  totalFrames:            { type: Number,  required: true },
  jumpFilter:             { type: String,  required: true },
  jumpFilterClassIds:     { type: Object,  required: true },   // Set<number>
  jumpFrameCount:         { type: Number,  default: null },
  changedFramesAvailable: { type: Boolean, default: false },
  displayedClasses:       { type: Array,   required: true },
  viewMode:               { type: String,  default: "filtered" },
  filteredAvailable:      { type: Boolean, default: false },
});
const emit = defineEmits([
  "toggle-play",
  "set-rate",
  "seek-prev",
  "seek-next",
  "seek-frame",
  "update:jumpFilter",
  "toggle-jump-class",
  "update:viewMode",
]);

const timeInputError = ref(null);

function handleFrameInput(value) {
  const frameNum = parseInt(value, 10);
  if (isNaN(frameNum) || !props.totalFrames) return;
  const target = Math.max(1, Math.min(props.totalFrames, frameNum)) - 1;
  emit("seek-frame", target);
}

// Supported formats:
//   H:MM:SS or H:MM:SS.ms   → hours:minutes:seconds
//   M:SS or M:SS.ms         → minutes:seconds
//   <number>                → interpreted as minutes (decimals = fractional)
function handleTimeInput(value) {
  if (!props.fps || !props.totalFrames) return;
  const str = String(value).trim();
  if (!str) return;
  let totalSec = NaN;
  const hms = str.match(/^(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?$/);
  if (hms) {
    totalSec = parseInt(hms[1], 10) * 3600 + parseInt(hms[2], 10) * 60 + parseInt(hms[3], 10);
    if (hms[4]) totalSec += parseFloat("0." + hms[4]);
  } else {
    const ms = str.match(/^(\d+):(\d{1,2})(?:\.(\d+))?$/);
    if (ms) {
      totalSec = parseInt(ms[1], 10) * 60 + parseInt(ms[2], 10);
      if (ms[3]) totalSec += parseFloat("0." + ms[3]);
    } else {
      const bare = parseFloat(str);
      if (!isNaN(bare) && /^[\d.]+$/.test(str)) totalSec = bare * 60;
    }
  }
  if (isNaN(totalSec) || totalSec < 0) {
    timeInputError.value = "Invalid format — use M:SS, H:MM:SS, or a number (minutes)";
    setTimeout(() => { timeInputError.value = null; }, 3000);
    return;
  }
  timeInputError.value = null;
  const frame = Math.round(totalSec * props.fps);
  const clamped = Math.max(0, Math.min(props.totalFrames - 1, frame));
  emit("seek-frame", clamped);
}

function pillStyle(idx) {
  const c = CLASS_COLORS[idx % CLASS_COLORS.length];
  const active = props.jumpFilterClassIds.has(idx);
  return {
    border:     active ? `1px solid ${c}` : "1px solid var(--bg-hover)",
    background: active ? `${c}33` : "transparent",
    color:      active ? c : "var(--text-faint)",
  };
}
</script>

<style scoped>
.section { margin-bottom: 20px; }
.section-label {
  font-size: 12px; color: var(--text-faint); letter-spacing: 2px;
  margin-bottom: 10px; text-transform: uppercase; display: block;
}

.transport { display: flex; gap: 6px; margin-bottom: 12px; }
.rates     { display: flex; gap: 4px; }

.btn {
  padding: 8px 14px;
  background: var(--bg-2); border: 1px solid var(--border-strong);
  border-radius: 5px;
  color: var(--text-dim); font-size: 14px; cursor: pointer;
  font-family: var(--font-mono);
}
.btn-icon  { display: inline-flex; align-items: center; justify-content: center; }
.btn-play  { flex: 1; }
.btn-play--pause { background: var(--warn-soft);   border-color: var(--warn-border); }
.btn-play--go    { background: var(--accent-bg);   border-color: var(--accent-border); }
.btn-rate {
  flex: 1; min-width: 0; padding: 8px 4px;
  background: transparent; font-size: 11px;
  white-space: nowrap; text-align: center;
}
.btn-rate--active { background: var(--accent-bg); border-color: var(--accent-border); }

/* View toggle (raw vs filtered detections) */
.view-block { margin-top: 12px; }
.view-label {
  font-size: 11px; color: var(--text-faint);
  letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px;
}
.view-toggle {
  display: flex; gap: 3px;
  border: 1px solid var(--border-strong); border-radius: 4px;
  overflow: hidden;
}
.view-btn {
  flex: 1; padding: 6px 0; font-size: 11px;
  border: none; background: transparent;
  border-radius: 0;
}
.view-btn:disabled {
  color: var(--text-faint); opacity: 0.5; cursor: not-allowed;
}

/* Jump filter */
.jump-block { margin-top: 12px; }
.jump-label {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 11px; color: var(--text-faint);
  letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px;
}
.jump-count {
  font-size: 10px; letter-spacing: 0; text-transform: none;
}
.jump-count--ok    { color: var(--accent); }
.jump-count--empty { color: var(--text-faint); }
.jump-filters { display: flex; flex-wrap: wrap; gap: 3px; }
.mode-btn {
  padding: 6px 14px; border: none; background: transparent;
  color: var(--text-faint); font-size: 12px; cursor: pointer;
  font-family: var(--font-mono); transition: all .15s;
}
.mode-btn--active { background: var(--accent-soft); color: var(--accent); }
.jump-btn {
  padding: 5px 6px; font-size: 10px;
  border: 1px solid var(--border-strong); border-radius: 4px;
}

.class-pills {
  display: flex; flex-wrap: wrap; gap: 4px;
  margin-top: 8px; max-height: 90px; overflow-y: auto; padding: 2px 0;
}
.class-pill {
  padding: 2px 8px; border-radius: 10px; font-size: 10px; cursor: pointer;
}

/* Time / frame display */
.time-display {
  padding: 12px; margin-bottom: 20px;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: 8px; text-align: center;
}
.time-value { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.time-input {
  width: 100%; padding: 0;
  background: transparent;
  border: none; border-bottom: 1px solid transparent;
  color: inherit; font-family: inherit; text-align: center;
  font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums;
  outline: none;
}
.time-input:hover { border-bottom-color: var(--border-strong); }
.time-input:focus { border-bottom-color: var(--accent); }
.time-input-error {
  font-size: 11px; color: var(--warn); margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.time-sub { font-size: 13px; color: var(--text-faint); margin-top: 4px; }
.frame-input {
  width: 60px; padding: 2px 4px;
  background: transparent;
  border: none; border-bottom: 1px solid var(--border-strong);
  color: var(--accent); font-family: monospace; font-size: 14px;
  text-align: center; outline: none;
  -webkit-appearance: none; -moz-appearance: textfield;
}
.frame-input::-webkit-inner-spin-button,
.frame-input::-webkit-outer-spin-button {
  -webkit-appearance: none; margin: 0;
}
.frame-input:focus { border-bottom-color: var(--accent); }
</style>
