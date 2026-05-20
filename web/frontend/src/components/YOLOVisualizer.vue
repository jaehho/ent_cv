<template>
  <!-- Loading placeholder — before API response (no data yet) -->
  <div v-if="isLoading && !dataReady" class="loading-screen">
    <div class="loading-spinner"></div>
    <p class="loading-label">Loading...</p>
  </div>

  <!-- ── Main Interface ─────────────────────────────────────────────────── -->
  <!-- Rendered as soon as data is ready so the video element exists in the DOM.
       The loadedmetadata listener (which sets videoReady) requires videoRef to be
       mounted — hiding the whole viewer with v-if prevents that listener from firing. -->
  <div v-if="data" class="app-root">

    <!-- Loading overlay — covers viewer while video initializes after API responds -->
    <div v-if="isLoading" class="loading-screen loading-screen--overlay">
      <div class="loading-spinner"></div>
      <p class="loading-label">Loading...</p>
    </div>

    <AppHeader
      :case-id="props.id"
      :username="username"
      :filter-mode="filterMode"
      v-model:video-mode="videoMode"
      :has-prediction-frames="!!data?.has_prediction_frames"
      @back="goBack"
      @logout="emit('logout')"
    />

    <div class="body-row">

      <!-- ── Left Panel ────────────────────────────────────────────────── -->
      <div class="left-panel" :style="{ width: leftPanelWidth + 'px' }">

        <PlayerControls
          :is-playing="isPlaying"
          :playback-rate="playbackRate"
          :current-frame="currentFrame"
          :current-time="currentTime"
          :fps="data?.fps"
          :total-frames="data?.total_frames ?? 0"
          v-model:jump-filter="jumpFilter"
          :jump-filter-class-ids="jumpFilterClassIds"
          :jump-frame-count="jumpFrames?.length ?? null"
          :changed-frames-available="!!changedFrames"
          :displayed-classes="displayedClasses"
          @toggle-play="togglePlay"
          @set-rate="setRate"
          @seek-prev="seekFiltered(-1)"
          @seek-next="seekFiltered(1)"
          @seek-frame="seekToFrame"
          @toggle-jump-class="toggleJumpClass"
        />

        <!-- View: raw or filtered detections (filter is produced offline by CLI) -->
        <div class="section">
          <div class="section-label">View</div>

          <div class="mode-toggle" style="margin-bottom:12px">
            <button class="mode-btn" :class="{ 'mode-btn--active': filterMode === 'raw' }" style="flex:1" @click="filterMode = 'raw'">Raw</button>
            <button class="mode-btn" :class="{ 'mode-btn--active': filterMode === 'filtered' }" style="flex:1" @click="filterMode = 'filtered'">Filtered</button>
          </div>

          <div v-if="filterMode === 'filtered' && filterInfo" style="font-size:11px;color:var(--text-faint);padding:7px 8px;background:var(--bg-2);border-radius:6px;border:1px solid var(--border)">
            <span style="color:var(--text-faint)">{{ filterInfo.method.replace(/_/g, ' ') }}</span>
            <template v-if="filterInfo.min_duration_sec != null">
              &nbsp;· min {{ filterInfo.min_duration_sec }}s / gap {{ filterInfo.gap_fill_sec }}s
            </template>
            <template v-else-if="filterInfo.window_sec != null">
              &nbsp;· window {{ filterInfo.window_sec }}s / thr {{ filterInfo.vote_threshold }}
            </template>
          </div>
        </div>
      </div>

      <!-- Left resize handle -->
      <div class="resize-handle resize-handle--col" @mousedown="startResize('left', $event)"></div>

      <!-- ── Main Content ───────────────────────────────────────────────── -->
      <div class="main-content">

        <!-- Video / Frame display -->
        <div
          class="video-area"
          :style="{ flex: (videoSrc || videoMode === 'prediction') ? 1 : 0, minHeight: (videoSrc || videoMode === 'prediction') ? '200px' : '100px' }"
        >
          <!-- Prediction mode: show JPEG frames extracted from annotated AVI -->
          <div v-if="videoMode === 'prediction'" class="video-wrapper">
            <img
              v-if="currentPartPredictionFrameUrl"
              :src="currentPartPredictionFrameUrl"
              class="video-el"
            />
            <div v-else style="text-align:center;padding:20px;color:var(--text-faint)">No prediction frame</div>
          </div>
          <!-- Raw mode: video element + canvas overlay -->
          <div v-else-if="videoSrc" class="video-wrapper">
            <video
              ref="videoRef"
              class="video-el"
              playsinline
              preload="auto"
              muted
            />
            <!-- Canvas overlay only shown in raw mode —annotations are baked into prediction videos -->
            <canvas
              v-if="videoMode === 'raw'"
              ref="overlayRef"
              class="overlay-canvas"
            />
          </div>
          <div v-else style="text-align:center;padding:20px">
            <div style="font-size:14px;color:var(--border-strong);margin-bottom:4px">
              No video loaded —
              using frame simulation
            </div>
            <div class="sim-time">{{ formatTime(currentTime) }}</div>
          </div>

          <div v-if="currentPartName" class="part-badge">{{ currentPartName }}</div>

          <div class="det-count-overlay">
            <span style="font-size:24px;font-weight:700;color:var(--accent)">{{ currentDetections.length }}</span>
            <span style="font-size:13px;color:var(--text-faint);margin-left:6px">detections</span>
          </div>

          <div v-if="currentDetections.length > 0" class="det-bar">
            <span
              v-for="(d, i) in currentDetections"
              :key="i"
              class="det-badge"
              :style="{
                '--badge-color': CLASS_COLORS[d.class_id % CLASS_COLORS.length]
              }"
            >
              {{ d.class_name }} <span style="opacity:0.6">{{ d.confidence.toFixed(2) }}</span>
            </span>
          </div>
        </div>

        <!-- Resize handle: video ↔ raster -->
        <div class="resize-handle resize-handle--row" @mousedown="startResize('raster', $event)"></div>

        <!-- Class labels + Raster -->
        <div style="display:flex;border-top:1px solid var(--border);flex-shrink:0">
          <div style="width:120px;flex-shrink:0;background:var(--bg-1);border-right:1px solid var(--border)">
            <div
              v-for="(item, displayIdx) in displayedClasses"
              :key="item.idx"
              class="raster-label"
              :style="{
                height: rasterLabelHeight,
                color: enabledClasses.has(item.idx) ? 'var(--text-dim)' : 'var(--border-strong)',
                cursor: 'grab',
              }"
              draggable="true"
              @dragstart="handleDragStart($event, displayIdx)"
              @dragover.prevent="handleDragOver($event, displayIdx)"
              @drop="handleDrop($event, displayIdx)"
              @dragend="handleDragEnd"
            >
              <div
                class="raster-label-bar"
                :style="{
                  height: rasterLabelBarHeight,
                  background: enabledClasses.has(item.idx) ? CLASS_COLORS[item.idx % CLASS_COLORS.length] : 'var(--bg-hover)'
                }"
              />
              <span style="overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{{ item.cls }}</span>
            </div>
          </div>

          <div style="flex:1;position:relative;overflow:hidden;">
            <canvas
              ref="rasterRef"
              :style="{ 
                width: '100%', 
                height: rasterHeight + 'px', 
                display: 'block', 
                cursor: 'crosshair',
                touchAction: 'none',
                overscrollBehavior: 'none'
              }"
              @click="handleRasterClick"
              @mousemove="handleRasterMouseMove"
              @mousedown="handleRasterMouseDown"
              @mouseleave="hoveredFrame = null"
              @wheel.passive="handleWheel"
            />
            <div v-if="hoveredFrame !== null" :style="{
              position: 'absolute',
              left: hoverX + 'px',
              top: 0,
              bottom: 0,
              width: '1px',
              borderLeft: '1px dashed rgba(255,255,255,0.5)',
              pointerEvents: 'none'
            }"></div>
            <div v-if="hoveredFrame !== null" class="hover-tooltip" :style="{ left: hoverX + 8 + 'px', right: 'auto' }">
              Frame {{ hoveredFrame }} ·
              {{ formatTime(hoveredFrame / data.fps) }}
            </div>
          </div>
        </div>

        <!-- Minimap -->
        <div style="padding:6px 0 6px 120px;border-top:1px solid var(--border);background:var(--bg-1);flex-shrink:0">
          <canvas
            ref="minimapRef"
            style="width:100%;height:24px;display:block;cursor:pointer;border-radius:3px"
            @click="handleMinimapClick"
          />
        </div>

        <!-- Keyboard hints -->
        <div class="kbd-bar">
          <span><kbd class="kbd">Space</kbd> Play/Pause</span>
          <span><kbd class="kbd">←→</kbd> Frame step</span>
          <span><kbd class="kbd">Shift+←→</kbd> 10 frames</span>
          <span><kbd class="kbd">+/-</kbd> Zoom</span>
          <span><kbd class="kbd">0</kbd> Reset zoom</span>
          <span><kbd class="kbd">Scroll</kbd> Zoom at cursor</span>
        </div>
      </div>

      <!-- Resize handle: main ↔ right panel -->
      <div class="resize-handle resize-handle--col" @mousedown="startResize('right', $event)"></div>

      <!-- ── Right Stats Panel ──────────────────────────────────────────── -->
      <div class="right-panel" ref="matrixContainerRef" :style="{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '0', width: rightPanelWidth + 'px' }">

        <ClassPanel
          :displayed-classes="displayedClasses"
          :enabled-classes="enabledClasses"
          :class-stats="classStats"
          :sparklines="sparklines"
          :filter-mode="filterMode"
          :filtered-summary="filteredSummary"
          v-model:class-sort-mode="classSortMode"
          :dragging-class-idx="draggingClassIdx"
          @toggle-class="guardedToggleClass"
          @dblclick-class="handleClassDoubleClick"
          @drag-start="handleDragStart"
          @drag-over="handleDragOver"
          @drop="handleDrop"
          @drag-end="handleDragEnd"
        />

        <!-- Transitions (filtered mode, when summary available) -->
        <div v-if="transitionMatrix"
          style="flex-shrink:0;border-top:1px solid var(--border);padding:10px 14px;max-height:40%;overflow-y:auto">
          <div class="section-label" style="margin-bottom:8px">Transitions</div>
          <div style="display:flex">
            <!-- Row labels -->
            <div :style="{ display:'flex', flexDirection:'column', marginRight:'8px', width: transitionMatrix.rowLabelW + 'px', flexShrink: 0 }">
              <!-- spacer aligns row labels with grid rows (below column label area) -->
              <div style="height:80px;flex-shrink:0"></div>
              <div :style="{ minHeight: transitionMatrix.cellSize + 'px', width: transitionMatrix.rowLabelW + 'px' }" v-for="cls in transitionMatrix.classes" :key="'rl-'+cls"
                style="display:flex;align-items:center;justify-content:flex-end;font-size:9px;color:var(--text-faint);white-space:normal;overflow-wrap:normal;word-break:keep-all;text-align:right;padding:2px 0"
                :title="cls">{{ cls }}</div>
            </div>
            <div style="flex:1;min-width:0">
              <!-- Column labels: rotated 90° upward -->
              <div style="display:flex;align-items:flex-end;height:80px">
                <div v-for="cls in transitionMatrix.classes" :key="'cl-'+cls"
                  :style="{ width: transitionMatrix.cellSize + 'px', flexShrink: 0 }"
                  style="display:flex;align-items:flex-end;justify-content:center;overflow:visible">
                  <span :style="{ writingMode:'vertical-rl', transform:'rotate(180deg)', display:'block', fontSize:'9px', color:'var(--text-faint)', whiteSpace:'normal', overflowWrap:'normal', wordBreak:'keep-all', maxHeight:'78px', overflow:'hidden' }"
                    :title="cls">{{ cls }}</span>
                </div>
              </div>
              <!-- Matrix grid -->
              <div v-for="(row, ri) in transitionMatrix.grid" :key="ri" style="display:flex">
                <div v-for="(cell, ci) in row" :key="ci"
                  :style="{
                    width: transitionMatrix.cellSize + 'px',
                    height: transitionMatrix.cellSize + 'px',
                    background: cell.count > 0
                      ? `rgba(78, 205, 196, ${0.15 + 0.85 * cell.intensity})`
                      : 'var(--bg-2)',
                    border: '1px solid var(--border)',
                    borderRadius: '2px',
                    cursor: cell.count > 0 ? 'default' : undefined,
                    position: 'relative',
                  }"
                  :title="cell.from + ' → ' + cell.to + ': ' + cell.count"
                >
                  <span v-if="cell.count > 0" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:8px;color:#fff;font-weight:600">
                    {{ cell.count }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

      </div>

    </div>
  </div>
</template>

<script setup>
import {
  ref, computed, watch, watchEffect, onMounted, onUnmounted, nextTick,
} from "vue";
import { useRouter } from "vue-router";
import { CLASS_COLORS, formatTime } from "../utils/index.js";
import { useCaseData } from "../composables/useCaseData.js";
import AppHeader from "./AppHeader.vue";
import PlayerControls from "./PlayerControls.vue";
import ClassPanel from "./ClassPanel.vue";

const props = defineProps({
  id: { type: String, required: true },
  username: { type: String, default: "" },
});
const emit = defineEmits(["logout"]);
const router = useRouter();

// ── Constants ──────────────────────────────────────────────────────────────
const SPARKLINE_BINS = 20;

// ── Canvas palette ─────────────────────────────────────────────────────────
// Canvas drawing can't reference CSS custom properties directly, so we mirror
// the relevant tokens here. Dark-only — single source of truth.
const _canvas = {
  bg: "#0a0f17",                       // one notch below --bg-0 — raster reads inset
  marker: "#ffffff",
  label: "#ffffff",
  grid: "rgba(255,255,255,0.05)",
  viewportFill: "rgba(255,255,255,0.06)",
};
// Precomputed RGB values for CLASS_COLORS to avoid repeated hex parsing in hot loops
const CLASS_COLORS_RGB = CLASS_COLORS.map(hex => ({
  r: parseInt(hex.slice(1, 3), 16),
  g: parseInt(hex.slice(3, 5), 16),
  b: parseInt(hex.slice(5, 7), 16),
}));

// ── Case data (composable) ────────────────────────────────────────────────
// data / video readiness / filter view state / per-frame and per-part lookups.
const {
  data, dataReady, videoSrc, videoReady, activeCaseName,
  filterMode, filterInfo, rawFrameSet, filteredSummary, isLoading,
  frameMap, partStartTs, partStartFrame,
  fetchCase, fetchFilteredView,
} = useCaseData();

// ── UI state ──────────────────────────────────────────────────────────────
const currentFrame      = ref(0);
const isPlaying         = ref(false);
const enabledClasses    = ref(new Set());
const zoomLevel         = ref(1);
const panOffset         = ref(0);
const hoveredFrame      = ref(null);
const hoverX = computed(() => {
  if (hoveredFrame.value === null || !data.value || !rasterRef.value) return 0;
  const W = rasterRef.value.clientWidth;
  if (!W) return 0;
  const totalFrames = data.value.total_frames;
  const visibleFraction = 1 / zoomLevel.value;
  const startFrame = panOffset.value * totalFrames;
  const visibleFrames = visibleFraction * totalFrames;
  const pxPerFrame = W / visibleFrames;
  return (hoveredFrame.value - startFrame) * pxPerFrame;
});
const playbackRate      = ref(1);
const isDraggingTimeline = ref(false);
const videoMode         = ref('raw');  // 'raw' | 'prediction'
const classSortMode     = ref('custom');  // 'default' | 'frequency' | 'alphabetical' | 'custom'
const customOrder       = ref([]);  // array of class indices (empty = not initialized)
const draggingClassIdx  = ref(null);  // currently dragging class index

// ── Panel resizing state ──────────────────────────────────────────────────
const leftPanelWidth    = ref(280);
const rightPanelWidth   = ref(280);
const matrixContainerRef   = ref(null)
const matrixContainerWidth = ref(0)
let _matrixResizeObserver  = null
const rasterHeight      = ref(200);   // px height of raster canvas

// ── Jump-filter state ─────────────────────────────────────────────────────
const jumpFilter        = ref('none'); // 'none'|'any'|'class'|'onset'|'changed'
const jumpFilterClassIds = ref(new Set()); // class indices when jumpFilter==='class'

// ── Refs ───────────────────────────────────────────────────────────────────
const videoRef    = ref(null);
const overlayRef  = ref(null);
const rasterRef   = ref(null);
const minimapRef  = ref(null);
const animFrameRef = ref(null);
const isPanningRef = ref(false);
const panStartRef  = ref({ x: 0, offset: 0 });

// RAF-based draw scheduling
let _rafId = null;
let _drawFlags = 0;  // bitmask: 1=overlay, 2=raster, 4=minimap
function scheduleDraws(flags) {
  _drawFlags |= flags;
  if (_rafId) return;
  _rafId = requestAnimationFrame(() => {
    _rafId = null;
    const f = _drawFlags;
    _drawFlags = 0;
    if (f & 1) drawOverlay();
    if (f & 2) drawRaster();
    if (f & 4) drawMinimap();
  });
}

// ── Derived ────────────────────────────────────────────────────────────────
const currentTime = computed(() => {
  const fps = data.value?.fps;
  if (!fps || isNaN(currentFrame.value)) return 0;
  return currentFrame.value / fps;
});

// Pre-sorted parts table — built once per data change, used for O(log n)
// lookup of which part owns the current frame. Five separate currentPart*
// computeds used to each do a fresh O(n) scan of partStartFrame; one shared
// lookup beats that during scrubbing where currentFrame fires many times/sec.
const sortedParts = computed(() => {
  if (!data.value) return [];
  // Server-provided `parts` (flat-format detections) include endFrame too — prefer them.
  if (data.value.parts && data.value.parts.length > 0) {
    return [...data.value.parts].sort((a, b) => a.startFrame - b.startFrame);
  }
  // Fallback: derive from the partStartFrame map (grouped detections).
  const entries = [...partStartFrame.value.entries()]
    .map(([source, startFrame]) => ({ source, startFrame }))
    .sort((a, b) => a.startFrame - b.startFrame);
  return entries.map((p, i) => ({
    source: p.source,
    startFrame: p.startFrame,
    endFrame: i + 1 < entries.length ? entries[i + 1].startFrame - 1 : Infinity,
  }));
});

// Binary search: largest `parts[i].startFrame <= frame`, or null if `frame` < first.
function findPartForFrame(parts, frame) {
  if (parts.length === 0 || frame < parts[0].startFrame) return null;
  let lo = 0, hi = parts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >>> 1;
    if (parts[mid].startFrame <= frame) lo = mid;
    else hi = mid - 1;
  }
  return parts[lo];
}

// Single source of truth for "which part owns currentFrame". Direct frame-map
// hit wins (handles the same global frame appearing in two parts if that ever
// happens); binary search on sortedParts is the universal fallback.
const currentPart = computed(() => {
  if (!data.value || sortedParts.value.length === 0) return null;
  const frame = currentFrame.value;
  const entry = frameMap.value.get(frame);
  let part = entry ? sortedParts.value.find(p => p.source === entry.source) : null;
  if (!part) part = findPartForFrame(sortedParts.value, frame);
  // Treat parts without a source path the same as no part — the old fallback
  // chains all guarded `if (!source) return null` at the end, so this matches
  // that behaviour (and is hit by detection fixtures that omit `source`).
  if (!part || !part.source) return null;
  const partName = part.source.split('/').pop().replace(/\.mp4$/i, '');
  return {
    source: part.source,
    partName,
    startFrame: part.startFrame,
    startTs: partStartTs.value.get(part.source) ?? part.startFrame / (data.value.fps || 1),
  };
});

const currentPartRawUrl = computed(() => {
  const p = currentPart.value;
  return p && activeCaseName.value
    ? `/api/cases/${activeCaseName.value}/raw/${p.partName}.mp4`
    : null;
});

const currentPartPredictionFrameUrl = computed(() => {
  const p = currentPart.value;
  if (!p || !activeCaseName.value) return null;
  // Prediction frames are saved 1-indexed by Ultralytics (clip frame 0 → _1.jpg).
  const localFrame = currentFrame.value - p.startFrame + 1;
  return `/api/cases/${activeCaseName.value}/predictions/${p.partName}_frames/${p.partName}_${localFrame}.jpg`;
});

// Prediction mode shows frames via <img>; null URL suppresses the video watcher.
const currentPartVideoUrl = computed(() =>
  videoMode.value === 'prediction' ? null : currentPartRawUrl.value
);

const currentPartName = computed(() => currentPart.value?.partName ?? null);

const currentPartStartTs = computed(() => currentPart.value?.startTs ?? 0);

// Local seconds within the current part — used to seek the freshly-loaded
// <video> element after a part switch. Returns 0 when the part is unknown
// (the old code returned currentFrame.value, which is frame count, not
// seconds — clamped to video duration in practice).
const currentPartTimestamp = computed(() => {
  const p = currentPart.value;
  const fps = data.value?.fps;
  if (!p || !fps) return 0;
  return (currentFrame.value - p.startFrame) / fps;
});

// Precomputed raster label heights (avoid repeated template expressions)
const rasterLabelHeight = computed(() =>
  data.value ? `${rasterHeight.value / data.value.classes.length}px` : '20px'
);
const rasterLabelBarHeight = computed(() =>
  data.value ? `${rasterHeight.value / data.value.classes.length - 6}px` : '14px'
);

// ── Optimized: single-pass sparse filtered data ────────────────────────────
// Builds a sparse Map of frames with detections (respecting enabled classes only;
// confidence filtering is handled by postprocess.py, not the display layer).
const filteredSparseMap = computed(() => {
  if (!data.value) return null;
  const numClasses = data.value.classes.length;
  const enabled = enabledClasses.value;
  const result = new Map();

  for (const [frameNum, entry] of frameMap.value) {
    let row = null;
    for (const det of entry.detections) {
      if (enabled.has(det.class_id)) {
        if (!row) row = new Float32Array(numClasses);
        if (det.confidence > row[det.class_id]) row[det.class_id] = det.confidence;
      }
    }
    if (row) result.set(frameNum, row);
  }
  return result;
});

// Frames changed by filtering: present in raw but absent in filtered, or vice-versa.
// Only non-null when in filtered mode — drives the change-strip in the raster/minimap.
const changedFrames = computed(() => {
  if (filterMode.value !== 'filtered' || !rawFrameSet.value || !data.value) return null;
  const rawSet = rawFrameSet.value;
  const filtSet = new Set(frameMap.value.keys());
  const changed = new Set();
  for (const f of rawSet) if (!filtSet.has(f)) changed.add(f);
  for (const f of filtSet) if (!rawSet.has(f)) changed.add(f);
  return changed.size > 0 ? changed : null;
});

// Sorted array of frame numbers that match the active jump filter.
// Returns null when filter is off or no frames match.
const jumpFrames = computed(() => {
  if (!data.value || jumpFilter.value === 'none') return null;
  const jf = jumpFilter.value;
  const fsm = filteredSparseMap.value;
  if (!fsm) return null;

  if (jf === 'any') {
    return [...fsm.keys()].sort((a, b) => a - b);
  }

  if (jf === 'class') {
    const ids = jumpFilterClassIds.value;
    if (!ids.size) return null;
    const frames = [];
    for (const [f, row] of fsm) {
      for (const id of ids) {
        if (row[id] > 0) { frames.push(f); break; }
      }
    }
    return frames.sort((a, b) => a - b);
  }

  if (jf === 'onset') {
    // First frame of each contiguous detection run (per class, respecting gaps ≥ 2 frames)
    const sorted = [...fsm.keys()].sort((a, b) => a - b);
    const numClasses = data.value.classes.length;
    const lastSeen = new Int32Array(numClasses).fill(-999);
    const onsets = new Set();
    for (const f of sorted) {
      const row = fsm.get(f);
      for (let c = 0; c < numClasses; c++) {
        if (row[c] > 0) {
          if (lastSeen[c] < f - 1) onsets.add(f);
          lastSeen[c] = f;
        }
      }
    }
    return [...onsets].sort((a, b) => a - b);
  }

  if (jf === 'offset') {
    // Last frame of each contiguous detection run (per class, preceding a gap ≥ 2 or end)
    const sorted = [...fsm.keys()].sort((a, b) => a - b);
    const numClasses = data.value.classes.length;
    const offsets = new Set();
    for (let i = 0; i < sorted.length; i++) {
      const f = sorted[i];
      const nextF = i + 1 < sorted.length ? sorted[i + 1] : Infinity;
      const row = fsm.get(f);
      for (let c = 0; c < numClasses; c++) {
        if (row[c] > 0) {
          const nextRow = nextF === f + 1 ? fsm.get(nextF) : null;
          if (!nextRow || nextRow[c] === 0) {
            offsets.add(f);
            break;
          }
        }
      }
    }
    return [...offsets].sort((a, b) => a - b);
  }

  if (jf === 'changed') {
    if (!changedFrames.value) return null;
    return [...changedFrames.value].sort((a, b) => a - b);
  }

  return null;
});

const currentDetections = computed(() => {
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry) return [];
  return entry.detections.filter(d => enabledClasses.value.has(d.class_id));
});

// ── Optimized: single-pass class stats ─────────────────────────────────────
const classStats = computed(() => {
  if (!data.value) return [];
  const numClasses = data.value.classes.length;
  const enabled = enabledClasses.value;
  const totalFrames = data.value.total_frames || 1;

  const counts = new Uint32Array(numClasses);

  for (const [, entry] of frameMap.value) {
    for (const det of entry.detections) {
      const cid = det.class_id;
      if (enabled.has(cid)) counts[cid]++;
    }
  }

  return data.value.classes.map((name, idx) => ({
    name,
    count: counts[idx],
    pct: counts[idx] / totalFrames * 100,
  }));
});

// Classes list display order — drives both left panel AND raster row order
const displayedClasses = computed(() => {
  if (!data.value) return [];
  const mode = classSortMode.value;
  const classes = data.value.classes;
  
  if (mode === 'frequency') {
    // Sort by descending pct
    return classStats.value
      .map((stats, idx) => ({ cls: stats.name, idx, pct: stats.pct }))
      .sort((a, b) => b.pct - a.pct);
  }
  
  if (mode === 'alphabetical') {
    // Sort alphabetically by class name
    return classes
      .map((cls, idx) => ({ cls, idx }))
      .sort((a, b) => a.cls.localeCompare(b.cls));
  }
  
  if (mode === 'custom' && customOrder.value.length === classes.length) {
    // Use custom order from localStorage
    return customOrder.value.map(idx => ({ cls: classes[idx], idx }));
  }
  
  // Default: preserve detections.json order
  return classes.map((cls, idx) => ({ cls, idx }));
});

// Map from original class index to display row index (for raster drawing)
const classIdxToRowIdx = computed(() => {
  const map = new Map();
  displayedClasses.value.forEach((item, rowIdx) => {
    map.set(item.idx, rowIdx);
  });
  return map;
});

// ── Optimized: memoized sparklines (computed once, not per-class in render) ─
const sparklines = computed(() => {
  if (!data.value || !filteredSparseMap.value) return {};
  const totalFrames = data.value.total_frames;
  const numClasses = data.value.classes.length;
  const binSize = Math.ceil(totalFrames / SPARKLINE_BINS);
  const result = {};

  // Accumulate all classes in one pass over the sparse map
  // counts[classIdx][binIdx]
  const counts = Array.from({ length: numClasses }, () => new Uint16Array(SPARKLINE_BINS));

  for (const [frameNum, row] of filteredSparseMap.value) {
    const bin = Math.min(Math.floor(frameNum / binSize), SPARKLINE_BINS - 1);
    for (let c = 0; c < numClasses; c++) {
      if (row[c] > 0) counts[c][bin]++;
    }
  }

  for (let c = 0; c < numClasses; c++) {
    const points = new Array(SPARKLINE_BINS);
    let maxP = 0;
    for (let b = 0; b < SPARKLINE_BINS; b++) {
      points[b] = counts[c][b] / binSize;
      if (points[b] > maxP) maxP = points[b];
    }
    if (maxP < 0.001) maxP = 0.001;
    for (let b = 0; b < SPARKLINE_BINS; b++) points[b] /= maxP;
    result[c] = points;
  }

  return result;
});

// ── Transition matrix (computed from filteredSummary.transition_matrix) ────
const transitionMatrix = computed(() => {
  const tm = filteredSummary.value?.transition_matrix;
  if (!tm || !data.value) return null;
  // Collect all classes that appear in the matrix
  const classSet = new Set();
  for (const [from, targets] of Object.entries(tm)) {
    classSet.add(from);
    for (const to of Object.keys(targets)) classSet.add(to);
  }
  if (classSet.size === 0) return null;
  // Use display order if available, otherwise alphabetical
  const allClasses = data.value.classes;
  const classes = allClasses.filter(c => classSet.has(c));
  // Find max count for intensity normalization
  let maxCount = 1;
  for (const targets of Object.values(tm)) {
    for (const count of Object.values(targets)) {
      if (count > maxCount) maxCount = count;
    }
  }
  // Build grid (rows = from, cols = to)
  const grid = classes.map(from =>
    classes.map(to => ({
      from, to,
      count: tm[from]?.[to] ?? 0,
      intensity: (tm[from]?.[to] ?? 0) / maxCount,
    }))
  );
  // 28px = 14px section padding each side; 80px = row-label column; 8px = margin
  const ROW_LABEL_W = 80
  const available = matrixContainerWidth.value > 0 ? Math.max(0, matrixContainerWidth.value - 28 - ROW_LABEL_W - 8) : 100
  const cellSize = Math.max(1, Math.floor(available / classes.length))
  return { classes, grid, cellSize, rowLabelW: ROW_LABEL_W };
});

// ── Actions ────────────────────────────────────────────────────────────────
async function loadCase(caseName) {
  try {
    const parsed = await fetchCase(caseName, { isPredictionMode: videoMode.value === "prediction" });

    // UI-state resets that don't belong to case-data: classes, customOrder,
    // playback position, timeline zoom/pan, playback rate.
    enabledClasses.value = new Set(parsed.classes.map((_, i) => i));
    jumpFilterClassIds.value = new Set();
    if (customOrder.value.length !== parsed.classes.length) {
      customOrder.value = parsed.classes.map((_, i) => i);
    }
    currentFrame.value = 0;
    zoomLevel.value = 1;
    panOffset.value = 0;
    playbackRate.value = 1;

    nextTick(() => seekToFrame(0));
  } catch (err) {
    alert(`Failed to load case "${caseName}": ${err.message}`);
  }
}

// Seek to the nearest frame matching the active jump filter (dir: +1 forward, -1 back).
// Falls back to ±1 frame step when no filter is active or no match exists.
function seekFiltered(dir) {
  if (!data.value) return;
  const frames = jumpFrames.value;
  if (!frames || !frames.length) {
    const cf = currentFrame.value;
    seekToFrame(dir > 0
      ? Math.min(data.value.total_frames - 1, cf + 1)
      : Math.max(0, cf - 1));
    return;
  }
  const cf = currentFrame.value;
  if (dir > 0) {
    const idx = frames.findIndex(f => f > cf);
    seekToFrame(idx >= 0 ? frames[idx] : frames[0]);
  } else {
    let idx = -1;
    for (let i = frames.length - 1; i >= 0; i--) {
      if (frames[i] < cf) { idx = i; break; }
    }
    seekToFrame(idx >= 0 ? frames[idx] : frames[frames.length - 1]);
  }
}

function toggleJumpClass(classId) {
  const s = new Set(jumpFilterClassIds.value);
  s.has(classId) ? s.delete(classId) : s.add(classId);
  jumpFilterClassIds.value = s;
}

function goBack() {
  router.push({ name: "cases" });
}

// Sync the <video> element's currentTime to the JS currentFrame ref.
// Each call asks the browser to seek (decode keyframes + intermediate frames),
// which is the bottleneck during a drag — so we skip it while the user is
// actively scrubbing and call it once on mouseup instead.
function syncVideoTime() {
  if (!videoRef.value || !videoSrc.value || !data.value) return;
  const part = currentPart.value;
  if (!part) return;
  videoRef.value.currentTime = (currentFrame.value - part.startFrame) / data.value.fps;
}

function seekToFrame(frame, { syncVideo = true } = {}) {
  if (!data.value) return;
  currentFrame.value = frame;
  scheduleDraws(1);  // overlay always redraws for instant visual feedback
  if (syncVideo) syncVideoTime();
}

function togglePlay() {
  if (videoRef.value) {
    videoRef.value.paused ? videoRef.value.play() : videoRef.value.pause();
  } else {
    isPlaying.value = !isPlaying.value;
  }
}

function setRate(r) {
  playbackRate.value = r;
  if (videoRef.value) videoRef.value.playbackRate = r;
}

function toggleClass(idx) {
  const next = new Set(enabledClasses.value);
  next.has(idx) ? next.delete(idx) : next.add(idx);
  enabledClasses.value = next;
}

function handleClassDoubleClick(idx) {
  if (!data.value) return;
  // If only this class is enabled, restore all classes
  if (enabledClasses.value.size === 1 && enabledClasses.value.has(idx)) {
    enabledClasses.value = new Set(data.value.classes.map((_, i) => i));
  } else {
    // Solo this class (enable only this one)
    enabledClasses.value = new Set([idx]);
  }
}

// ── Drag-to-reorder handlers (custom mode only) ──────────────────────
let _justDragged = false;

function handleDragStart(e, displayIdx) {
  _justDragged = false;
  draggingClassIdx.value = displayIdx;
  e.dataTransfer.effectAllowed = 'move';
}

function handleDragOver(e, displayIdx) {
  if (draggingClassIdx.value === null) return;
  e.dataTransfer.dropEffect = 'move';
}

function handleDrop(e, targetIdx) {
  e.preventDefault();
  if (draggingClassIdx.value === null || draggingClassIdx.value === targetIdx) return;
  _justDragged = true;
  
  // Ensure we have a custom order to reorder
  if (customOrder.value.length !== data.value.classes.length) {
    customOrder.value = data.value.classes.map((_, i) => i);
  }
  // Reorder customOrder array
  const newOrder = [...customOrder.value];
  const [removed] = newOrder.splice(draggingClassIdx.value, 1);
  newOrder.splice(targetIdx, 0, removed);
  customOrder.value = newOrder;
  // Auto-switch to custom sort mode
  if (classSortMode.value !== 'custom') classSortMode.value = 'custom';
}

function handleDragEnd() {
  draggingClassIdx.value = null;
}

function guardedToggleClass(idx) {
  if (_justDragged) { _justDragged = false; return; }
  toggleClass(idx);
}

// ── Panel resize handlers ──────────────────────────────────────────────────
function startResize(target, e) {
  e.preventDefault();
  const startX = e.clientX;
  const startY = e.clientY;
  const startVal = target === 'left' ? leftPanelWidth.value
    : target === 'right' ? rightPanelWidth.value
    : rasterHeight.value;
  const onMove = (ev) => {
    if (target === 'left') {
      leftPanelWidth.value = Math.max(180, Math.min(500, startVal + (ev.clientX - startX)));
    } else if (target === 'right') {
      rightPanelWidth.value = Math.max(180, Math.min(500, startVal - (ev.clientX - startX)));
    } else if (target === 'raster') {
      // Raster is at the bottom; its resize handle is its top edge.
      // Dragging DOWN means the handle moves down → raster shrinks (more space above).
      // Dragging UP means the handle moves up → raster grows.
      rasterHeight.value = Math.max(80, Math.min(600, startVal - (ev.clientY - startY)));
    }
  };
  const onUp = () => {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup', onUp);
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    scheduleDraws(6); // redraw raster + minimap after resize
  };
  document.body.style.cursor = target === 'raster' ? 'row-resize' : 'col-resize';
  document.body.style.userSelect = 'none';
  document.addEventListener('mousemove', onMove);
  document.addEventListener('mouseup', onUp);
}

// ── Canvas helpers ─────────────────────────────────────────────────────────
function drawOverlay() {
  const canvas = overlayRef.value;
  const vid = videoRef.value;
  if (!canvas || !vid) return;
  const nW = vid.videoWidth, nH = vid.videoHeight;
  if (!nW || !nH) return;

  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  canvas.width  = cw * dpr;
  canvas.height = ch * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cw, ch);

  const dets = currentDetections.value;
  if (!dets.length) return;

  const d = data.value;
  const inferW = d.inference_width  || nW;
  const inferH = d.inference_height || nH;

  // Compute object-fit:contain letterbox/pillarbox offset.
  // The canvas covers the full wrapper; video content occupies a centred sub-rect.
  const videoAspect  = nW / nH;
  const canvasAspect = cw / ch;
  let contentW, contentH, offX, offY;
  if (videoAspect > canvasAspect) {
    // Letterbox — black bars top & bottom
    contentW = cw;
    contentH = cw / videoAspect;
    offX = 0;
    offY = (ch - contentH) / 2;
  } else {
    // Pillarbox — black bars left & right
    contentH = ch;
    contentW = ch * videoAspect;
    offX = (cw - contentW) / 2;
    offY = 0;
  }
  const scaleX = contentW / inferW;
  const scaleY = contentH / inferH;

  ctx.lineWidth = 2.5;
  ctx.font = "bold 13px 'JetBrains Mono', monospace";

  for (const det of dets) {
    if (!det.bbox) continue;
    const [x1, y1, x2, y2] = det.bbox;
    const color = CLASS_COLORS[det.class_id % CLASS_COLORS.length];
    const rx = offX + x1 * scaleX;
    const ry = offY + y1 * scaleY;
    const rw = (x2 - x1) * scaleX;
    const rh = (y2 - y1) * scaleY;

    ctx.strokeStyle = color;
    ctx.strokeRect(rx, ry, rw, rh);

    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
    const textW = ctx.measureText(label).width;
    const lh = 19;
    const ly = ry > lh ? ry - lh : ry + lh;
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = color;
    ctx.fillRect(rx - 1, ly - lh + 4, textW + 10, lh);
    ctx.globalAlpha = 1;
    ctx.fillStyle = _canvas.label;
    ctx.fillText(label, rx + 4, ly - 2);
  }
}

function drawRaster() {
  const canvas = rasterRef.value;
  const sparse = filteredSparseMap.value;
  const d = data.value;
  if (!canvas || !sparse || !d) return;

  const ctx = canvas.getContext("2d", { alpha: false }); // Opt: Disable alpha composition if canvas is opaque
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const numClasses = d.classes.length;
  const rowH = H / numClasses;
  const totalFrames = d.total_frames;
  const visibleFraction = 1 / zoomLevel.value;
  const startFrame = Math.max(0, Math.floor(panOffset.value * totalFrames));
  const endFrame = Math.min(Math.ceil((panOffset.value + visibleFraction) * totalFrames), totalFrames);
  const visibleFrames = endFrame - startFrame;
  const pxPerFrame = W / visibleFrames;

  ctx.fillStyle = _canvas.bg;
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = _canvas.grid;
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  for (let i = 0; i < numClasses; i++) {
    ctx.moveTo(0, i * rowH); 
    ctx.lineTo(W, i * rowH); 
  }
  ctx.stroke();

  // Build class index to display row mapping
  const clsToRow = classIdxToRowIdx.value;

  // ⚡ Optimization: Look up sparse coordinates directly instead of looping evaluating entries
  // Moves from O(All Detections) down to bounds-constrained O(Visible Frames) limits.
  for (let f = startFrame; f < endFrame; f++) {
    const row = sparse.get(f);
    if (!row) continue;
    
    const x = (f - startFrame) * pxPerFrame;
    const barW = Math.max(pxPerFrame, 1);
    
    for (let c = 0; c < numClasses; c++) {
      if (row[c] > 0) {
        const displayRow = clsToRow.get(c) ?? c; 
        const alpha = 0.3 + row[c] * 0.7;
        const rgb = CLASS_COLORS_RGB[c % CLASS_COLORS_RGB.length];
        
        ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
        ctx.fillRect(x, displayRow * rowH + 1, barW, rowH - 2);
      }
    }
  }

  // Playhead
  const playheadX = (currentFrame.value - startFrame) * pxPerFrame;
  if (playheadX >= 0 && playheadX <= W) {
    ctx.strokeStyle = _canvas.marker;
    ctx.lineWidth = 2;
    ctx.shadowColor = _canvas.marker;
    ctx.shadowBlur = 6;
    ctx.beginPath(); ctx.moveTo(playheadX, 0); ctx.lineTo(playheadX, H); ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Changed-frame markers (filtered mode) — 2-px strip at raster bottom
  const cf = changedFrames.value;
  if (cf) {
    ctx.fillStyle = 'rgba(255, 210, 50, 0.85)';
    cf.forEach((f) => {
      if (f >= startFrame && f < endFrame) {
        const x = (f - startFrame) * pxPerFrame;
        ctx.fillRect(x, H - 2, Math.max(pxPerFrame, 1), 2);
      }
    });
  }
}

function drawMinimap() {
  const canvas = minimapRef.value;
  const sparse = filteredSparseMap.value;
  const d = data.value;
  if (!canvas || !sparse || !d) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const numClasses = d.classes.length;
  const rowH = H / numClasses;
  const totalFrames = d.total_frames;

  ctx.fillStyle = _canvas.bg;
  ctx.fillRect(0, 0, W, H);

  const pxPerFrame = W / totalFrames;
  const step = Math.max(1, Math.floor(totalFrames / W));

  // Iterate sparse map (much faster than dense loop for large total_frames)
  for (const [f, row] of sparse) {
    // Snap to step grid for consistent appearance
    if (f % step !== 0) continue;
    const x = (f / totalFrames) * W;
    const barW = Math.max(pxPerFrame * step, 1);
    for (let c = 0; c < numClasses; c++) {
      if (row[c] > 0) {
        ctx.fillStyle = CLASS_COLORS[c % CLASS_COLORS.length] + "99";
        ctx.fillRect(x, c * rowH, barW, rowH);
      }
    }
  }

  // Viewport indicator
  const visibleFraction = 1 / zoomLevel.value;
  const vpX = panOffset.value * W;
  const vpW = visibleFraction * W;
  ctx.fillStyle = _canvas.viewportFill;
  ctx.fillRect(vpX, 0, vpW, H);
  ctx.strokeStyle = _canvas.marker;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(vpX, 0, vpW, H);

  // Changed-frame markers (filtered mode) — 2-px strip at minimap top
  const mcf = changedFrames.value;
  if (mcf) {
    ctx.fillStyle = 'rgba(255, 210, 50, 0.9)';
    const pxPerFr = W / totalFrames;
    for (const f of mcf) {
      ctx.fillRect((f / totalFrames) * W, 0, Math.max(pxPerFr, 1), 2);
    }
  }

  // Playhead
  const phX = (currentFrame.value / totalFrames) * W;
  ctx.strokeStyle = "var(--warn)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(phX, 0); ctx.lineTo(phX, H); ctx.stroke();
}

// ── Reactive drawing (batched via RAF) ─────────────────────────────────────
// Also watch overlayRef so that when the canvas is (re-)mounted — e.g. when
// switching back to Raw mode — we redraw immediately instead of waiting for
// the next currentDetections / videoRef change.
watch(
  [currentDetections, videoRef, overlayRef],
  ([, vid]) => {
    if (vid && !vid.videoWidth) {
      vid.addEventListener("loadedmetadata", () => scheduleDraws(1), { once: true });
    } else {
      scheduleDraws(1);
    }
  },
  { flush: "post" }
);

// Persist custom order to localStorage
watch(customOrder, (order) => {
  if (order.length > 0) {
    localStorage.setItem('yolo-visualizer-custom-order', JSON.stringify(order));
  }
}, { deep: true });

// Reload data when filter mode toggles between raw and filtered
watch(filterMode, async (newMode, oldMode) => {
  if (!activeCaseName.value || !data.value) return;
  try {
    await fetchFilteredView(activeCaseName.value, newMode);
  } catch (err) {
    filterMode.value = oldMode;  // revert toggle
    alert(`Could not load ${newMode} detections: ${err.message}`);
  }
});

// Raster (the per-class horizontal strip) carries the playhead, so it needs
// to redraw on every frame change. The viewport box and the bars don't
// change with currentFrame, so the raster is the only dependent here.
watch(
  [filteredSparseMap, changedFrames, () => currentFrame.value, () => zoomLevel.value, () => panOffset.value],
  () => scheduleDraws(2),  // raster
  { flush: "post" }
);

// Minimap shows the whole timeline at low resolution + a viewport indicator;
// it doesn't draw a playhead. Splitting it off the currentFrame dep means
// scrubbing no longer repaints the minimap once per frame.
watch(
  [filteredSparseMap, changedFrames, () => zoomLevel.value, () => panOffset.value],
  () => scheduleDraws(4),  // minimap
  { flush: "post" }
);

// Redraw raster when display order changes
watch(displayedClasses, () => {
  scheduleDraws(2);  // 2=raster
}, { flush: "post" });

// TRANS-01: start ResizeObserver when right-panel ref becomes available (after v-if="data" renders)
watch(matrixContainerRef, (el) => {
  if (el && !_matrixResizeObserver) {
    _matrixResizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        matrixContainerWidth.value = entry.contentRect.width
      }
    })
    _matrixResizeObserver.observe(el)
  }
})

// ── Video part switching ───────────────────────────────────────────────────
// Flag set by the ended handler so the URL watcher knows to auto-play the next part
let _partEndedContinue = false;

// Stop & unload video when entering prediction mode; re-trigger load on return to raw.
watch(videoMode, (mode) => {
  if (mode === 'prediction') {
    if (videoRef.value) videoRef.value.pause();
    videoSrc.value = null;
    isPlaying.value = false;
  }
});

watch(currentPartVideoUrl, (newUrl) => {
  if (!newUrl || newUrl === videoSrc.value) return;
  const wasPlaying = isPlaying.value || _partEndedContinue;
  _partEndedContinue = false;
  if (videoRef.value) videoRef.value.pause();
  videoSrc.value = newUrl;
  nextTick(() => {
    if (!videoRef.value) return;
    const seekTs = currentPartTimestamp.value;
    const seekAndPlay = () => {
      videoRef.value.currentTime = seekTs;
      videoReady.value = true;
      if (wasPlaying) videoRef.value.play();
    };
    videoRef.value.src = newUrl;
    videoRef.value.addEventListener('loadedmetadata', seekAndPlay, { once: true });
  });
});

// ── Video sync ─────────────────────────────────────────────────────────────
watch([videoRef, data], ([videoEl, d], _old, onCleanup) => {
  if (!videoEl || !d) return;
  const useRVFC = typeof videoEl.requestVideoFrameCallback === 'function';
  const sync = (now, metadata) => {
    // Only update currentFrame while actively playing. When paused, seekToFrame()
    // already sets currentFrame directly. Letting RVFC/rAF override it while paused
    // causes FP rounding to snap the counter back, making jumps appear to do nothing.
    if (videoEl.paused) return;
    const mediaTime = metadata ? metadata.mediaTime : videoEl.currentTime;
    const globalTime = mediaTime + currentPartStartTs.value;
    // +0.001 guards against FP rounding (e.g. floor(frame/fps*fps) == frame-1)
    currentFrame.value = Math.min(Math.floor(globalTime * d.fps + 0.001), d.total_frames - 1);
    if (useRVFC) {
      animFrameRef.value = videoEl.requestVideoFrameCallback(sync);
    } else {
      animFrameRef.value = requestAnimationFrame(() => sync());
    }
  };
  const onPlay  = () => {
    isPlaying.value = true;
    if (useRVFC) {
      animFrameRef.value = videoEl.requestVideoFrameCallback(sync);
    } else {
      sync();
    }
  };
  const onPause = () => {
    isPlaying.value = false;
    if (useRVFC) {
      videoEl.cancelVideoFrameCallback(animFrameRef.value);
    } else {
      cancelAnimationFrame(animFrameRef.value);
    }
  };
  // When a part ends, advance currentFrame to the next part's start so
  // currentPartVideoUrl changes and the URL watcher loads/plays the new part.
  const onEnded = () => {
    const starts = [...partStartFrame.value.entries()].sort((a, b) => a[1] - b[1]);
    const nextPart = starts.find(([, sf]) => sf > currentFrame.value);
    if (nextPart) {
      _partEndedContinue = true;
      currentFrame.value = nextPart[1];
    } else {
      isPlaying.value = false;
    }
  };
  // Redraw overlay after the browser finishes seeking (video frame is now visible).
  const onSeeked = () => scheduleDraws(1);

  videoEl.addEventListener("play", onPlay);
  videoEl.addEventListener("pause", onPause);
  videoEl.addEventListener("ended", onEnded);
  videoEl.addEventListener("seeked", onSeeked);

  // Remove listeners when data changes or the watcher stops, preventing stacked
  // handlers that would fire multiple times per ended event and skip multiple parts.
  onCleanup(() => {
    videoEl.removeEventListener("play", onPlay);
    videoEl.removeEventListener("pause", onPause);
    videoEl.removeEventListener("ended", onEnded);
    videoEl.removeEventListener("seeked", onSeeked);
    if (useRVFC) {
      videoEl.cancelVideoFrameCallback(animFrameRef.value);
    } else {
      cancelAnimationFrame(animFrameRef.value);
    }
  });
});

// ── No-video frame simulation ──────────────────────────────────────────────
watchEffect((onCleanup) => {
  if (videoSrc.value || !data.value || !isPlaying.value) return;
  const total = data.value.total_frames;
  const id = setInterval(() => {
    const next = currentFrame.value + 1;
    if (next >= total) { isPlaying.value = false; return; }
    currentFrame.value = next;
  }, 1000 / (data.value.fps * playbackRate.value));
  onCleanup(() => clearInterval(id));
});

// ── Auto-pan to follow playhead ────────────────────────────────────────────
watch([() => currentFrame.value, () => isPlaying.value], () => {
  if (!data.value || !isPlaying.value) return;
  const visibleFraction = 1 / zoomLevel.value;
  const playheadPos = currentFrame.value / data.value.total_frames;
  const viewEnd = panOffset.value + visibleFraction;
  if (playheadPos > viewEnd - 0.02) {
    panOffset.value = Math.min(playheadPos - visibleFraction * 0.8, 1 - visibleFraction);
  }
});

// ── Timeline interaction ───────────────────────────────────────────────────
function getFrameFromMouse(e, canvas) {
  if (!data.value || !canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const visibleFraction = 1 / zoomLevel.value;
  const frame = Math.floor((panOffset.value + x * visibleFraction) * data.value.total_frames);
  return Math.max(0, Math.min(frame, data.value.total_frames - 1));
}

function handleRasterClick(e) {
  const frame = getFrameFromMouse(e, rasterRef.value);
  if (frame !== null) seekToFrame(frame);
}

function handleRasterMouseMove(e) {
  const frame = getFrameFromMouse(e, rasterRef.value);
  hoveredFrame.value = frame;
  // While the user is actively dragging, skip the video seek — only update
  // the JS frame + overlay. The final video.currentTime write happens once
  // in onGlobalMouseUp when the drag ends.
  if (isDraggingTimeline.value && frame !== null) seekToFrame(frame, { syncVideo: false });
}

function handleRasterMouseDown(e) {
  if (e.button === 1) {
    e.preventDefault();
    isPanningRef.value = true;
    panStartRef.value = { x: e.clientX, offset: panOffset.value };
  } else if (e.button === 0) {
    isDraggingTimeline.value = true;
    const frame = getFrameFromMouse(e, rasterRef.value);
    if (frame !== null) seekToFrame(frame);
  }
}

function handleWheel(e) {
  if (!data.value) return;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newZoom = Math.max(1, Math.min(zoomLevel.value * delta, 100));
  const rect = rasterRef.value?.getBoundingClientRect();
  if (rect) {
    const mouseX = (e.clientX - rect.left) / rect.width;
    const oldFrac = 1 / zoomLevel.value;
    const newFrac = 1 / newZoom;
    const mousePos = panOffset.value + mouseX * oldFrac;
    panOffset.value = Math.max(0, Math.min(mousePos - mouseX * newFrac, 1 - newFrac));
  }
  zoomLevel.value = newZoom;
}

function handleMinimapClick(e) {
  if (!data.value || !minimapRef.value) return;
  const rect = minimapRef.value.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;
  const visibleFraction = 1 / zoomLevel.value;
  panOffset.value = Math.max(0, Math.min(x - visibleFraction / 2, 1 - visibleFraction));
}

// ── Global event listeners ─────────────────────────────────────────────────
function onGlobalMouseMove(e) {
  if (isPanningRef.value && rasterRef.value) {
    const rect = rasterRef.value.getBoundingClientRect();
    const dx = (e.clientX - panStartRef.value.x) / rect.width;
    const visibleFraction = 1 / zoomLevel.value;
    const newOffset = panStartRef.value.offset - dx * visibleFraction;
    panOffset.value = Math.max(0, Math.min(newOffset, 1 - visibleFraction));
  }
}

function onGlobalMouseUp() {
  const wasDraggingTimeline = isDraggingTimeline.value;
  isPanningRef.value = false;
  isDraggingTimeline.value = false;
  // Drag finished — now actually seek the video to where the user landed.
  // One decode-and-render instead of dozens during the drag.
  if (wasDraggingTimeline) syncVideoTime();
}

function onKeyDown(e) {
  if (!data.value) return;
  switch (e.key) {
    case " ":
      e.preventDefault();
      togglePlay();
      break;
    case "ArrowLeft":
      e.preventDefault();
      seekToFrame(Math.max(0, currentFrame.value - (e.shiftKey ? 10 : 1)));
      break;
    case "ArrowRight":
      e.preventDefault();
      seekToFrame(Math.min(data.value.total_frames - 1, currentFrame.value + (e.shiftKey ? 10 : 1)));
      break;
    case "+":
    case "=":
      zoomLevel.value = Math.min(zoomLevel.value * 1.3, 100); break;
    case "-":
      zoomLevel.value = Math.max(zoomLevel.value / 1.3, 1); break;
    case "0":
      zoomLevel.value = 1; panOffset.value = 0; break;
  }
}

onMounted(() => {
  window.addEventListener("mousemove", onGlobalMouseMove);
  window.addEventListener("mouseup", onGlobalMouseUp);
  window.addEventListener("keydown", onKeyDown);

  // Load custom order from localStorage
  const saved = localStorage.getItem('yolo-visualizer-custom-order');
  if (saved) {
    try {
      customOrder.value = JSON.parse(saved);
    } catch (e) {
      console.warn('Failed to parse custom order from localStorage:', e);
    }
  }

  loadCase(props.id);

});

onUnmounted(() => {
  window.removeEventListener("mousemove", onGlobalMouseMove);
  window.removeEventListener("mouseup", onGlobalMouseUp);
  window.removeEventListener("keydown", onKeyDown);
  if (_rafId) cancelAnimationFrame(_rafId);
  // Cancel pending video frame callback if active
  if (videoRef.value && typeof videoRef.value.cancelVideoFrameCallback === 'function') {
    videoRef.value.cancelVideoFrameCallback(animFrameRef.value);
  }
  if (_matrixResizeObserver) {
    _matrixResizeObserver.disconnect()
    _matrixResizeObserver = null
  }
});
</script>

<style scoped>
* { box-sizing: border-box; }

/* ── App layout ─────────────────────────────────────────────────────────── */
.app-root {
  height: 100vh; overflow: hidden; background: var(--bg-0);
  font-family: var(--font-mono);
  color: var(--text); display: flex; flex-direction: column;
}
.mode-toggle {
  display: flex; border: 1px solid var(--border-strong); border-radius: 5px; overflow: hidden;
  font-family: var(--font-mono);
}
.mode-btn {
  padding: 6px 14px; border: none; background: transparent;
  color: var(--text-faint); font-size: 12px; cursor: pointer;
  font-family: inherit; transition: all .15s;
}
.mode-btn--active { background: var(--accent-soft); color: var(--accent); }

.body-row { display: flex; flex: 1; overflow: hidden; }

/* ── Left panel ─────────────────────────────────────────────────────────── */
.left-panel {
  border-right: 1px solid var(--border); overflow-y: auto;
  background: var(--bg-1); padding: 16px 14px; flex-shrink: 0;
}
.section { margin-bottom: 20px; }
.section-label {
  font-size: 12px; color: var(--text-faint); letter-spacing: 2px;
  margin-bottom: 10px; text-transform: uppercase; display: block;
}
/* ── Main content ───────────────────────────────────────────────────────── */
.main-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.video-area {
  /* flex / minHeight set inline; just need positioning context + background */
  position: relative; background: #000; overflow: hidden; min-height: 0;
}
.video-wrapper {
  /* Fill the video-area exactly so max-height: 100% on .video-el resolves correctly.
     Inner flex centers the video/img with letterbox/pillarbox in the black area. */
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
}
.video-el {
  display: block;
  max-width: 100%; max-height: 100%;
  width: auto; height: auto;
}
.overlay-canvas {
  /* Also fills the wrapper; drawOverlay accounts for letterbox/pillarbox offset */
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
}
.sim-time {
  font-size: 40px; font-weight: 700; font-variant-numeric: tabular-nums;
  color: var(--text);
}
.part-badge {
  position: absolute; top: 10px; left: 10px; padding: 4px 12px;
  background: rgba(0,0,0,0.7); border-radius: 5px; backdrop-filter: blur(8px);
  border: 1px solid var(--border-strong); font-size: 12px; color: var(--accent);
}
.det-count-overlay {
  position: absolute; top: 10px; right: 10px; padding: 6px 12px;
  background: rgba(0,0,0,0.7); border-radius: 6px; backdrop-filter: blur(8px);
  border: 1px solid var(--border-strong);
}
.det-bar {
  position: absolute; bottom: 0; left: 0; right: 0;
  padding: 6px 12px; background: rgba(0,0,0,0.8);
  display: flex; gap: 8px; flex-wrap: wrap; backdrop-filter: blur(8px);
}
.det-badge {
  font-size: 13px; padding: 3px 10px; border-radius: 4px;
  background: color-mix(in srgb, var(--badge-color) 20%, transparent);
  color: var(--badge-color);
  border: 1px solid color-mix(in srgb, var(--badge-color) 27%, transparent);
}
.raster-label {
  display: flex; align-items: center; padding: 0 8px;
  font-size: 11px; border-bottom: 1px solid var(--bg-hover);
}
.raster-label-bar { width: 4px; border-radius: 2px; margin-right: 6px; flex-shrink: 0; }
.hover-tooltip {
  position: absolute; top: 4px; right: 4px; padding: 5px 12px;
  background: rgba(0,0,0,0.85); border-radius: 4px; font-size: 13px;
  color: var(--text-dim); pointer-events: none; border: 1px solid var(--border-strong);
}
.kbd-bar {
  padding: 7px 16px; border-top: 1px solid var(--border); background: var(--bg-1);
  font-size: 12px; color: var(--border-strong); display: flex; gap: 16px; flex-shrink: 0;
}
.kbd {
  padding: 2px 6px; background: var(--bg-hover); border-radius: 3px;
  border: 1px solid var(--border-strong); font-size: 11px; font-family: inherit;
}

/* ── Right panel ────────────────────────────────────────────────────────── */
.right-panel {
  border-left: 1px solid var(--border); overflow: hidden;
  background: var(--bg-1); flex-shrink: 0; display: flex; flex-direction: column;
}
.resize-handle { flex-shrink: 0; z-index: 10; }
.resize-handle--col {
  width: 5px; cursor: col-resize; background: transparent;
  transition: background 0.15s;
}
.resize-handle--col:hover, .resize-handle--col:active { background: var(--accent-soft); }
.resize-handle--row {
  height: 5px; cursor: row-resize; background: transparent;
  transition: background 0.15s;
}
.resize-handle--row:hover, .resize-handle--row:active { background: var(--accent-soft); }
.loading-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: var(--bg-0);
  gap: 16px;
}
.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid var(--border-strong);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to { transform: rotate(360deg); }
}
.loading-label {
  color: var(--text-faint);
  font-size: 13px;
  letter-spacing: 1px;
}
.loading-screen--overlay {
  position: absolute;
  inset: 0;
  height: 100%;
  z-index: 100;
}
</style>