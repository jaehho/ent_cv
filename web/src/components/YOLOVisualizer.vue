<template>
  <!-- ── Case Picker Screen ────────────────────────────────────────────── -->
  <div v-if="showPicker" class="upload-root">
    <div class="upload-center">
      <div class="upload-label">Ultralytics</div>
      <h1 class="upload-title">YOLO Visualizer</h1>
      <p class="upload-subtitle">Select a prediction case to examine</p>

      <div v-if="loadingCases" class="picker-status">Loading cases…</div>
      <div v-else-if="cases.length === 0" class="picker-status">No cases found in <code>/mnt/data/ent_cv/predictions/</code></div>
      <div v-else class="cases-grid">
        <button
          v-for="c in cases"
          :key="c"
          class="case-card"
          @click="loadCase(c)"
        >
          <div class="case-icon">📁</div>
          <div class="case-name">{{ c }}</div>
        </button>
      </div>

    </div>
  </div>

  <!-- ── Main Interface ─────────────────────────────────────────────────── -->
  <div v-else-if="data" class="app-root">

    <!-- Header -->
    <div class="header">
      <div style="display:flex;align-items:center;gap:16px">
        <span class="header-title">YOLO VIZ</span>
        <span class="header-meta">{{ data.total_frames }} frames · {{ data.fps }} fps · {{ data.classes.length }} classes</span>
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <button
          class="hdr-btn"
          :class="{ 'hdr-btn--active': showStats }"
          @click="showStats = !showStats"
        >Stats</button>
        <div class="mode-toggle">
          <button
            class="mode-btn"
            :class="{ 'mode-btn--active': videoMode === 'raw' }"
            @click="videoMode = 'raw'"
          >Raw + Overlay</button>
          <button
            class="mode-btn"
            :class="{ 'mode-btn--active': videoMode === 'prediction' }"
            @click="videoMode = 'prediction'"
          >Prediction Video</button>
        </div>
        <button class="hdr-btn" @click="newSession">← Cases</button>
      </div>
    </div>

    <div class="body-row">

      <!-- ── Left Panel ────────────────────────────────────────────────── -->
      <div class="left-panel">

        <!-- Playback -->
        <div class="section">
          <div class="section-label">Playback</div>
          <div style="display:flex;gap:6px;margin-bottom:12px">
            <button class="btn" @click="seekToFrame(Math.max(0, currentFrame - 1))">◀◀</button>
            <button
              class="btn btn-play"
              :class="isPlaying ? 'btn-play--pause' : 'btn-play--go'"
              @click="togglePlay"
            >{{ isPlaying ? "⏸" : "▶" }}</button>
            <button class="btn" @click="seekToFrame(Math.min(data.total_frames - 1, currentFrame + 1))">▶▶</button>
          </div>
          <div style="display:flex;gap:4px">
            <button
              v-for="r in RATES"
              :key="r"
              class="btn btn-rate"
              :class="{ 'btn-rate--active': playbackRate === r }"
              @click="setRate(r)"
            >{{ r }}x</button>
          </div>
        </div>

        <!-- Time display -->
        <div class="time-display">
          <div class="time-value">
            {{ data.fps === 1 ? formatRealTime(currentFrame) : formatTime(currentTime) }}
          </div>
          <div class="time-sub">Frame {{ currentFrame }} / {{ data.total_frames - 1 }}</div>
        </div>

        <!-- Confidence -->
        <div class="section">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span class="section-label" style="margin-bottom:0">Confidence</span>
            <span style="font-size:14px;color:#4ecdc4;font-weight:600">{{ confidenceThreshold.toFixed(2) }}</span>
          </div>
          <input type="range" min="0" max="1" step="0.01" :value="confidenceThreshold"
            @input="e => confidenceThreshold = parseFloat(e.target.value)"
            style="width:100%;accent-color:#4ecdc4" />
        </div>

        <!-- Zoom -->
        <div class="section">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span class="section-label" style="margin-bottom:0">Zoom</span>
            <span style="font-size:14px;color:#45b7d1;font-weight:600">{{ zoomLevel.toFixed(1) }}x</span>
          </div>
          <input type="range" min="1" max="50" step="0.1" :value="zoomLevel"
            @input="onZoomInput"
            style="width:100%;accent-color:#45b7d1" />
        </div>

        <!-- Classes -->
        <div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <span class="section-label" style="margin-bottom:0">Classes</span>
            <button class="text-btn" @click="toggleAllClasses">
              {{ enabledClasses.size === data.classes.length ? "None" : "All" }}
            </button>
          </div>
          <div
            v-for="(cls, idx) in data.classes"
            :key="idx"
            class="class-row"
            :style="{ background: enabledClasses.has(idx) ? '#0e0e1a' : 'transparent', opacity: enabledClasses.has(idx) ? 1 : 0.35 }"
            @click="toggleClass(idx)"
          >
            <div class="class-dot" :style="{ background: CLASS_COLORS[idx % CLASS_COLORS.length] }" />
            <div style="flex:1;min-width:0">
              <div class="class-name">{{ cls }}</div>
              <div v-if="showStats && classStats[idx]" class="class-stat">
                {{ classStats[idx].pct.toFixed(0) }}% frames · avg {{ classStats[idx].avgConf.toFixed(2) }}
              </div>
            </div>
            <div v-if="enabledClasses.has(idx) && classStats[idx]" style="width:40px;height:12px;flex-shrink:0">
              <svg width="40" height="12" viewBox="0 0 40 12">
                <rect
                  v-for="(p, i) in sparklines[idx]"
                  :key="i"
                  :x="i * 2"
                  :y="12 - p * 12"
                  width="1.5"
                  :height="p * 12"
                  :fill="CLASS_COLORS[idx % CLASS_COLORS.length]"
                  opacity="0.7"
                />
              </svg>
            </div>
          </div>
        </div>
      </div>

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
            <div v-else style="text-align:center;padding:20px;color:#444">No prediction frame</div>
          </div>
          <!-- Raw mode: video element + canvas overlay -->
          <div v-else-if="videoSrc" class="video-wrapper">
            <video
              ref="videoRef"
              class="video-el"
              playsinline
              preload="auto"
            />
            <!-- Canvas overlay only shown in raw mode —annotations are baked into prediction videos -->
            <canvas
              v-if="videoMode === 'raw'"
              ref="overlayRef"
              class="overlay-canvas"
            />
          </div>
          <div v-else style="text-align:center;padding:20px">
            <div style="font-size:14px;color:#333;margin-bottom:4px">
              No video loaded —
              {{ data.fps === 1 ? 'each frame = 1 second' : 'using frame simulation' }}
            </div>
            <div class="sim-time">{{ data.fps === 1 ? formatRealTime(currentFrame) : formatTime(currentTime) }}</div>
          </div>

          <div v-if="currentPartName" class="part-badge">{{ currentPartName }}</div>

          <div class="det-count-overlay">
            <span style="font-size:24px;font-weight:700;color:#4ecdc4">{{ currentDetections.length }}</span>
            <span style="font-size:13px;color:#666;margin-left:6px">detections</span>
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

        <!-- Class labels + Raster -->
        <div style="display:flex;border-top:1px solid #1a1a24;flex-shrink:0">
          <div style="width:120px;flex-shrink:0;background:#08080e;border-right:1px solid #1a1a24">
            <div
              v-for="(cls, idx) in data.classes"
              :key="idx"
              class="raster-label"
              :style="{
                height: rasterLabelHeight,
                color: enabledClasses.has(idx) ? '#999' : '#333'
              }"
            >
              <div
                class="raster-label-bar"
                :style="{
                  height: rasterLabelBarHeight,
                  background: enabledClasses.has(idx) ? CLASS_COLORS[idx % CLASS_COLORS.length] : '#222'
                }"
              />
              <span style="overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{{ cls }}</span>
            </div>
          </div>

          <div style="flex:1;position:relative">
            <canvas
              ref="rasterRef"
              style="width:100%;height:200px;display:block;cursor:crosshair"
              @click="handleRasterClick"
              @mousemove="handleRasterMouseMove"
              @mousedown="handleRasterMouseDown"
              @mouseleave="hoveredFrame = null"
              @wheel.prevent="handleWheel"
            />
            <div v-if="hoveredFrame !== null" class="hover-tooltip">
              Frame {{ hoveredFrame }} ·
              {{ data.fps === 1 ? formatRealTime(hoveredFrame) : formatTime(hoveredFrame / data.fps) }}
            </div>
          </div>
        </div>

        <!-- Minimap -->
        <div style="padding:6px 0 6px 120px;border-top:1px solid #1a1a24;background:#08080e;flex-shrink:0">
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

      <!-- ── Right Stats Panel ──────────────────────────────────────────── -->
      <div v-if="showStats" class="right-panel">
        <div class="section-label">Frame Analysis</div>
        <div v-if="currentDetections.length === 0" style="font-size:14px;color:#333;padding:20px 0;text-align:center">
          No detections in frame
        </div>
        <div
          v-else
          v-for="(d, i) in currentDetections"
          :key="i"
          class="det-card"
        >
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
            <div class="det-dot" :style="{ background: CLASS_COLORS[d.class_id % CLASS_COLORS.length] }" />
            <span style="font-size:15px;font-weight:600">{{ d.class_name }}</span>
          </div>
          <div style="display:flex;align-items:center;gap:8px">
            <div style="flex:1;height:4px;background:#1a1a24;border-radius:2px;overflow:hidden">
              <div
                :style="{
                  width: `${d.confidence * 100}%`,
                  height: '100%',
                  borderRadius: '2px',
                  background: `linear-gradient(90deg, ${CLASS_COLORS[d.class_id % CLASS_COLORS.length]}88, ${CLASS_COLORS[d.class_id % CLASS_COLORS.length]})`
                }"
              />
            </div>
            <span style="font-size:13px;color:#888;font-variant-numeric:tabular-nums;width:42px;text-align:right">
              {{ (d.confidence * 100).toFixed(0) }}%
            </span>
          </div>
          <div v-if="d.bbox" style="font-size:11px;color:#444;margin-top:4px">
            bbox: [{{ d.bbox.map(v => v.toFixed(0)).join(", ") }}]
          </div>
        </div>

        <div class="section-label" style="margin-top:24px">Session Summary</div>
        <div
          v-for="(s, i) in sortedClassStats"
          :key="i"
          style="margin-bottom:8px"
        >
          <div style="display:flex;justify-content:space-between;font-size:13px;margin-bottom:3px">
            <span style="color:#888">{{ s.name }}</span>
            <span style="color:#555">{{ s.pct.toFixed(0) }}%</span>
          </div>
          <div style="height:3px;background:#1a1a24;border-radius:2px;overflow:hidden">
            <div
              :style="{
                width: `${s.pct}%`,
                height: '100%',
                borderRadius: '2px',
                background: CLASS_COLORS[data.classes.indexOf(s.name) % CLASS_COLORS.length]
              }"
            />
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import {
  ref, computed, watch, watchEffect, onMounted, onUnmounted, nextTick, shallowRef,
} from "vue";
import { CLASS_COLORS, formatTime, formatRealTime } from "../utils/index.js";

// ── Constants ──────────────────────────────────────────────────────────────
const RATES = [0.25, 0.5, 1, 2, 4];
const SPARKLINE_BINS = 20;

// Precomputed RGB values for CLASS_COLORS to avoid repeated hex parsing in hot loops
const CLASS_COLORS_RGB = CLASS_COLORS.map(hex => ({
  r: parseInt(hex.slice(1, 3), 16),
  g: parseInt(hex.slice(3, 5), 16),
  b: parseInt(hex.slice(5, 7), 16),
}));

// ── State ──────────────────────────────────────────────────────────────────
const data              = shallowRef(null);  // large object — shallowRef avoids deep reactivity
const videoSrc          = ref(null);
const activeCaseName    = ref(null);
const currentFrame      = ref(0);
const isPlaying         = ref(false);
const confidenceThreshold = ref(0.25);
const enabledClasses    = ref(new Set());
const zoomLevel         = ref(1);
const panOffset         = ref(0);
const hoveredFrame      = ref(null);
const showStats         = ref(true);
const playbackRate      = ref(1);
const isDraggingTimeline = ref(false);
const showPicker        = ref(true);
const cases             = shallowRef([]);
const loadingCases      = ref(false);
const videoMode         = ref('raw');  // 'raw' | 'prediction'

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
const currentTime = computed(() => currentFrame.value / (data.value?.fps || 30));

// Derive the source video fps from timestamp intervals, then snap to the nearest
// standard framerate. Using just two timestamps gives ~30.03 which accumulates
// drift; averaging over many samples + snapping to a standard value is exact.
const sourceFps = computed(() => {
  const r = data.value?.results;
  if (!r || r.length < 2) return 30;
  // Average over up to 300 intervals for a stable estimate
  const n = Math.min(r.length - 1, 300);
  const dt = (r[n].timestamp - r[0].timestamp) / n;
  const raw = dt > 0 ? 1 / dt : 30;
  // Snap to nearest standard framerate
  const standards = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];
  return standards.reduce((a, b) => Math.abs(b - raw) < Math.abs(a - raw) ? b : a);
});

const currentPartRawUrl = computed(() => {
  if (!activeCaseName.value) return null;
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry?.source) return null;
  const partName = entry.source.split('/').pop().replace(/\.mp4$/i, '');
  return `/api/raw/${activeCaseName.value}/${partName}.mp4`;
});

const currentPartPredictionFrameUrl = computed(() => {
  if (!activeCaseName.value) return null;
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry?.source) return null;
  const partName = entry.source.split('/').pop().replace(/\.mp4$/i, '');
  // Compute exact 0-based frame index within the part using the source video fps.
  // Sending ?n= (frame index) instead of ?t= (timestamp) avoids drift caused by
  // AVI fps metadata != source fps (e.g. AVI says 29fps but source is 30fps).
  const t_within_part = entry.timestamp - (partStartTs.value.get(entry.source) ?? 0);
  const frameIndex = Math.round(t_within_part * sourceFps.value);
  return `/api/prediction-frame/${activeCaseName.value}/${partName}.avi?n=${frameIndex}`;
});

// In prediction mode we show frames via <img>; no video element is needed.
// currentPartVideoUrl returns null in prediction mode so the video watcher is a no-op.
const currentPartVideoUrl = computed(() =>
  videoMode.value === 'prediction' ? null : currentPartRawUrl.value
);

const currentPartTimestamp = computed(() => {
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry) return currentFrame.value;
  const startTs = partStartTs.value.get(entry.source) ?? 0;
  return entry.timestamp - startTs;
});

const currentPartName = computed(() => {
  const entry = frameMap.value.get(currentFrame.value);
  return entry?.source?.split('/').pop().replace(/\.mp4$/i, '') ?? null;
});

// Sparse lookup: frame number → result entry
const frameMap = computed(() => {
  if (!data.value) return new Map();
  return new Map(data.value.results.map(r => [r.frame, r]));
});

// Per-part start timestamp (global ts of the first frame in each part file).
// Used to convert global timestamps → per-file currentTime for the video element.
const partStartTs = computed(() => {
  if (!data.value) return new Map();
  const map = new Map(); // source path → start timestamp
  for (const r of data.value.results) {
    if (!map.has(r.source)) map.set(r.source, r.timestamp);
  }
  return map;
});

// Start timestamp of whichever part is currently loaded in the video element.
// Derived from currentFrame so it works correctly during part switches.
const currentPartStartTs = computed(() => {
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry) return 0;
  return partStartTs.value.get(entry.source) ?? 0;
});

// Precomputed raster label heights (avoid repeated template expressions)
const rasterLabelHeight = computed(() =>
  data.value ? `${200 / data.value.classes.length}px` : '20px'
);
const rasterLabelBarHeight = computed(() =>
  data.value ? `${200 / data.value.classes.length - 6}px` : '14px'
);

// ── Optimized: single-pass sparse filtered data ────────────────────────────
// Instead of building a dense array[total_frames], build a sparse Map of only
// frames that have detections passing filters. Used by raster/minimap drawing.
const filteredSparseMap = computed(() => {
  if (!data.value) return null;
  const numClasses = data.value.classes.length;
  const enabled = enabledClasses.value;
  const threshold = confidenceThreshold.value;
  const result = new Map();

  for (const [frameNum, entry] of frameMap.value) {
    let row = null;
    for (const det of entry.detections) {
      if (enabled.has(det.class_id) && det.confidence >= threshold) {
        if (!row) {
          row = new Float32Array(numClasses); // 0-initialized, compact
        }
        // Keep max confidence per class per frame
        if (det.confidence > row[det.class_id]) {
          row[det.class_id] = det.confidence;
        }
      }
    }
    if (row) result.set(frameNum, row);
  }
  return result;
});

const currentDetections = computed(() => {
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry) return [];
  return entry.detections.filter(
    d => enabledClasses.value.has(d.class_id) && d.confidence >= confidenceThreshold.value
  );
});

// ── Optimized: single-pass class stats ─────────────────────────────────────
const classStats = computed(() => {
  if (!data.value) return [];
  const numClasses = data.value.classes.length;
  const enabled = enabledClasses.value;
  const threshold = confidenceThreshold.value;
  const sampledCount = data.value.results.length || 1;

  // Single pass over all results
  const counts = new Uint32Array(numClasses);
  const totals = new Float64Array(numClasses);
  const maxes  = new Float64Array(numClasses);

  for (const [, entry] of frameMap.value) {
    for (const det of entry.detections) {
      const cid = det.class_id;
      if (enabled.has(cid) && det.confidence >= threshold) {
        counts[cid]++;
        totals[cid] += det.confidence;
        if (det.confidence > maxes[cid]) maxes[cid] = det.confidence;
      }
    }
  }

  return data.value.classes.map((name, idx) => ({
    name,
    count: counts[idx],
    avgConf: counts[idx] ? totals[idx] / counts[idx] : 0,
    maxConf: maxes[idx],
    pct: counts[idx] / sampledCount * 100,
  }));
});

const sortedClassStats = computed(() =>
  classStats.value.filter(s => s.count > 0).sort((a, b) => b.pct - a.pct)
);

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

// ── Actions ────────────────────────────────────────────────────────────────
async function loadCases() {
  loadingCases.value = true;
  try {
    const res = await fetch("/api/cases");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    cases.value = await res.json();
  } catch (err) {
    console.error("Failed to load cases:", err);
  } finally {
    loadingCases.value = false;
  }
}

async function loadCase(caseName) {
  try {
    const res = await fetch(`/data/predictions/${caseName}/detections.json`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const parsed = await res.json();
    data.value = parsed;
    activeCaseName.value = caseName;
    enabledClasses.value = new Set(parsed.classes.map((_, i) => i));
    showPicker.value = false;
    videoSrc.value = null;
    currentFrame.value = 0;
    nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0));
  } catch (err) {
    alert(`Failed to load case "${caseName}": ${err.message}`);
  }
}

function newSession() {
  showPicker.value = true;
  data.value = null;
  videoSrc.value = null;
  activeCaseName.value = null;
}

function seekToFrame(frame) {
  if (!data.value) return;
  currentFrame.value = frame;
  if (videoRef.value && videoSrc.value) {
    const entry = frameMap.value.get(frame);
    const startTs = entry ? (partStartTs.value.get(entry.source) ?? 0) : 0;
    videoRef.value.currentTime = entry ? entry.timestamp - startTs : frame;
  }
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

function toggleAllClasses() {
  if (!data.value) return;
  enabledClasses.value = enabledClasses.value.size === data.value.classes.length
    ? new Set()
    : new Set(data.value.classes.map((_, i) => i));
}

function onZoomInput(e) {
  zoomLevel.value = parseFloat(e.target.value);
  panOffset.value = Math.min(panOffset.value, 1 - 1 / zoomLevel.value);
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
    ctx.fillStyle = "#fff";
    ctx.fillText(label, rx + 4, ly - 2);
  }
}

function drawRaster() {
  const canvas = rasterRef.value;
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
  const visibleFraction = 1 / zoomLevel.value;
  const startFrame = Math.floor(panOffset.value * totalFrames);
  const endFrame = Math.min(Math.ceil((panOffset.value + visibleFraction) * totalFrames), totalFrames);
  const visibleFrames = endFrame - startFrame;
  const pxPerFrame = W / visibleFrames;

  ctx.fillStyle = "#0a0a0f";
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i < numClasses; i++) {
    ctx.beginPath(); ctx.moveTo(0, i * rowH); ctx.lineTo(W, i * rowH); ctx.stroke();
  }

  // Only iterate sparse entries in the visible range
  for (const [f, row] of sparse) {
    if (f < startFrame || f >= endFrame) continue;
    const x = (f - startFrame) * pxPerFrame;
    const barW = Math.max(pxPerFrame, 1);
    for (let c = 0; c < numClasses; c++) {
      if (row[c] > 0) {
        const alpha = 0.3 + row[c] * 0.7;
        const rgb = CLASS_COLORS_RGB[c % CLASS_COLORS_RGB.length];
        ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
        ctx.fillRect(x, c * rowH + 1, barW, rowH - 2);
      }
    }
  }

  // Playhead
  const playheadX = (currentFrame.value - startFrame) * pxPerFrame;
  if (playheadX >= 0 && playheadX <= W) {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 6;
    ctx.beginPath(); ctx.moveTo(playheadX, 0); ctx.lineTo(playheadX, H); ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Hover line
  if (hoveredFrame.value !== null) {
    const hx = (hoveredFrame.value - startFrame) * pxPerFrame;
    if (hx >= 0 && hx <= W) {
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(hx, 0); ctx.lineTo(hx, H); ctx.stroke();
      ctx.setLineDash([]);
    }
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

  ctx.fillStyle = "#0a0a0f";
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
  ctx.fillStyle = "rgba(255,255,255,0.05)";
  ctx.fillRect(vpX, 0, vpW, H);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(vpX, 0, vpW, H);

  // Playhead
  const phX = (currentFrame.value / totalFrames) * W;
  ctx.strokeStyle = "#ff6b6b";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(phX, 0); ctx.lineTo(phX, H); ctx.stroke();
}

// ── Reactive drawing (batched via RAF) ─────────────────────────────────────
watch(
  [currentDetections, videoRef],
  ([, vid]) => {
    if (vid && !vid.videoWidth) {
      vid.addEventListener("loadedmetadata", () => scheduleDraws(1), { once: true });
    } else {
      scheduleDraws(1);
    }
  },
  { flush: "post" }
);

// Single watcher for raster + minimap (they share most deps)
watch(
  [filteredSparseMap, () => currentFrame.value, () => zoomLevel.value, () => panOffset.value, () => hoveredFrame.value],
  () => scheduleDraws(6),  // 2 | 4 = raster + minimap
  { flush: "post" }
);

// ── Video part switching ───────────────────────────────────────────────────
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
  const wasPlaying = isPlaying.value;
  if (videoRef.value) videoRef.value.pause();
  videoSrc.value = newUrl;
  nextTick(() => {
    if (!videoRef.value) return;
    const seekAndPlay = () => {
      videoRef.value.currentTime = currentPartTimestamp.value;
      if (wasPlaying) videoRef.value.play();
    };
    videoRef.value.src = newUrl;
    videoRef.value.addEventListener('loadedmetadata', seekAndPlay, { once: true });
  });
});

// ── Video sync ─────────────────────────────────────────────────────────────
watch([videoRef, data], ([videoEl, d]) => {
  if (!videoEl || !d) return;
  const sync = () => {
    const globalTime = videoEl.currentTime + currentPartStartTs.value;
    currentFrame.value = Math.min(Math.floor(globalTime * d.fps), d.total_frames - 1);
    if (!videoEl.paused) animFrameRef.value = requestAnimationFrame(sync);
  };
  const onPlay  = () => { isPlaying.value = true;  sync(); };
  const onPause = () => { isPlaying.value = false; cancelAnimationFrame(animFrameRef.value); };
  videoEl.addEventListener("play", onPlay);
  videoEl.addEventListener("pause", onPause);
  videoEl.addEventListener("seeked", sync);
});

// ── No-video frame simulation ──────────────────────────────────────────────
watchEffect((onCleanup) => {
  if (videoSrc.value || !data.value || !isPlaying.value) return;
  const fps = data.value.fps;
  const total = data.value.total_frames;
  const id = setInterval(() => {
    const next = currentFrame.value + 1;
    if (next >= total) { isPlaying.value = false; return; }
    currentFrame.value = next;
  }, 1000 / (fps * playbackRate.value));
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
  if (isDraggingTimeline.value && frame !== null) seekToFrame(frame);
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
  isPanningRef.value = false;
  isDraggingTimeline.value = false;
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

async function loadFromUrl(url) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const parsed = await res.json();
    data.value = parsed;
    enabledClasses.value = new Set(parsed.classes.map((_, i) => i));
    showPicker.value = false;
  } catch (err) {
    alert(`Failed to load ${url}: ${err.message}`);
  }
}

onMounted(() => {
  window.addEventListener("mousemove", onGlobalMouseMove);
  window.addEventListener("mouseup", onGlobalMouseUp);
  window.addEventListener("keydown", onKeyDown);

  const params = new URLSearchParams(window.location.search);
  const dataUrl = params.get("data");
  if (dataUrl) loadFromUrl(dataUrl);

  loadCases();
});

onUnmounted(() => {
  window.removeEventListener("mousemove", onGlobalMouseMove);
  window.removeEventListener("mouseup", onGlobalMouseUp);
  window.removeEventListener("keydown", onKeyDown);
  if (_rafId) cancelAnimationFrame(_rafId);
});
</script>

<style scoped>
* { box-sizing: border-box; }

/* ── Upload screen ──────────────────────────────────────────────────────── */
.upload-root {
  min-height: 100vh; background: #06060a;
  display: flex; align-items: center; justify-content: center;
  font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
  color: #e0e0e6;
}
.upload-center { text-align: center; max-width: 860px; width: 100%; padding: 40px; }
.upload-label { font-size: 13px; letter-spacing: 6px; color: #555; margin-bottom: 12px; text-transform: uppercase; }
.upload-title {
  font-size: 42px; font-weight: 700; margin: 0 0 8px;
  background: linear-gradient(135deg, #ff6b6b, #4ecdc4, #45b7d1);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.upload-subtitle { color: #666; font-size: 14px; margin-bottom: 48px; line-height: 1.6; }
.cases-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px; width: 100%; margin-bottom: 8px;
}
.case-card {
  padding: 20px 16px; border: 1px dashed #2a2a35;
  border-radius: 10px; cursor: pointer; background: #0c0c14;
  transition: border-color 0.2s, background 0.2s;
  text-align: center; font-family: inherit; color: #ccc;
}
.case-card:hover { border-color: #4ecdc4; background: #0e0e18; }
.case-icon { font-size: 22px; margin-bottom: 8px; }
.case-name { font-size: 11px; font-weight: 600; word-break: break-all; }
.picker-status { font-size: 13px; color: #555; padding: 32px 0; }
.picker-status code { color: #4ecdc4; }

/* ── App layout ─────────────────────────────────────────────────────────── */
.app-root {
  height: 100vh; overflow: hidden; background: #06060a;
  font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
  color: #e0e0e6; display: flex; flex-direction: column;
}
.header {
  padding: 10px 20px; display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid #1a1a24; background: #08080e; flex-shrink: 0;
}
.header-title {
  font-size: 18px; font-weight: 700;
  background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-meta { font-size: 14px; color: #444; }
.hdr-btn {
  padding: 6px 14px; border: 1px solid #2a2a35; border-radius: 5px;
  color: #999; font-size: 13px; cursor: pointer; background: transparent;
  font-family: 'JetBrains Mono', monospace;
}
.hdr-btn--active { background: #1a1a2e; }
.mode-toggle {
  display: flex; border: 1px solid #2a2a35; border-radius: 5px; overflow: hidden;
  font-family: 'JetBrains Mono', monospace;
}
.mode-btn {
  padding: 6px 14px; border: none; background: transparent;
  color: #555; font-size: 12px; cursor: pointer;
  font-family: inherit; transition: all .15s;
}
.mode-btn--active { background: #1a1a2e; color: #4ecdc4; }

.body-row { display: flex; flex: 1; overflow: hidden; }

/* ── Left panel ─────────────────────────────────────────────────────────── */
.left-panel {
  width: 280px; border-right: 1px solid #1a1a24; overflow-y: auto;
  background: #08080e; padding: 16px 14px; flex-shrink: 0;
}
.section { margin-bottom: 20px; }
.section-label {
  font-size: 12px; color: #555; letter-spacing: 2px;
  margin-bottom: 10px; text-transform: uppercase; display: block;
}
.btn {
  padding: 8px 14px; background: #0c0c16; border: 1px solid #2a2a35;
  border-radius: 5px; color: #999; font-size: 14px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace;
}
.btn-play { flex: 1; }
.btn-play--pause { background: #ff6b6b22; border-color: #ff6b6b44; }
.btn-play--go { background: #4ecdc422; border-color: #4ecdc444; }
.btn-rate { flex: 1; font-size: 12px; background: transparent; }
.btn-rate--active { background: #4ecdc422; border-color: #4ecdc444; }

.time-display {
  padding: 12px; background: #0c0c16; border-radius: 8px; margin-bottom: 20px;
  border: 1px solid #1a1a24; text-align: center;
}
.time-value { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.time-sub { font-size: 13px; color: #555; margin-top: 4px; }
.text-btn {
  font-size: 12px; color: #666; background: none; border: none;
  cursor: pointer; font-family: inherit; padding: 2px 6px;
}
.class-row {
  display: flex; align-items: center; gap: 8px; padding: 7px 8px;
  margin-bottom: 2px; border-radius: 6px; cursor: pointer; transition: all 0.15s;
}
.class-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
.class-name { font-size: 14px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.class-stat { font-size: 11px; color: #555; margin-top: 1px; }

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
  background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.part-badge {
  position: absolute; top: 10px; left: 10px; padding: 4px 12px;
  background: rgba(0,0,0,0.7); border-radius: 5px; backdrop-filter: blur(8px);
  border: 1px solid #2a2a35; font-size: 12px; color: #45b7d1;
}
.det-count-overlay {
  position: absolute; top: 10px; right: 10px; padding: 6px 12px;
  background: rgba(0,0,0,0.7); border-radius: 6px; backdrop-filter: blur(8px);
  border: 1px solid #2a2a35;
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
  font-size: 11px; border-bottom: 1px solid #0e0e18;
}
.raster-label-bar { width: 4px; border-radius: 2px; margin-right: 6px; flex-shrink: 0; }
.hover-tooltip {
  position: absolute; top: 4px; right: 4px; padding: 5px 12px;
  background: rgba(0,0,0,0.85); border-radius: 4px; font-size: 13px;
  color: #aaa; pointer-events: none; border: 1px solid #2a2a35;
}
.kbd-bar {
  padding: 7px 16px; border-top: 1px solid #1a1a24; background: #08080e;
  font-size: 12px; color: #333; display: flex; gap: 16px; flex-shrink: 0;
}
.kbd {
  padding: 2px 6px; background: #12121e; border-radius: 3px;
  border: 1px solid #2a2a35; font-size: 11px; font-family: inherit;
}

/* ── Right panel ────────────────────────────────────────────────────────── */
.right-panel {
  width: 240px; border-left: 1px solid #1a1a24; overflow-y: auto;
  background: #08080e; padding: 16px 14px; flex-shrink: 0;
}
.det-card {
  padding: 10px; margin-bottom: 6px; border-radius: 8px;
  background: #0c0c16; border: 1px solid #1a1a24;
}
.det-dot { width: 8px; height: 8px; border-radius: 2px; }
</style>