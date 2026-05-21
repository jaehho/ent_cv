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
      v-model:video-mode="videoMode"
      :has-prediction-frames="!!data?.has_prediction_frames"
      @back="goBack"
      @logout="emit('logout')"
    />

    <div class="body-row">

      <!-- ── Left Panel ────────────────────────────────────────────────── -->
      <div
        class="left-panel"
        :class="{ 'side-panel--collapsed': leftPanelCollapsed }"
        :style="{ width: (leftPanelCollapsed ? 36 : leftPanelWidth) + 'px' }"
      >
        <button
          class="panel-toggle"
          @click="leftPanelCollapsed = !leftPanelCollapsed"
          :title="leftPanelCollapsed ? 'Show controls' : 'Hide controls'"
          :aria-label="leftPanelCollapsed ? 'Show controls' : 'Hide controls'"
        >
          <ChevronRight v-if="leftPanelCollapsed" :size="14" :stroke-width="2" />
          <ChevronLeft  v-else                    :size="14" :stroke-width="2" />
        </button>
        <template v-if="!leftPanelCollapsed">
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
            v-model:view-mode="detectionViewMode"
            :filtered-available="hasFilteredOverlay"
            @toggle-play="togglePlay"
            @set-rate="setRate"
            @seek-prev="seekFiltered(-1)"
            @seek-next="seekFiltered(1)"
            @seek-frame="seekToFrame"
            @toggle-jump-class="toggleJumpClass"
          />

          <!-- Filter info: shown when filtered detections are loaded (raw + filtered both on screen) -->
          <div v-if="hasFilteredOverlay && filterInfo" class="filter-info-card">
            <div class="filter-info-row">
              <span class="filter-info-label">Filter</span>
              <span class="filter-info-method">{{ filterInfo.method.replace(/_/g, ' ') }}</span>
            </div>
            <div class="filter-info-row filter-info-row--params">
              <template v-if="filterInfo.min_duration_sec != null">
                min {{ filterInfo.min_duration_sec }}s · gap {{ filterInfo.gap_fill_sec }}s
              </template>
              <template v-else-if="filterInfo.window_sec != null">
                window {{ filterInfo.window_sec }}s · thr {{ filterInfo.vote_threshold }}
              </template>
            </div>
          </div>
        </template>
      </div>

      <!-- Left resize handle: only meaningful when panel is expanded. -->
      <div
        v-if="!leftPanelCollapsed"
        class="resize-handle resize-handle--col"
        @mousedown="startResize('left', $event)"
      ></div>

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
          <!-- Raw mode: video element + canvas overlay drawing raw bboxes on top.
               (Raw-vs-filtered comparison overlay parked 2026-05-20 — see DESIGN_LOG.md.) -->
          <div v-else-if="videoSrc" class="video-wrapper">
            <video
              ref="videoRef"
              class="video-el"
              playsinline
              preload="auto"
              muted
            />
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

          <div ref="rasterWrapRef" style="flex:1;position:relative;overflow:hidden;">
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
              @dblclick="resetZoom"
              @mousemove="handleRasterMouseMove"
              @mousedown="handleRasterMouseDown"
              @mouseleave="hoveredFrame = null"
              @wheel.passive="handleWheel"
            />
            <!-- Playhead drawn as a transform-only div so currentFrame updates don't repaint the raster canvas -->
            <div
              v-if="playheadX >= 0"
              class="raster-playhead"
              :style="{ transform: `translateX(${playheadX}px)` }"
            ></div>
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

            <!-- Shift+drag region select. Visible only while the gesture is active. -->
            <div
              v-if="regionSelect"
              class="region-select"
              :style="{
                left: Math.min(regionSelect.startX, regionSelect.currentX) + 'px',
                width: Math.max(1, Math.abs(regionSelect.currentX - regionSelect.startX)) + 'px',
              }"
            ></div>
          </div>
        </div>

        <!-- Minimap. Drag the viewport block to pan; drag its left/right edge to
             rescale (zoom). Clicking outside the viewport recenters on cursor.
             Viewport rectangle and playhead are CSS divs so pan/zoom/scrub
             never repaint the minimap canvas. -->
        <div style="padding:6px 0 6px 120px;border-top:1px solid var(--border);background:var(--bg-1);flex-shrink:0">
          <div ref="minimapWrapRef" style="position:relative;line-height:0">
            <canvas
              ref="minimapRef"
              :style="{ width:'100%', height:'24px', display:'block', borderRadius:'3px', cursor: minimapCursor }"
              @mousedown="handleMinimapMouseDown"
              @mousemove="handleMinimapHover"
              @mouseleave="minimapCursor = 'pointer'"
              @dblclick="resetZoom"
            />
            <div
              v-if="data && minimapWidth > 0"
              class="minimap-viewport"
              :style="{ left: minimapViewport.left + 'px', width: minimapViewport.width + 'px' }"
            ></div>
            <div
              v-if="data && minimapWidth > 0"
              class="minimap-playhead"
              :style="{ transform: `translateX(${minimapPlayheadX}px)` }"
            ></div>
          </div>
        </div>

        <!-- Keyboard hints -->
        <div class="kbd-bar">
          <span><kbd class="kbd">Space</kbd> Play/Pause</span>
          <span><kbd class="kbd">←→</kbd> Frame step</span>
          <span><kbd class="kbd">Shift+←→</kbd> 10 frames</span>
          <span><kbd class="kbd">Shift+drag</kbd> Zoom region</span>
          <span><kbd class="kbd">Dbl-click</kbd> Reset zoom</span>
          <span><kbd class="kbd">Scroll</kbd> Zoom at cursor</span>
        </div>
      </div>

      <!-- Resize handle: only meaningful when right panel is expanded. -->
      <div
        v-if="!rightPanelCollapsed"
        class="resize-handle resize-handle--col"
        @mousedown="startResize('right', $event)"
      ></div>

      <!-- ── Right Stats Panel ──────────────────────────────────────────── -->
      <div
        class="right-panel"
        :class="{ 'side-panel--collapsed': rightPanelCollapsed }"
        ref="matrixContainerRef"
        :style="{
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          padding: 0,
          width: (rightPanelCollapsed ? 36 : rightPanelWidth) + 'px',
        }"
      >
        <button
          class="panel-toggle"
          @click="rightPanelCollapsed = !rightPanelCollapsed"
          :title="rightPanelCollapsed ? 'Show class panel' : 'Hide class panel'"
          :aria-label="rightPanelCollapsed ? 'Show class panel' : 'Hide class panel'"
        >
          <ChevronLeft  v-if="rightPanelCollapsed" :size="14" :stroke-width="2" />
          <ChevronRight v-else                     :size="14" :stroke-width="2" />
        </button>
        <template v-if="!rightPanelCollapsed">

        <ClassPanel
          :displayed-classes="displayedClasses"
          :enabled-classes="enabledClasses"
          :class-stats="classStats"
          :filtered-class-stats="filteredClassStats"
          :sparklines="sparklines"
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
            <!-- Row labels — height locked to cellSize so labels align row-for-row with the grid.
                 Long labels truncate with ellipsis; hover the title for the full name. -->
            <div :style="{ display:'flex', flexDirection:'column', marginRight:'8px', width: transitionMatrix.rowLabelW + 'px', flexShrink: 0 }">
              <!-- spacer aligns row labels with grid rows (below column label area) -->
              <div style="height:80px;flex-shrink:0"></div>
              <div v-for="cls in transitionMatrix.classes" :key="'rl-'+cls"
                :style="{ height: transitionMatrix.cellSize + 'px', width: transitionMatrix.rowLabelW + 'px' }"
                style="display:flex;align-items:center;justify-content:flex-end;font-size:9px;color:var(--text-faint);text-align:right;padding:0 4px 0 0;overflow:hidden;flex-shrink:0"
                :title="cls">
                <span style="display:inline-block;max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ cls }}</span>
              </div>
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
        </template>

      </div>

    </div>
  </div>
</template>

<script setup>
import {
  ref, computed, watch, watchEffect, onMounted, onUnmounted, nextTick,
} from "vue";
import { useRouter } from "vue-router";
import { ChevronLeft, ChevronRight } from "lucide-vue-next";
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
// `data` holds the universal context (classes, total_frames, fps, parts,
// _filter). The overlay and per-frame stats follow `detectionViewMode`:
// 'filtered' (default) sources from filteredOverlayResults so in_use flags
// and post-processed boxes appear; 'raw' shows the model's unfiltered
// output for debugging. The composable swaps the underlying results array
// based on this ref — no re-fetch, both payloads are kept in memory.
const detectionViewMode = ref("filtered");  // 'raw' | 'filtered'
const {
  data, filteredOverlayResults, dataReady, videoSrc, videoReady, activeCaseName,
  filterInfo, filteredFrameSet, filteredSummary, isLoading, hasFilteredOverlay,
  frameMap, partStartTs, partStartFrame,
  fetchCase,
} = useCaseData(detectionViewMode);

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
// Collapsed = a 36px icon strip with only the toggle chevron visible. Useful
// in compare mode where the user wants maximum vertical real estate for the
// two video panes.
const leftPanelCollapsed  = ref(localStorage.getItem('yolo-leftPanelCollapsed')  === '1');
const rightPanelCollapsed = ref(localStorage.getItem('yolo-rightPanelCollapsed') === '1');
watch(leftPanelCollapsed,  (v) => localStorage.setItem('yolo-leftPanelCollapsed',  v ? '1' : '0'));
watch(rightPanelCollapsed, (v) => localStorage.setItem('yolo-rightPanelCollapsed', v ? '1' : '0'));
const matrixContainerRef   = ref(null)
const matrixContainerWidth = ref(0)
let _matrixResizeObserver  = null
let _rasterResizeObserver  = null   // keeps rasterWidth in sync with the raster wrapper's CSS px width
const rasterHeight      = ref(200);   // px height of raster canvas

// ── Jump-filter state ─────────────────────────────────────────────────────
const jumpFilter        = ref('none'); // 'none'|'any'|'class'|'onset'|'changed'
const jumpFilterClassIds = ref(new Set()); // class indices when jumpFilter==='class'

// ── Refs ───────────────────────────────────────────────────────────────────
const videoRef     = ref(null);
const overlayRef   = ref(null);
const rasterRef    = ref(null);
const rasterWrapRef = ref(null);
const rasterWidth  = ref(0);   // CSS px, kept in sync via ResizeObserver
const minimapRef    = ref(null);
const minimapWrapRef = ref(null);
const minimapWidth  = ref(0);   // CSS px, kept in sync via ResizeObserver
let _minimapResizeObserver = null;
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

// Minimap viewport rectangle position/size (CSS px). Computed off the same
// state as the canvas-drawn version was, but driving a CSS div instead.
const minimapViewport = computed(() => {
  if (!data.value || minimapWidth.value <= 0) return { left: 0, width: 0 };
  const W = minimapWidth.value;
  return { left: panOffset.value * W, width: (1 / zoomLevel.value) * W };
});
const minimapPlayheadX = computed(() => {
  if (!data.value || minimapWidth.value <= 0) return 0;
  return (currentFrame.value / data.value.total_frames) * minimapWidth.value;
});

// Playhead position in CSS pixels, or -1 when out of view. Rendered as a
// transform on a sibling div so that currentFrame changes during playback /
// scrub move a single GPU-composited line instead of repainting the raster.
const playheadX = computed(() => {
  if (!data.value || rasterWidth.value <= 0) return -1;
  const totalFrames = data.value.total_frames;
  const visibleFraction = 1 / zoomLevel.value;
  const startFrame = Math.max(0, Math.floor(panOffset.value * totalFrames));
  const endFrame = Math.min(Math.ceil((panOffset.value + visibleFraction) * totalFrames), totalFrames);
  const visibleFrames = endFrame - startFrame;
  if (visibleFrames <= 0) return -1;
  const x = (currentFrame.value - startFrame) * (rasterWidth.value / visibleFrames);
  if (x < 0 || x > rasterWidth.value) return -1;
  return x;
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

// Parallel to filteredSparseMap: per-class in-use bitset per frame. A class
// counts as "in use" at frame f when any detection in that frame has
// in_use === true. Returns null when no detection in the case carries an
// in_use field (raw mode, or in-use disabled at postprocess time).
const inUseSparseMap = computed(() => {
  if (!data.value) return null;
  const numClasses = data.value.classes.length;
  const enabled = enabledClasses.value;
  const result = new Map();
  let sawInUseField = false;

  for (const [frameNum, entry] of frameMap.value) {
    let row = null;
    for (const det of entry.detections) {
      if (typeof det.in_use === 'boolean') sawInUseField = true;
      if (det.in_use === true && enabled.has(det.class_id)) {
        if (!row) row = new Uint8Array(numClasses);
        row[det.class_id] = 1;
      }
    }
    if (row) result.set(frameNum, row);
  }
  return sawInUseField ? result : null;
});

// Frames where raw and filtered disagree: detection in raw with none in filtered
// (rejection) or vice versa. Drives the change-strip in the raster/minimap.
// Null when there's no filtered overlay to compare against. With raw as primary,
// frameMap is the raw side and filteredFrameSet is the filter side.
const changedFrames = computed(() => {
  if (!hasFilteredOverlay.value || !filteredFrameSet.value || !data.value) return null;
  const rawSet = new Set(frameMap.value.keys());
  const filtSet = filteredFrameSet.value;
  const changed = new Set();
  for (const f of rawSet)  if (!filtSet.has(f)) changed.add(f);
  for (const f of filtSet) if (!rawSet.has(f))  changed.add(f);
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

// Filter annotation frame map: built only when filtered_detections.json was
// loaded. Each entry is a "kept by filter" marker that the overlay draws as
// a thin teal inset stroke inside the matching raw box.
const filteredAnnotationFrameMap = computed(() => {
  if (!filteredOverlayResults.value) return null;
  return new Map(filteredOverlayResults.value.map(r => [r.frame, r]));
});

const currentFilteredAnnotations = computed(() => {
  if (!filteredAnnotationFrameMap.value) return null;
  const entry = filteredAnnotationFrameMap.value.get(currentFrame.value);
  if (!entry) return [];
  return entry.detections.filter(d => enabledClasses.value.has(d.class_id));
});

// ── Optimized: single-pass class stats ─────────────────────────────────────
const classStats = computed(() => computeClassStats(frameMap.value));
// Parallel stats over the filtered annotation — null when no filter file exists,
// so the class panel falls back to showing just the raw count.
const filteredClassStats = computed(() =>
  filteredAnnotationFrameMap.value ? computeClassStats(filteredAnnotationFrameMap.value) : null
);

function computeClassStats(map) {
  if (!data.value || !map) return [];
  const numClasses = data.value.classes.length;
  const enabled = enabledClasses.value;
  const totalFrames = data.value.total_frames || 1;
  const counts = new Uint32Array(numClasses);
  for (const [, entry] of map) {
    for (const det of entry.detections) {
      if (enabled.has(det.class_id)) counts[det.class_id]++;
    }
  }
  return data.value.classes.map((name, idx) => ({
    name,
    count: counts[idx],
    pct: counts[idx] / totalFrames * 100,
  }));
}

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
  // Only resize on actual change: assigning canvas.width/height reallocates
  // the GPU backing store and wipes the context state, which is what made
  // per-tick overlay redraws so expensive during playback.
  const targetW = cw * dpr, targetH = ch * dpr;
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width  = targetW;
    canvas.height = targetH;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);   // identity-scale; replaces the implicit reset from resize
  ctx.clearRect(0, 0, cw, ch);

  const d = data.value;
  if (!d) return;
  const inferW = d.inference_width  || nW;
  const inferH = d.inference_height || nH;

  // Letterbox/pillarbox offset (object-fit: contain).
  const videoAspect  = nW / nH;
  const canvasAspect = cw / ch;
  let contentW, contentH, offX, offY;
  if (videoAspect > canvasAspect) {
    contentW = cw; contentH = cw / videoAspect;
    offX = 0;     offY = (ch - contentH) / 2;
  } else {
    contentH = ch; contentW = ch * videoAspect;
    offX = (cw - contentW) / 2; offY = 0;
  }
  const scaleX = contentW / inferW;
  const scaleY = contentH / inferH;

  // Single-view paint: raw detections only (no filter file on disk).
  const dets = currentDetections.value;
  if (!dets.length) return;

  ctx.font = "bold 13px 'JetBrains Mono', monospace";
  ctx.lineWidth = 2.5;
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

    const inUseSuffix = det.in_use === true ? "  IN USE" : "";
    const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%${inUseSuffix}`;
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
  const targetW = rect.width * dpr, targetH = rect.height * dpr;
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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

  // ⚡ Optimization: When many frames fall in one pixel (low zoom), iterate by
  // a step rather than every frame — multiple per-pixel writes are wasted work.
  // step = ceil(0.5 / pxPerFrame) keeps ~2 samples per pixel, more than enough
  // to resolve detection density visually. At high zoom (pxPerFrame >= 1)
  // step==1 and behavior is identical to the dense loop.
  const inUseSparse = inUseSparseMap.value;
  const step = pxPerFrame >= 1 ? 1 : Math.max(1, Math.ceil(0.5 / pxPerFrame));
  const barW = Math.max(pxPerFrame * step, 1);

  for (let f = startFrame; f < endFrame; f += step) {
    const row = sparse.get(f);
    if (!row) continue;
    const inUseRow = inUseSparse ? inUseSparse.get(f) : null;
    const x = (f - startFrame) * pxPerFrame;

    for (let c = 0; c < numClasses; c++) {
      if (row[c] > 0) {
        const displayRow = clsToRow.get(c) ?? c;
        let alpha = 0.3 + row[c] * 0.7;
        if (inUseSparse && !(inUseRow && inUseRow[c])) {
          alpha *= 0.35;  // present-but-idle: clearly dimmer than in-use
        }
        const rgb = CLASS_COLORS_RGB[c % CLASS_COLORS_RGB.length];

        ctx.fillStyle = `rgba(${rgb.r},${rgb.g},${rgb.b},${alpha})`;
        ctx.fillRect(x, displayRow * rowH + 1, barW, rowH - 2);
      }
    }
  }

  // Playhead is a sibling DOM element now — see template <div class="raster-playhead">.
  // Keeping it out of the canvas means currentFrame changes never repaint the raster.

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

// The minimap canvas only paints static-per-data content (bars + changed
// strip). The viewport rectangle and playhead live as sibling CSS divs so
// pan / zoom / scrub move only those — no canvas redraw. This is the same
// trick the raster playhead uses; without it, every minimap drag forces a
// full sparse-iteration repaint and the gesture jitters at high frame counts.
function drawMinimap() {
  const canvas = minimapRef.value;
  const sparse = filteredSparseMap.value;
  const d = data.value;
  if (!canvas || !sparse || !d) return;

  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const targetW = rect.width * dpr, targetH = rect.height * dpr;
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
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

  // Changed-frame markers (filtered mode) — 2-px strip at minimap top
  const mcf = changedFrames.value;
  if (mcf) {
    ctx.fillStyle = 'rgba(255, 210, 50, 0.9)';
    const pxPerFr = W / totalFrames;
    for (const f of mcf) {
      ctx.fillRect((f / totalFrames) * W, 0, Math.max(pxPerFr, 1), 2);
    }
  }
  // Viewport rectangle + playhead intentionally NOT drawn here — see CSS divs.
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


// Raster repaints only when something on it actually changed. The playhead
// lives in a sibling div (see `playheadX` + `.raster-playhead`), so currentFrame
// is intentionally NOT a dep here — that's the whole point of this refactor.
watch(
  [filteredSparseMap, changedFrames, () => zoomLevel.value, () => panOffset.value],
  () => scheduleDraws(2),  // raster
  { flush: "post" }
);

// Minimap content (background bars + changed-frame strip) only changes with
// the underlying data. Viewport rect + playhead are CSS divs now, so panning,
// zooming, and scrubbing never need to repaint this canvas.
watch(
  [filteredSparseMap, changedFrames],
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

// Keep rasterWidth in sync with the raster wrapper. Needed so playheadX
// stays correct on resize without us repainting the raster canvas.
watch(rasterWrapRef, (el) => {
  if (el && !_rasterResizeObserver) {
    rasterWidth.value = el.clientWidth;
    _rasterResizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) rasterWidth.value = entry.contentRect.width;
    });
    _rasterResizeObserver.observe(el);
  }
});

// Track minimap width too — drives the CSS-positioned viewport rectangle and
// playhead without forcing a canvas repaint.
watch(minimapWrapRef, (el) => {
  if (el && !_minimapResizeObserver) {
    minimapWidth.value = el.clientWidth;
    _minimapResizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) minimapWidth.value = entry.contentRect.width;
    });
    _minimapResizeObserver.observe(el);
  }
});

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
// "Pan only when off-screen": when the playhead leaves the visible window —
// whether from playback, arrow-step, jump, or click — pan to recenter it.
// While inside the window we leave the viewport alone, so manual mid-view
// scrubbing doesn't jitter.
//
// Exception: during playback we add a small leading-edge margin so the
// viewport scrolls a beat before the playhead actually exits, avoiding a
// visible flicker at the right edge.
watch([() => currentFrame.value, () => zoomLevel.value], () => {
  if (!data.value || zoomLevel.value <= 1) return;
  const visibleFraction = 1 / zoomLevel.value;
  const playheadPos = currentFrame.value / data.value.total_frames;
  const viewStart = panOffset.value;
  const viewEnd = panOffset.value + visibleFraction;
  const margin = isPlaying.value ? visibleFraction * 0.02 : 0;
  if (playheadPos < viewStart || playheadPos > viewEnd - margin) {
    panOffset.value = Math.max(0, Math.min(playheadPos - visibleFraction / 2, 1 - visibleFraction));
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
  // Active Shift+drag region select — update the rectangle's right edge.
  if (regionSelect.value && rasterRef.value) {
    const rect = rasterRef.value.getBoundingClientRect();
    regionSelect.value = { startX: regionSelect.value.startX, currentX: e.clientX - rect.left };
    return;
  }
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
    if (e.shiftKey && rasterRef.value) {
      // Shift+drag → region-zoom: don't scrub, just track a selection rectangle.
      e.preventDefault();
      const rect = rasterRef.value.getBoundingClientRect();
      const x = e.clientX - rect.left;
      regionSelect.value = { startX: x, currentX: x };
      return;
    }
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

function resetZoom() { zoomLevel.value = 1; panOffset.value = 0; }

// ── Minimap drag interactions ─────────────────────────────────────────────
// Hit zones on the viewport rectangle drawn over the minimap:
//   left edge (±MINIMAP_EDGE_PX) → drag to resize from left side  → zoom + pan
//   right edge (±MINIMAP_EDGE_PX) → drag to resize from right side → zoom only
//   inside the viewport          → drag to pan                    → pan only
//   outside the viewport         → press recenters; drag pans     → pan only
// `minimapCursor` ref reflects the hover zone so the cursor previews the action.
const MINIMAP_EDGE_PX = 6;
const minimapCursor = ref('pointer');
// Active drag: { mode: 'pan'|'resize-left'|'resize-right', startClientX, startOffset, startFrac, mapWidth }
let _minimapDrag = null;

function _minimapZone(x, vpStart, vpEnd) {
  if (x >= vpStart - MINIMAP_EDGE_PX && x <= vpStart + MINIMAP_EDGE_PX) return 'edge-left';
  if (x >= vpEnd - MINIMAP_EDGE_PX && x <= vpEnd + MINIMAP_EDGE_PX) return 'edge-right';
  if (x >= vpStart && x <= vpEnd) return 'inside';
  return 'outside';
}

function handleMinimapHover(e) {
  if (_minimapDrag || !minimapRef.value || !data.value) return;
  const rect = minimapRef.value.getBoundingClientRect();
  const W = rect.width;
  const x = e.clientX - rect.left;
  const visibleFraction = 1 / zoomLevel.value;
  const vpStart = panOffset.value * W;
  const vpEnd = vpStart + visibleFraction * W;
  const zone = _minimapZone(x, vpStart, vpEnd);
  minimapCursor.value =
    zone === 'edge-left' || zone === 'edge-right' ? 'ew-resize'
    : zone === 'inside' ? 'grab'
    : 'pointer';
}

function handleMinimapMouseDown(e) {
  if (!data.value || !minimapRef.value || e.button !== 0) return;
  const rect = minimapRef.value.getBoundingClientRect();
  const W = rect.width;
  const x = e.clientX - rect.left;
  const visibleFraction = 1 / zoomLevel.value;
  const vpStart = panOffset.value * W;
  const vpEnd = vpStart + visibleFraction * W;
  const zone = _minimapZone(x, vpStart, vpEnd);

  if (zone === 'edge-left' || zone === 'edge-right') {
    _minimapDrag = {
      mode: zone === 'edge-left' ? 'resize-left' : 'resize-right',
      startClientX: e.clientX,
      startOffset: panOffset.value,
      startFrac: visibleFraction,
      mapWidth: W,
    };
    minimapCursor.value = 'ew-resize';
  } else {
    // Inside-viewport drag pans from current offset.
    // Outside-viewport press recenters on cursor first, then drag pans from there.
    let startOffset = panOffset.value;
    if (zone === 'outside') {
      startOffset = Math.max(0, Math.min(x / W - visibleFraction / 2, 1 - visibleFraction));
      panOffset.value = startOffset;
    }
    _minimapDrag = {
      mode: 'pan',
      startClientX: e.clientX,
      startOffset,
      startFrac: visibleFraction,
      mapWidth: W,
    };
    minimapCursor.value = 'grabbing';
  }
}

function _onMinimapDragMove(e) {
  const d = _minimapDrag;
  if (!d || !data.value) return;
  const dx = (e.clientX - d.startClientX) / d.mapWidth;
  if (d.mode === 'pan') {
    panOffset.value = Math.max(0, Math.min(d.startOffset + dx, 1 - d.startFrac));
  } else if (d.mode === 'resize-left') {
    // Right edge stays fixed; drag changes left edge → newFrac = endFrac - newStart
    const endFrac = d.startOffset + d.startFrac;
    const newStart = Math.max(0, Math.min(d.startOffset + dx, endFrac - 0.01));
    const newFrac = endFrac - newStart;
    zoomLevel.value = Math.min(1 / newFrac, 100);
    panOffset.value = newStart;
  } else if (d.mode === 'resize-right') {
    // Left edge stays fixed; drag changes right edge → newFrac = newEnd - startOffset
    const newEnd = Math.max(d.startOffset + 0.01, Math.min(d.startOffset + d.startFrac + dx, 1));
    const newFrac = newEnd - d.startOffset;
    zoomLevel.value = Math.min(1 / newFrac, 100);
    // panOffset unchanged
  }
}

// ── Raster region-select (Shift+drag → zoom into region) ─────────────────
// `regionSelect` is set while a Shift+drag is in flight; the template overlays
// a translucent rectangle keyed off these CSS-px coordinates.
const regionSelect = ref(null); // { startX, currentX } in raster-wrapper px

function _finishRegionSelect() {
  const r = regionSelect.value;
  regionSelect.value = null;
  if (!r || !data.value || !rasterRef.value) return;
  const rect = rasterRef.value.getBoundingClientRect();
  const W = rect.width;
  if (W <= 0) return;
  const a = Math.max(0, Math.min(r.startX, W));
  const b = Math.max(0, Math.min(r.currentX, W));
  const left = Math.min(a, b), right = Math.max(a, b);
  if (right - left < 4) return; // accidental click, ignore
  const visibleFraction = 1 / zoomLevel.value;
  const startFrac = panOffset.value + (left  / W) * visibleFraction;
  const endFrac   = panOffset.value + (right / W) * visibleFraction;
  const newFrac = Math.max(endFrac - startFrac, 0.01);  // cap at 100×
  zoomLevel.value = Math.min(1 / newFrac, 100);
  panOffset.value = Math.max(0, Math.min(startFrac, 1 - 1 / zoomLevel.value));
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
  if (_minimapDrag) _onMinimapDragMove(e);
  if (regionSelect.value && rasterRef.value) {
    // Continue tracking even when the cursor leaves the canvas — the rectangle
    // clamps to the visible width on commit (_finishRegionSelect).
    const rect = rasterRef.value.getBoundingClientRect();
    regionSelect.value = { startX: regionSelect.value.startX, currentX: e.clientX - rect.left };
  }
}

function onGlobalMouseUp() {
  const wasDraggingTimeline = isDraggingTimeline.value;
  isPanningRef.value = false;
  isDraggingTimeline.value = false;
  if (_minimapDrag) {
    _minimapDrag = null;
    minimapCursor.value = 'pointer';
  }
  if (regionSelect.value) _finishRegionSelect();
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
    case "0":
      resetZoom(); break;
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
  if (_rasterResizeObserver) {
    _rasterResizeObserver.disconnect();
    _rasterResizeObserver = null;
  }
  if (_minimapResizeObserver) {
    _minimapResizeObserver.disconnect();
    _minimapResizeObserver = null;
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
  position: relative;
}
/* Collapsed side panel: 36px-wide strip with just the toggle chevron showing.
   Lets the user reclaim panel width for the compare canvases. */
.side-panel--collapsed {
  padding: 0 !important; overflow: hidden;
}
.panel-toggle {
  background: transparent; border: none;
  color: var(--text-faint); cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
  padding: 6px;
  border-radius: 4px;
  transition: color .15s, background .15s;
}
.panel-toggle:hover { color: var(--text); background: var(--bg-hover); }
.left-panel  .panel-toggle { float: right; margin-bottom: 8px; }
.right-panel .panel-toggle { float: left;  margin: 8px 0 8px 4px; }
.side-panel--collapsed .panel-toggle {
  float: none; margin: 0; padding: 12px 8px;
  width: 100%;
}
.section { margin-bottom: 20px; }
.section-label {
  font-size: 12px; color: var(--text-faint); letter-spacing: 2px;
  margin-bottom: 10px; text-transform: uppercase; display: block;
}

/* Filter info card — shown when a filtered overlay is loaded */
.filter-info-card {
  margin-bottom: 20px; padding: 10px 12px;
  background: var(--bg-2); border: 1px solid var(--border);
  border-radius: 6px;
}
.filter-info-row {
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 11px;
}
.filter-info-label {
  color: var(--text-faint); letter-spacing: 2px; text-transform: uppercase;
}
.filter-info-method { color: var(--text); }
.filter-info-row--params {
  margin-top: 4px;
  color: var(--text-faint); font-family: var(--font-mono);
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
/* Transform-positioned playhead — kept off the canvas so currentFrame updates
   compose on the GPU instead of triggering a raster redraw. */
.raster-playhead {
  position: absolute; top: 0; bottom: 0; left: 0; width: 2px;
  background: #fff; box-shadow: 0 0 6px #fff;
  pointer-events: none;
  will-change: transform;
}
.hover-tooltip {
  position: absolute; top: 4px; right: 4px; padding: 5px 12px;
  background: rgba(0,0,0,0.85); border-radius: 4px; font-size: 13px;
  color: var(--text-dim); pointer-events: none; border: 1px solid var(--border-strong);
}

/* Shift+drag region select rectangle on the raster. Spans the raster's full
   height; only the horizontal range matters for zooming. */
.region-select {
  position: absolute; top: 0; bottom: 0;
  background: rgba(78, 205, 196, 0.18);
  border-left: 1px solid var(--accent);
  border-right: 1px solid var(--accent);
  pointer-events: none;
  z-index: 4;
}

/* Minimap viewport rectangle + playhead are CSS-positioned overlays over
   the static minimap canvas. Updating their position/size never triggers a
   canvas repaint — only a compositor-level update — so pan/zoom/scrub is
   buttery smooth even at 100k+ frames. */
.minimap-viewport {
  position: absolute; top: 0; bottom: 0;
  background: rgba(255, 255, 255, 0.06);
  border-left:  1.5px solid #fff;
  border-right: 1.5px solid #fff;
  box-sizing: border-box;
  pointer-events: none;
  will-change: left, width;
}
.minimap-playhead {
  position: absolute; top: 0; bottom: 0; left: 0;
  width: 1px;
  background: var(--warn);
  pointer-events: none;
  will-change: transform;
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