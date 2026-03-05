<template>
  <!-- ── Case Picker Screen ────────────────────────────────────────────── -->
  <div v-if="showPicker" class="upload-root">
    <div class="upload-center">
      <div class="upload-label">Mount Sinai &amp; The Cooper Union</div>
      <h1 class="upload-title">ENT Computer Vision</h1>
      <p class="upload-subtitle">Select a prediction case to examine</p>

      <div v-if="loadingCases" class="picker-status">Loading cases…</div>
      <div v-else-if="cases.length === 0" class="picker-status">No cases found in <code>/mnt/data/ent_cv/predictions/</code></div>
      <div v-else>
        <div v-for="group in groupedCases" :key="group.label" style="margin-bottom:28px">
          <div class="date-group-label">{{ group.label }}</div>
          <div class="cases-grid">
            <button
              v-for="c in group.cases"
              :key="c.name"
              class="case-card"
              @click="loadCase(c.name)"
            >
              <div class="case-icon">📁</div>
              <div class="case-name">{{ c.display }}</div>
            </button>
          </div>
        </div>
      </div>

    </div>
  </div>

  <!-- ── Main Interface ─────────────────────────────────────────────────── -->
  <div v-else-if="data" class="app-root">

    <!-- Header -->
    <div class="header">
      <div style="display:flex;align-items:center;gap:16px">
        <button class="hdr-btn" @click="newSession">← Cases</button>
      </div>
      <div style="display:flex;gap:10px;align-items:center">
        <span v-if="username" style="font-size:12px;color:#666">{{ username }}</span>
        <button class="hdr-btn" @click="emit('logout')" style="font-size:11px">Sign Out</button>
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
      </div>
    </div>

    <div class="body-row">

      <!-- ── Left Panel ────────────────────────────────────────────────── -->
      <div class="left-panel" :style="{ width: leftPanelWidth + 'px' }">

        <!-- Playback -->
        <div class="section">
          <div class="section-label">Playback</div>
          <div style="display:flex;gap:6px;margin-bottom:12px">
            <button class="btn" @click="seekFiltered(-1)">◀◀</button>
            <button
              class="btn btn-play"
              :class="isPlaying ? 'btn-play--pause' : 'btn-play--go'"
              @click="togglePlay"
            >{{ isPlaying ? "⏸" : "▶" }}</button>
            <button class="btn" @click="seekFiltered(1)">▶▶</button>
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

          <!-- Jump filter -->
          <div style="margin-top:12px">
            <div style="font-size:11px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">
              Jump
              <span v-if="jumpFilter !== 'none' && jumpFrames" style="color:#4ecdc4;font-size:10px;letter-spacing:0;text-transform:none;float:right">{{ jumpFrames.length }} frames</span>
              <span v-else-if="jumpFilter === 'changed' && !changedFrames" style="color:#553;font-size:10px;letter-spacing:0;text-transform:none;float:right">run filter first</span>
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:3px">
              <button
                v-for="jf in JUMP_FILTERS"
                :key="jf.value"
                class="mode-btn"
                :class="{ 'mode-btn--active': jumpFilter === jf.value }"
                style="font-size:10px;padding:5px 6px;border:1px solid #2a2a35;border-radius:4px"
                @click="jumpFilter = jf.value"
              >{{ jf.label }}</button>
            </div>
            <!-- Class picker -->
            <div v-if="jumpFilter === 'class'" style="margin-top:8px;max-height:90px;overflow-y:auto;display:flex;flex-wrap:wrap;gap:4px;padding:2px 0">
              <button
                v-for="item in displayedClasses"
                :key="item.idx"
                @click="toggleJumpClass(item.idx)"
                :style="{
                  padding: '2px 8px',
                  borderRadius: '10px',
                  fontSize: '10px',
                  border: jumpFilterClassIds.has(item.idx)
                    ? `1px solid ${CLASS_COLORS[item.idx % CLASS_COLORS.length]}`
                    : '1px solid #222',
                  background: jumpFilterClassIds.has(item.idx)
                    ? CLASS_COLORS[item.idx % CLASS_COLORS.length] + '33'
                    : 'transparent',
                  color: jumpFilterClassIds.has(item.idx)
                    ? CLASS_COLORS[item.idx % CLASS_COLORS.length]
                    : '#555',
                  cursor: 'pointer',
                }"
              >{{ item.cls }}</button>
            </div>
          </div>
        </div>

        <!-- Time display -->
        <div class="time-display">
          <div class="time-value">
            <input
              type="text"
              :value="formatTime(currentTime)"
              @keydown.enter="e => { handleTimeInput(e.target.value); e.target.blur(); }"
              @blur="e => handleTimeInput(e.target.value)"
              class="time-input"
              title="Enter time: M:SS, H:MM:SS, or bare number (minutes, e.g. 30 → 30:00, 30.5 → 30:30)"
            />
            <div v-if="timeInputError" class="time-input-error">{{ timeInputError }}</div>
          </div>
          <div class="time-sub">
            Frame 
            <input
              type="number"
              :value="currentFrame + 1"
              @keydown.enter="e => handleFrameInput(e.target.value)"
              @blur="e => handleFrameInput(e.target.value)"
              min="1"
              :max="data.total_frames"
              class="frame-input"
            />
            / {{ data.total_frames }}
          </div>
        </div>

        <!-- Post-process -->
        <div class="section">
          <div class="section-label">Post-process</div>

          <!-- Raw / Filtered toggle -->
          <div class="mode-toggle" style="margin-bottom:12px">
            <button class="mode-btn" :class="{ 'mode-btn--active': filterMode === 'raw' }" style="flex:1" @click="filterMode = 'raw'">Raw</button>
            <button class="mode-btn" :class="{ 'mode-btn--active': filterMode === 'filtered' }" style="flex:1" @click="filterMode = 'filtered'">Filtered</button>
          </div>

          <!-- Active filter info chip -->
          <div v-if="filterMode === 'filtered' && filterInfo" style="font-size:11px;color:#555;margin-bottom:10px;padding:7px 8px;background:#0c0c16;border-radius:6px;border:1px solid #1a1a24">
            <span style="color:#666">{{ filterInfo.method.replace(/_/g, ' ') }}</span>
            <template v-if="filterInfo.min_duration_sec != null">
              &nbsp;· min {{ filterInfo.min_duration_sec }}s / gap {{ filterInfo.gap_fill_sec }}s
            </template>
            <template v-else-if="filterInfo.window_sec != null">
              &nbsp;· window {{ filterInfo.window_sec }}s / thr {{ filterInfo.vote_threshold }}
            </template>
          </div>

          <!-- Method selector -->
          <div style="margin-bottom:10px">
            <div style="font-size:11px;color:#555;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">Method</div>
            <div class="mode-toggle">
              <button v-for="m in PP_METHODS" :key="m" class="mode-btn" :class="{ 'mode-btn--active': ppMethod === m }" style="flex:1;font-size:10px;padding:5px 2px" @click="ppMethod = m">{{ m.replace(/_/g, ' ') }}</button>
            </div>
          </div>

          <!-- run_length params -->
          <template v-if="ppMethod === 'run_length'">
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:11px;color:#555">Min duration</span>
                <span style="font-size:11px;color:#4ecdc4">{{ ppMinDuration.toFixed(1) }}s</span>
              </div>
              <input type="range" min="0.1" max="5" step="0.1" :value="ppMinDuration" @input="e => ppMinDuration = parseFloat(e.target.value)" style="width:100%;accent-color:#4ecdc4" />
            </div>
            <div style="margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:11px;color:#555">Gap fill</span>
                <span style="font-size:11px;color:#4ecdc4">{{ ppGapFill.toFixed(1) }}s</span>
              </div>
              <input type="range" min="0" max="5" step="0.1" :value="ppGapFill" @input="e => ppGapFill = parseFloat(e.target.value)" style="width:100%;accent-color:#4ecdc4" />
            </div>
          </template>

          <!-- majority_vote / gaussian params -->
          <template v-else>
            <div style="margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:11px;color:#555">Window</span>
                <span style="font-size:11px;color:#4ecdc4">{{ ppWindowSec.toFixed(1) }}s</span>
              </div>
              <input type="range" min="0.5" max="10" step="0.5" :value="ppWindowSec" @input="e => ppWindowSec = parseFloat(e.target.value)" style="width:100%;accent-color:#4ecdc4" />
            </div>
            <div style="margin-bottom:12px">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:11px;color:#555">Vote threshold</span>
                <span style="font-size:11px;color:#4ecdc4">{{ ppVoteThreshold.toFixed(2) }}</span>
              </div>
              <input type="range" min="0.1" max="0.9" step="0.01" :value="ppVoteThreshold" @input="e => ppVoteThreshold = parseFloat(e.target.value)" style="width:100%;accent-color:#4ecdc4" />
            </div>
          </template>

          <!-- Min Confidence -->
          <div style="margin-bottom:12px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
              <span style="font-size:11px;color:#555">Min confidence</span>
              <span style="font-size:11px;color:#4ecdc4">{{ confidenceThreshold.toFixed(2) }}</span>
            </div>
            <input type="range" min="0" max="1" step="0.01" :value="confidenceThreshold"
              @input="e => confidenceThreshold = parseFloat(e.target.value)"
              style="width:100%;accent-color:#4ecdc4" />
          </div>

          <button class="btn" :disabled="ppRunning" style="display:none" />
          <!-- Status indicator replaces Run button -->
          <div style="display:flex;align-items:center;gap:8px;padding:4px 0;min-height:26px">
            <span v-if="ppRunning" style="font-size:11px;color:#888">⟳ Processing…</span>
            <span v-else-if="ppError" style="font-size:11px;color:#ff6b6b;cursor:help" :title="ppError">✗ Error</span>
            <span v-else-if="ppResult" style="font-size:11px;color:#4ecdc4">✓ Done</span>
            <span v-else style="font-size:11px;color:#333">Waiting for case…</span>
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
            <div v-else style="text-align:center;padding:20px;color:#444">No prediction frame</div>
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
            <div style="font-size:14px;color:#333;margin-bottom:4px">
              No video loaded —
              using frame simulation
            </div>
            <div class="sim-time">{{ formatTime(currentTime) }}</div>
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

        <!-- Resize handle: video ↔ raster -->
        <div class="resize-handle resize-handle--row" @mousedown="startResize('raster', $event)"></div>

        <!-- Class labels + Raster -->
        <div style="display:flex;border-top:1px solid #1a1a24;flex-shrink:0">
          <div style="width:120px;flex-shrink:0;background:#08080e;border-right:1px solid #1a1a24">
            <div
              v-for="(item, displayIdx) in displayedClasses"
              :key="item.idx"
              class="raster-label"
              :style="{
                height: rasterLabelHeight,
                color: enabledClasses.has(item.idx) ? '#999' : '#333',
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
                  background: enabledClasses.has(item.idx) ? CLASS_COLORS[item.idx % CLASS_COLORS.length] : '#222'
                }"
              />
              <span style="overflow:hidden;white-space:nowrap;text-overflow:ellipsis">{{ item.cls }}</span>
            </div>
          </div>

          <div style="flex:1;position:relative">
            <canvas
              ref="rasterRef"
              :style="{ width: '100%', height: rasterHeight + 'px', display: 'block', cursor: 'crosshair' }"
              @click="handleRasterClick"
              @mousemove="handleRasterMouseMove"
              @mousedown="handleRasterMouseDown"
              @mouseleave="hoveredFrame = null"
              @wheel.prevent="handleWheel"
            />
            <div v-if="hoveredFrame !== null" class="hover-tooltip">
              Frame {{ hoveredFrame }} ·
              {{ formatTime(hoveredFrame / data.fps) }}
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

      <!-- Resize handle: main ↔ right panel -->
      <div class="resize-handle resize-handle--col" @mousedown="startResize('right', $event)"></div>

      <!-- ── Right Stats Panel ──────────────────────────────────────────── -->
      <div class="right-panel" :style="{ display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '0', width: rightPanelWidth + 'px' }">

        <!-- Classes (top, scrollable) -->
        <div style="flex:1;min-height:0;overflow-y:auto;padding:10px 14px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <span class="section-label" style="margin-bottom:0">Classes</span>
            <div class="custom-dropdown">
              <span class="section-label" style="margin-bottom:0;margin-right:4px">Sort:</span>
              <div class="dropdown-trigger" @click="showSortDropdown = !showSortDropdown">
                <span class="dropdown-value">{{ classSortMode }}</span>
                <span class="dropdown-chevron">▾</span>
              </div>
              <div v-if="showSortDropdown" class="dropdown-menu">
                <div class="dropdown-item" @click="classSortMode = 'default'; showSortDropdown = false">Default</div>
                <div class="dropdown-item" @click="classSortMode = 'frequency'; showSortDropdown = false">Frequency</div>
                <div class="dropdown-item" @click="classSortMode = 'alphabetical'; showSortDropdown = false">Alphabetical</div>
                <div class="dropdown-item" @click="classSortMode = 'custom'; showSortDropdown = false">Custom</div>
              </div>
            </div>
          </div>
          <div
            v-for="(item, displayIdx) in displayedClasses"
            :key="item.idx"
            class="class-row"
            :class="{ 'class-row--dragging': draggingClassIdx === displayIdx }"
            :style="{
              background: enabledClasses.has(item.idx) ? '#0e0e1a' : 'transparent',
              opacity: classStats[item.idx]?.count === 0 ? 0.25 : (enabledClasses.has(item.idx) ? 1 : 0.35),
              cursor: classSortMode === 'custom' ? 'move' : 'pointer',
              pointerEvents: classStats[item.idx]?.count === 0 ? 'none' : undefined,
            }"
            :draggable="classSortMode === 'custom'"
            @click="guardedToggleClass(item.idx)"
            @dblclick.stop="handleClassDoubleClick(item.idx)"
            @dragstart="handleDragStart($event, displayIdx)"
            @dragover.prevent="handleDragOver($event, displayIdx)"
            @drop="handleDrop($event, displayIdx)"
            @dragend="handleDragEnd"
          >
            <div v-if="classSortMode === 'custom'" style="margin-right:6px;color:#666;cursor:grab;user-select:none" @mousedown.stop>⋮⋮</div>
            <div class="class-dot" :style="{ background: classStats[item.idx]?.count === 0 ? '#333' : CLASS_COLORS[item.idx % CLASS_COLORS.length] }" />
            <div style="flex:1;min-width:0">
              <div class="class-name">{{ item.cls }}</div>
              <div style="display:flex;gap:6px;align-items:baseline;flex-wrap:wrap">
                <div v-if="classStats[item.idx]?.count === 0" class="class-stat" style="color:#444;font-style:italic">
                  no detections
                </div>
                <div v-else-if="classStats[item.idx]" class="class-stat">
                  {{ classStats[item.idx].pct.toFixed(0) }}% frames
                </div>
                <div v-if="filteredSummary?.onset_count?.[item.cls] != null" style="font-size:11px;color:#888">
                  {{ filteredSummary.onset_count[item.cls] }}× onset
                </div>
                <div v-if="filterMode === 'filtered' && filteredSummary?.class_time_sec?.[item.cls] != null" style="font-size:11px;color:#4ecdc4">
                  {{ filteredSummary.class_time_sec[item.cls].toFixed(1) }}s
                </div>
              </div>
              <div v-if="enabledClasses.has(item.idx) && classStats[item.idx]" style="width:100%;height:24px;margin-top:6px">
                <svg width="100%" height="24" viewBox="0 0 40 24" preserveAspectRatio="none">
                  <rect
                    v-for="(p, i) in sparklines[item.idx]"
                    :key="i"
                    :x="i * 2"
                    :y="24 - p * 24"
                    width="1.5"
                    :height="p * 24"
                    :fill="CLASS_COLORS[item.idx % CLASS_COLORS.length]"
                    opacity="0.7"
                  />
                </svg>
              </div>
            </div>
          </div>
        </div>

        <!-- Transitions (filtered mode, when summary available) -->
        <div v-if="transitionMatrix"
          style="flex-shrink:0;border-top:1px solid #1a1a24;padding:10px 14px;max-height:40%;overflow-y:auto">
          <div class="section-label" style="margin-bottom:8px">Transitions</div>
          <div style="display:flex">
            <!-- Row labels -->
            <div :style="{ display:'flex', flexDirection:'column', justifyContent:'flex-end', marginRight:'4px' }">
              <div :style="{ height: transitionMatrix.cellSize + 'px' }" v-for="cls in transitionMatrix.classes" :key="'rl-'+cls"
                style="display:flex;align-items:center;justify-content:flex-end;font-size:9px;color:#666;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;max-width:60px"
                :title="cls">{{ cls }}</div>
            </div>
            <div>
              <!-- Column labels -->
              <div style="display:flex;margin-left:0">
                <div v-for="cls in transitionMatrix.classes" :key="'cl-'+cls"
                  :style="{ width: transitionMatrix.cellSize + 'px', textAlign:'center', fontSize:'9px', color:'#666', overflow:'hidden', whiteSpace:'nowrap', textOverflow:'ellipsis' }"
                  :title="cls">{{ cls.slice(0,3) }}</div>
              </div>
              <!-- Matrix grid -->
              <div v-for="(row, ri) in transitionMatrix.grid" :key="ri" style="display:flex">
                <div v-for="(cell, ci) in row" :key="ci"
                  :style="{
                    width: transitionMatrix.cellSize + 'px',
                    height: transitionMatrix.cellSize + 'px',
                    background: cell.count > 0
                      ? `rgba(78, 205, 196, ${0.15 + 0.85 * cell.intensity})`
                      : '#0c0c16',
                    border: '1px solid #1a1a24',
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
  ref, computed, watch, watchEffect, onMounted, onUnmounted, nextTick, shallowRef,
} from "vue";
import { CLASS_COLORS, formatTime } from "../utils/index.js";

defineProps({ username: { type: String, default: "" } });
const emit = defineEmits(["logout"]);

// ── Constants ──────────────────────────────────────────────────────────────
const RATES = [0.25, 0.5, 1, 2, 4];
const SPARKLINE_BINS = 20;
const PP_METHODS = ['run_length', 'majority_vote', 'gaussian'];
const JUMP_FILTERS = [
  { value: 'none',    label: 'Off' },
  { value: 'any',     label: 'Any' },
  { value: 'class',   label: 'Class' },
  { value: 'onset',   label: 'Onset' },
  { value: 'offset',  label: 'Offset' },
  { value: 'changed', label: 'Changed' },
];

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
const classSortMode     = ref('custom');  // 'default' | 'frequency' | 'alphabetical' | 'custom'
const customOrder       = ref([]);  // array of class indices (empty = not initialized)
const draggingClassIdx  = ref(null);  // currently dragging class index
const showSortDropdown  = ref(false);  // sort dropdown visibility

// ── Panel resizing state ──────────────────────────────────────────────────
const leftPanelWidth    = ref(280);
const rightPanelWidth   = ref(280);
const videoFlex         = ref(1);     // flex-grow of video area
const rasterHeight      = ref(200);   // px height of raster canvas

// ── Jump-filter state ─────────────────────────────────────────────────────
const jumpFilter        = ref('none'); // 'none'|'any'|'class'|'onset'|'changed'
const jumpFilterClassIds = ref(new Set()); // class indices when jumpFilter==='class'

// ── Post-process state ─────────────────────────────────────────────────────────────
const filterMode        = ref('raw');        // 'raw' | 'filtered'
const ppMethod          = ref('run_length');
const ppMinDuration     = ref(1.0);
const ppGapFill         = ref(1.0);
const ppWindowSec       = ref(3.0);
const ppVoteThreshold   = ref(0.5);
const ppRunning         = ref(false);
const ppError           = ref(null);
const ppResult          = ref(null);
const filterInfo        = ref(null);
const rawFrameSet       = shallowRef(null);  // Set<number> of raw frame indices
const filteredSummary   = ref(null);          // class_time_sec etc from filtered_summary.json

const timeInputError    = ref(null);  // shown near time input when format is invalid

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

const currentPartRawUrl = computed(() => {
  if (!activeCaseName.value) return null;
  let source = frameMap.value.get(currentFrame.value)?.source ?? null;
  if (!source) {
    const parts = data.value?.parts;
    if (parts) {
      for (const p of parts) {
        if (currentFrame.value >= p.startFrame && currentFrame.value <= p.endFrame) {
          source = p.source;
          break;
        }
      }
    }
    if (!source) {
      let bestStart = -1;
      for (const [src, sf] of partStartFrame.value) {
        if (sf <= currentFrame.value && sf > bestStart) {
          bestStart = sf;
          source = src;
        }
      }
    }
  }
  if (!source) return null;
  const partName = source.split('/').pop().replace(/\.mp4$/i, '');
  return `/api/raw/${activeCaseName.value}/${partName}.mp4`;
});

const currentPartPredictionFrameUrl = computed(() => {
  if (!activeCaseName.value) return null;

  // Determine which source (clip) this frame belongs to.
  // Prefer a direct detection entry; fall back to the parts boundary table
  // (present when detections.json was flat-format) or the first-detection map.
  let source = frameMap.value.get(currentFrame.value)?.source ?? null;
  let actualStartFrame = null;

  if (!source) {
    // Use server-provided part boundaries when available (most accurate).
    const parts = data.value?.parts;
    if (parts) {
      for (const p of parts) {
        if (currentFrame.value >= p.startFrame && currentFrame.value <= p.endFrame) {
          source = p.source;
          actualStartFrame = p.startFrame;
          break;
        }
      }
    }
    // Fallback: scan partStartFrame for the largest start ≤ currentFrame.
    if (!source) {
      let bestStart = -1;
      for (const [src, sf] of partStartFrame.value) {
        if (sf <= currentFrame.value && sf > bestStart) {
          bestStart = sf;
          source = src;
        }
      }
    }
  }

  if (!source) return null;

  // Resolve the actual part start frame (0-based global index of clip frame 0).
  if (actualStartFrame === null) {
    const parts = data.value?.parts;
    if (parts) {
      const p = parts.find(p => p.source === source);
      actualStartFrame = p ? p.startFrame : (partStartFrame.value.get(source) ?? 0);
    } else {
      actualStartFrame = partStartFrame.value.get(source) ?? 0;
    }
  }

  const partName = source.split('/').pop().replace(/\.mp4$/i, '');
  // Prediction frames are saved 1-indexed by Ultralytics (clip frame 0 → _1.jpg).
  const localFrame = currentFrame.value - actualStartFrame + 1;
  return `/data/predictions/${activeCaseName.value}/${partName}_frames/${partName}_${localFrame}.jpg`;
});

// In prediction mode we show frames via <img>; no video element is needed.
// currentPartVideoUrl returns null in prediction mode so the video watcher is a no-op.
const currentPartVideoUrl = computed(() =>
  videoMode.value === 'prediction' ? null : currentPartRawUrl.value
);

const currentPartTimestamp = computed(() => {
  const entry = frameMap.value.get(currentFrame.value);
  if (!entry) return currentFrame.value;
  const fps = data.value.fps;
  const startTs = partStartTs.value.get(entry.source) ?? 0;
  return entry.frame / fps - startTs;
});

const currentPartName = computed(() => {
  const frame = currentFrame.value;
  const entry = frameMap.value.get(frame);
  if (entry) {
    return entry.source?.split('/').pop().replace(/\.mp4$/i, '') ?? null;
  }
  // Fallback: scan partStartFrame for the largest start frame ≤ currentFrame
  let bestSource = null, bestStart = -1;
  for (const [source, startFrame] of partStartFrame.value) {
    if (startFrame <= frame && startFrame > bestStart) {
      bestStart = startFrame;
      bestSource = source;
    }
  }
  return bestSource ? bestSource.split('/').pop().replace(/\.mp4$/i, '') : null;
});

// Sparse lookup: frame number → result entry
const frameMap = computed(() => {
  if (!data.value) return new Map();
  return new Map(data.value.results.map(r => [r.frame, r]));
});

// Per-part start timestamp (global ts of the first frame in each part file).
// Derived from frame number + fps since no timestamp is stored in detections.json.
const partStartTs = computed(() => {
  if (!data.value) return new Map();
  const fps = data.value.fps;
  const map = new Map(); // source path → start timestamp
  for (const r of data.value.results) {
    if (!map.has(r.source)) map.set(r.source, r.frame / fps);
  }
  return map;
});

// Per-part start frame number (global frame index of the first frame in each part).
const partStartFrame = computed(() => {
  if (!data.value) return new Map();
  const map = new Map(); // source path → start frame number
  for (const r of data.value.results) {
    if (!map.has(r.source)) map.set(r.source, r.frame);
  }
  return map;
});

// Start timestamp of whichever part is currently loaded in the video element.
// Derived from currentFrame so it works correctly during part switches.
// When currentFrame has no detection entry (sparse map), scan partStartFrame to find
// whichever part's start frame is the largest value ≤ currentFrame.
const currentPartStartTs = computed(() => {
  const frame = currentFrame.value;
  const entry = frameMap.value.get(frame);
  if (entry) return partStartTs.value.get(entry.source) ?? 0;
  let bestSource = null, bestStart = -1;
  for (const [source, startFrame] of partStartFrame.value) {
    if (startFrame <= frame && startFrame > bestStart) {
      bestStart = startFrame;
      bestSource = source;
    }
  }
  return bestSource ? (partStartTs.value.get(bestSource) ?? 0) : 0;
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

const sortedClassStats = computed(() =>
  classStats.value.filter(s => s.count > 0).sort((a, b) => b.pct - a.pct)
);

// Cases organized by date (parsed from directory name like 20251113_02)
const DATE_CASE_RE = /^(\d{4})(\d{2})(\d{2})_(\d+)$/;
const groupedCases = computed(() => {
  if (!cases.value || cases.value.length === 0) return [];
  const dateMap = new Map();  // "YYYY/MM/DD" → [{name, display, sortKey}]
  const others = [];
  for (const c of cases.value) {
    const m = DATE_CASE_RE.exec(c);
    if (m) {
      const label = `${m[1]}/${m[2]}/${m[3]}`;
      if (!dateMap.has(label)) dateMap.set(label, []);
      dateMap.get(label).push({
        name: c,
        display: `Case ${parseInt(m[4], 10).toString().padStart(2, '0')}`,
        sortKey: parseInt(m[4], 10),
      });
    } else {
      others.push({ name: c, display: c, sortKey: c });
    }
  }
  // Sort cases within each date group by case number
  for (const arr of dateMap.values()) arr.sort((a, b) => a.sortKey - b.sortKey);
  // Sort date groups newest first
  const groups = [...dateMap.entries()]
    .sort((a, b) => b[0].localeCompare(a[0]))
    .map(([label, cs]) => ({ label, cases: cs }));
  if (others.length > 0) groups.push({ label: 'Others', cases: others });
  return groups;
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
  // Cell size adapts to number of classes (min 20, max 32)
  const cellSize = Math.max(20, Math.min(32, Math.floor(200 / classes.length)));
  return { classes, grid, cellSize };
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
    rawFrameSet.value = new Set(parsed.results.map(r => r.frame));
    filteredSummary.value = null;
    filterInfo.value = null;
    filterMode.value = 'raw';
    ppResult.value = null;
    ppError.value = null;
    activeCaseName.value = caseName;
    enabledClasses.value = new Set(parsed.classes.map((_, i) => i));
    jumpFilterClassIds.value = new Set();
    // Initialize custom order if not matching or empty
    if (customOrder.value.length !== parsed.classes.length) {
      customOrder.value = parsed.classes.map((_, i) => i);
    }
    showPicker.value = false;
    videoSrc.value = null;
    currentFrame.value = 0;
    nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0));
  } catch (err) {
    alert(`Failed to load case "${caseName}": ${err.message}`);
  }
}

async function reloadData(caseName, mode) {
  const filename = mode === 'filtered' ? 'filtered_detections.json' : 'detections.json';
  const res = await fetch(`/data/predictions/${caseName}/${filename}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} — no filtered data yet (run filter first)`);
  const parsed = await res.json();
  data.value = parsed;
  filterInfo.value = parsed._filter ?? null;
  if (mode === 'raw') {
    rawFrameSet.value = new Set(parsed.results.map(r => r.frame));
  } else {
    try {
      const sumRes = await fetch(`/data/predictions/${caseName}/filtered_summary.json`);
      if (sumRes.ok) filteredSummary.value = await sumRes.json();
    } catch {}
  }
}

async function runPostprocess() {
  if (!activeCaseName.value) return;
  ppRunning.value = true;
  ppError.value = null;
  try {
    const body = {
      caseName: activeCaseName.value,
      method: ppMethod.value,
      min_duration_sec: ppMinDuration.value,
      gap_fill_sec: ppGapFill.value,
      window_sec: ppWindowSec.value,
      vote_threshold: ppVoteThreshold.value,
      confidence_threshold: confidenceThreshold.value,
    };
    const res = await fetch('/api/postprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const result = await res.json();
    if (!result.ok) throw new Error(result.error);
    ppResult.value = result;
    filteredSummary.value = result.summary ?? null;
    // Switch to filtered view, then reload; if already filtered just reload
    if (filterMode.value !== 'filtered') {
      filterMode.value = 'filtered'; // watch fires reloadData
    } else {
      await reloadData(activeCaseName.value, 'filtered');
    }
  } catch (err) {
    ppError.value = err.message;
  } finally {
    ppRunning.value = false;
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

function newSession() {
  showPicker.value = true;
  data.value = null;
  videoSrc.value = null;
  activeCaseName.value = null;
}

function seekToFrame(frame) {
  if (!data.value) return;
  // Capture the current part's start timestamp BEFORE updating currentFrame,
  // because currentPartStartTs depends on currentFrame (via frameMap lookup).
  const prevPartStartTs = currentPartStartTs.value;
  currentFrame.value = frame;
  scheduleDraws(1);  // Immediately redraw overlay with new frame's detections
  if (videoRef.value && videoSrc.value) {
    const entry = frameMap.value.get(frame);
    const fps = data.value.fps;
    if (entry) {
      const startTs = partStartTs.value.get(entry.source) ?? 0;
      videoRef.value.currentTime = entry.frame / fps - startTs;
    } else {
      // Frame has no detection — seek within the same part using captured start ts
      videoRef.value.currentTime = frame / fps - prevPartStartTs;
    }
  }
}

function handleFrameInput(value) {
  if (!data.value) return;
  const frameNum = parseInt(value, 10);
  if (isNaN(frameNum)) return;
  // Clamp to valid range (1-indexed display, 0-indexed internal)
  const targetFrame = Math.max(1, Math.min(data.value.total_frames, frameNum)) - 1;
  seekToFrame(targetFrame);
}

// Parse a time string and seek to the nearest frame.
// Supported formats:
//   H:MM:SS or H:MM:SS.ms   → hours:minutes:seconds
//   M:SS or M:SS.ms         → minutes:seconds
//   <number>                → interpreted as minutes (decimals = fractional minutes,
//                             e.g. "30" → 30:00, "30.5" → 30:30)
// Shows timeInputError for unrecognisable input.
function handleTimeInput(value) {
  if (!data.value) return;
  const str = value.trim();
  if (!str) return;
  let totalSec = NaN;
  // Try H:MM:SS.ms or H:MM:SS
  const hms = str.match(/^(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?$/);
  if (hms) {
    totalSec = parseInt(hms[1], 10) * 3600 + parseInt(hms[2], 10) * 60 + parseInt(hms[3], 10);
    if (hms[4]) totalSec += parseFloat('0.' + hms[4]);
  } else {
    // Try M:SS.ms or M:SS
    const ms = str.match(/^(\d+):(\d{1,2})(?:\.(\d+))?$/);
    if (ms) {
      totalSec = parseInt(ms[1], 10) * 60 + parseInt(ms[2], 10);
      if (ms[3]) totalSec += parseFloat('0.' + ms[3]);
    } else {
      // Bare number: treat as minutes (decimals = fractional minutes).
      // e.g. "30" → 1800 s, "30.5" → 1830 s
      const bare = parseFloat(str);
      if (!isNaN(bare) && /^[\d.]+$/.test(str)) {
        totalSec = bare * 60;
      }
    }
  }
  if (isNaN(totalSec) || totalSec < 0) {
    timeInputError.value = 'Invalid format — use M:SS, H:MM:SS, or a number (minutes)';
    setTimeout(() => { timeInputError.value = null; }, 3000);
    return;
  }
  timeInputError.value = null;
  const fps = data.value.fps;
  const frame = Math.round(totalSec * fps);
  const clamped = Math.max(0, Math.min(data.value.total_frames - 1, frame));
  seekToFrame(clamped);
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

function toggleAllClasses() {
  if (!data.value) return;
  enabledClasses.value = enabledClasses.value.size === data.value.classes.length
    ? new Set()
    : new Set(data.value.classes.map((_, i) => i));
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

function handleClickOutside(e) {
  if (!showSortDropdown.value) return;
  const dropdown = e.target.closest('.custom-dropdown');
  if (!dropdown) {
    showSortDropdown.value = false;
  }
}

function onZoomInput(e) {
  zoomLevel.value = parseFloat(e.target.value);
  panOffset.value = Math.min(panOffset.value, 1 - 1 / zoomLevel.value);
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

  // Build class index to display row mapping
  const clsToRow = classIdxToRowIdx.value;

  // Only iterate sparse entries in the visible range
  for (const [f, row] of sparse) {
    if (f < startFrame || f >= endFrame) continue;
    const x = (f - startFrame) * pxPerFrame;
    const barW = Math.max(pxPerFrame, 1);
    for (let c = 0; c < numClasses; c++) {
      if (row[c] > 0) {
        const displayRow = clsToRow.get(c) ?? c;  // fallback to original index
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

  // Changed-frame markers (filtered mode) — 2-px strip at raster bottom
  const cf = changedFrames.value;
  if (cf) {
    ctx.fillStyle = 'rgba(255, 210, 50, 0.85)';
    for (const f of cf) {
      if (f < startFrame || f >= endFrame) continue;
      const x = (f - startFrame) * pxPerFrame;
      ctx.fillRect(x, H - 2, Math.max(pxPerFrame, 1), 2);
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
  ctx.strokeStyle = "#ff6b6b";
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
    await reloadData(activeCaseName.value, newMode);
  } catch (err) {
    filterMode.value = oldMode;  // revert toggle
    ppError.value = `Could not load ${newMode} detections: ${err.message}`;
  }
});

// ── Auto-run postprocess ───────────────────────────────────────────────────
let _ppDebounceId = null;
function schedulePostprocess(delay = 800) {
  clearTimeout(_ppDebounceId);
  if (!activeCaseName.value) return;
  _ppDebounceId = setTimeout(() => runPostprocess(), delay);
}

// Debounce on param changes (800 ms lets sliders settle)
watch(
  [ppMethod, ppMinDuration, ppGapFill, ppWindowSec, ppVoteThreshold],
  () => schedulePostprocess(800),
);
// Confidence threshold uses shorter delay so scrubbing the slider feels responsive
watch(confidenceThreshold, () => schedulePostprocess(300));

// Immediate run when a new case is loaded (short delay to let data settle)
watch(activeCaseName, (name) => {
  if (name) schedulePostprocess(300);
});

// Single watcher for raster + minimap (they share most deps)
watch(
  [filteredSparseMap, changedFrames, () => currentFrame.value, () => zoomLevel.value, () => panOffset.value, () => hoveredFrame.value],
  () => scheduleDraws(6),  // 2 | 4 = raster + minimap
  { flush: "post" }
);

// Redraw raster when display order changes
watch(displayedClasses, () => {
  scheduleDraws(2);  // 2=raster
}, { flush: "post" });

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
    // Initialize custom order if not matching or empty
    if (customOrder.value.length !== parsed.classes.length) {
      customOrder.value = parsed.classes.map((_, i) => i);
    }
    showPicker.value = false;
  } catch (err) {
    alert(`Failed to load ${url}: ${err.message}`);
  }
}

onMounted(() => {
  window.addEventListener("mousemove", onGlobalMouseMove);
  window.addEventListener("mouseup", onGlobalMouseUp);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("click", handleClickOutside);

  // Load custom order from localStorage
  const saved = localStorage.getItem('yolo-visualizer-custom-order');
  if (saved) {
    try {
      customOrder.value = JSON.parse(saved);
    } catch (e) {
      console.warn('Failed to parse custom order from localStorage:', e);
    }
  }

  const params = new URLSearchParams(window.location.search);
  const dataUrl = params.get("data");
  if (dataUrl) loadFromUrl(dataUrl);

  loadCases();
});

onUnmounted(() => {
  window.removeEventListener("mousemove", onGlobalMouseMove);
  window.removeEventListener("mouseup", onGlobalMouseUp);
  window.removeEventListener("keydown", onKeyDown);
  window.removeEventListener("click", handleClickOutside);
  if (_rafId) cancelAnimationFrame(_rafId);
  // Cancel pending video frame callback if active
  if (videoRef.value && typeof videoRef.value.cancelVideoFrameCallback === 'function') {
    videoRef.value.cancelVideoFrameCallback(animFrameRef.value);
  }
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
.date-group { width: 100%; margin-bottom: 20px; }
.date-group-label {
  font-size: 13px; font-weight: 600; color: #4ecdc4; letter-spacing: 1px;
  text-transform: none; margin-bottom: 10px; padding-bottom: 4px;
  border-bottom: 1px solid #1a1a25;
}
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
  border-right: 1px solid #1a1a24; overflow-y: auto;
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
.btn-rate { flex: 1; min-width: 0; font-size: 11px; background: transparent; white-space: nowrap; text-align: center; padding: 8px 4px; }
.btn-rate--active { background: #4ecdc422; border-color: #4ecdc444; }

.time-display {
  padding: 12px; background: #0c0c16; border-radius: 8px; margin-bottom: 20px;
  border: 1px solid #1a1a24; text-align: center;
}
.time-value { font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums; }
.time-input {
  font-size: 28px; font-weight: 700; font-variant-numeric: tabular-nums;
  background: transparent; border: none; border-bottom: 1px solid transparent;
  color: inherit; font-family: inherit; text-align: center; width: 100%;
  outline: none; padding: 0;
}
.time-input:hover { border-bottom-color: #333; }
.time-input:focus { border-bottom-color: #4ecdc4; }
.time-input-error {
  font-size: 11px; color: #ff6b6b; margin-top: 4px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.time-sub { font-size: 13px; color: #555; margin-top: 4px; }
.frame-input {
  width: 60px;
  background: transparent;
  border: none;
  border-bottom: 1px solid #333;
  color: #4ecdc4;
  font-family: monospace;
  font-size: 14px;
  text-align: center;
  padding: 2px 4px;
  outline: none;
  -webkit-appearance: none;
  -moz-appearance: textfield;
}
.frame-input::-webkit-inner-spin-button,
.frame-input::-webkit-outer-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
.frame-input:focus {
  border-bottom-color: #4ecdc4;
}
.text-btn {
  font-size: 12px; color: #666; background: none; border: none;
  cursor: pointer; font-family: inherit; padding: 2px 6px;
}

.custom-dropdown {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.dropdown-trigger {
  display: inline-flex;
  align-items: center;
  cursor: pointer;
  user-select: none;
}

.dropdown-value {
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #555;
  transition: color 0.15s;
}

.dropdown-trigger:hover .dropdown-value {
  color: #777;
}

.dropdown-chevron {
  margin-left: 4px;
  font-size: 10px;
  color: #555;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  right: 0;
  margin-top: 4px;
  background: #0c0c16;
  border: 1px solid #2a2a35;
  border-radius: 4px;
  padding: 4px 0;
  z-index: 1000;
  min-width: 120px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.dropdown-item {
  padding: 6px 12px;
  font-size: 12px;
  font-family: 'JetBrains Mono', monospace;
  color: #999;
  cursor: pointer;
  transition: background 0.15s;
  white-space: nowrap;
}

.dropdown-item:hover {
  background: #1a1a24;
  color: #ccc;
}

.class-row {
  display: flex; align-items: center; gap: 8px; padding: 7px 8px;
  margin-bottom: 2px; border-radius: 6px; cursor: pointer; transition: all 0.15s;
}
.class-row--dragging {
  opacity: 0.5;
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
  border-left: 1px solid #1a1a24; overflow: hidden;
  background: #08080e; flex-shrink: 0; display: flex; flex-direction: column;
}
.resize-handle { flex-shrink: 0; z-index: 10; }
.resize-handle--col {
  width: 5px; cursor: col-resize; background: transparent;
  transition: background 0.15s;
}
.resize-handle--col:hover, .resize-handle--col:active { background: rgba(78,205,196,0.18); }
.resize-handle--row {
  height: 5px; cursor: row-resize; background: transparent;
  transition: background 0.15s;
}
.resize-handle--row:hover, .resize-handle--row:active { background: rgba(78,205,196,0.18); }
.det-card {
  padding: 10px; margin-bottom: 6px; border-radius: 8px;
  background: #0c0c16; border: 1px solid #1a1a24;
}
.det-dot { width: 8px; height: 8px; border-radius: 2px; }
</style>