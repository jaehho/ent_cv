// Case data — owns the fetched detections, the video readiness flags, the
// filter-mode toggle state, and the derived per-frame / per-part lookups.
//
// Callers (currently just YOLOVisualizer.vue) are responsible for any
// UI-state resets that should accompany a case load (enabled classes,
// current frame, zoom, etc.), since those don't belong to this domain.

import { ref, shallowRef, computed } from "vue";

/**
 * Build a Set of all frame numbers off the main render thread so a long
 * detections list doesn't block the UI on initial case load.
 */
function buildFrameSetChunked(results, chunkSize = 15000) {
  const ric = typeof requestIdleCallback !== "undefined"
    ? requestIdleCallback
    : (cb) => setTimeout(cb, 0);
  return new Promise((resolve) => {
    const frameSet = new Set();
    let i = 0;
    function processChunk() {
      const end = Math.min(i + chunkSize, results.length);
      for (; i < end; i++) frameSet.add(results[i].frame);
      if (i < results.length) ric(processChunk);
      else resolve(frameSet);
    }
    processChunk();
  });
}

export function useCaseData() {
  const data            = shallowRef(null);      // large object — shallowRef avoids deep reactivity
  const dataReady       = ref(false);
  const videoSrc        = ref(null);
  const videoReady      = ref(false);
  const activeCaseName  = ref(null);
  const filterMode      = ref("raw");            // 'raw' | 'filtered'
  const filterInfo      = ref(null);             // _filter block from filtered_detections.json
  const rawFrameSet     = shallowRef(null);      // Set<number> of raw frame indices
  const filteredSummary = ref(null);             // class_time_sec etc from filtered_summary.json

  const isLoading = computed(() => !dataReady.value || !videoReady.value);

  // Sparse lookup: frame number → result entry.
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

  /**
   * Fetch a case's raw detections, reset filter state, kick off the
   * raw-frame-set build off the main render thread. Returns the parsed
   * payload so the caller can do its own post-load setup. Throws on
   * failure; caller should surface to the user.
   *
   * `isPredictionMode` means the UI is showing pre-rendered prediction
   * frames (no <video> element), so video readiness is granted immediately.
   */
  async function fetchCase(caseName, { isPredictionMode = false } = {}) {
    dataReady.value = false;
    videoReady.value = false;

    const res = await fetch(`/api/cases/${caseName}/detections/`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const parsed = await res.json();

    // Clear video src BEFORE updating data so the URL watcher fires
    // with a clean slate. Setting data first triggers the watcher early;
    // then nulling videoSrc aborts the load and loadedmetadata never fires,
    // leaving videoReady stuck at false and the loading screen stuck forever.
    videoSrc.value = null;

    // Deep-freeze to guarantee Vue's proxy skips traversing large nested arrays.
    data.value = Object.freeze(parsed);
    dataReady.value = true;
    if (isPredictionMode) videoReady.value = true;

    filteredSummary.value = null;
    filterInfo.value = null;
    filterMode.value = "raw";
    activeCaseName.value = caseName;

    buildFrameSetChunked(parsed.results).then((set) => {
      rawFrameSet.value = set;
    });

    return parsed;
  }

  /**
   * Reload detections in raw or filtered mode (for the View toggle).
   * Pulls filtered_summary.json alongside when entering filtered mode.
   * Throws on HTTP failure so the caller can revert the toggle.
   */
  async function fetchFilteredView(caseName, mode) {
    const url = `/api/cases/${caseName}/detections/${mode === "filtered" ? "?mode=filtered" : ""}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} — no filtered data yet (run filter first)`);
    const parsed = await res.json();
    data.value = parsed;
    filterInfo.value = parsed._filter ?? null;
    if (mode === "raw") {
      rawFrameSet.value = new Set(parsed.results.map(r => r.frame));
    } else {
      try {
        const sumRes = await fetch(`/api/cases/${caseName}/filtered-summary/`);
        if (sumRes.ok) filteredSummary.value = await sumRes.json();
      } catch { /* leave previous summary in place */ }
    }
  }

  return {
    // state
    data, dataReady, videoSrc, videoReady, activeCaseName,
    filterMode, filterInfo, rawFrameSet, filteredSummary, isLoading,
    // derived
    frameMap, partStartTs, partStartFrame,
    // actions
    fetchCase, fetchFilteredView,
  };
}
