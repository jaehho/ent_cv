// Case data — owns the fetched detections, the video readiness flags, and
// the derived per-frame / per-part lookups.
//
// On case load this composable fetches BOTH raw detections (required) and
// filtered detections (optional — exists only when postprocess was run via
// CLI). When filtered is present, it becomes the "primary" view and the raw
// results are exposed separately as `rawOverlayResults` so the overlay
// canvas can draw them at low opacity beneath the primary boxes.
//
// Callers (currently just YOLOVisualizer.vue) handle any UI-state resets
// that should accompany a case load (enabled classes, current frame, zoom,
// etc.), since those don't belong to this domain.

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
  // `data` is the primary view: filtered detections when they exist on disk,
  // raw detections otherwise. Everything in the UI that displays "the active
  // view" — raster bars, sparklines, jump frames, class panel percentages,
  // bounding-box overlay at full opacity — reads from this.
  const data            = shallowRef(null);

  // Raw results, populated only when filtered is the primary view. The
  // overlay canvas draws these at low opacity beneath the primary boxes so
  // you can see what the filter rejected. `null` when no filter file exists
  // (no overlay needed; `data` already holds raw).
  const rawOverlayResults = shallowRef(null);

  const dataReady       = ref(false);
  const videoSrc        = ref(null);
  const videoReady      = ref(false);
  const activeCaseName  = ref(null);
  const filterInfo      = ref(null);             // _filter block from filtered_detections.json (null when no filter)
  const rawFrameSet     = shallowRef(null);      // Set<number> of raw frame indices (drives "Changed" jump filter)
  const filteredSummary = ref(null);             // class_time_sec etc from filtered_summary.json (null when no filter)

  const isLoading = computed(() => !dataReady.value || !videoReady.value);

  // True when a filtered file was loaded (i.e. there's something to layer
  // beneath the primary view on the overlay canvas).
  const hasFilteredOverlay = computed(() => rawOverlayResults.value !== null);

  // Sparse lookup: frame number → result entry, on the primary view.
  const frameMap = computed(() => {
    if (!data.value) return new Map();
    return new Map(data.value.results.map(r => [r.frame, r]));
  });

  // Per-part start timestamp (global ts of the first frame in each part file).
  // Derived from frame number + fps since no timestamp is stored in detections.json.
  const partStartTs = computed(() => {
    if (!data.value) return new Map();
    const fps = data.value.fps;
    const map = new Map();
    for (const r of data.value.results) {
      if (!map.has(r.source)) map.set(r.source, r.frame / fps);
    }
    return map;
  });

  // Per-part start frame number (global frame index of the first frame in each part).
  const partStartFrame = computed(() => {
    if (!data.value) return new Map();
    const map = new Map();
    for (const r of data.value.results) {
      if (!map.has(r.source)) map.set(r.source, r.frame);
    }
    return map;
  });

  /**
   * Fetch a case. Always fetches raw detections; attempts filtered + summary
   * in parallel and silently falls back to raw-only when no filtered file
   * exists on disk. Throws if even the raw fetch fails; caller surfaces.
   *
   * `isPredictionMode` means the UI shows pre-rendered prediction frames
   * (no <video> element), so video readiness is granted immediately.
   */
  async function fetchCase(caseName, { isPredictionMode = false } = {}) {
    dataReady.value = false;
    videoReady.value = false;

    const [rawRes, filtRes] = await Promise.all([
      fetch(`/api/cases/${caseName}/detections/`),
      fetch(`/api/cases/${caseName}/detections/?mode=filtered`),
    ]);
    if (!rawRes.ok) throw new Error(`HTTP ${rawRes.status}`);
    const rawPayload = await rawRes.json();
    const filtPayload = filtRes.ok ? await filtRes.json() : null;

    // Filter summary in parallel (only meaningful when filtered exists).
    let summaryPayload = null;
    if (filtPayload) {
      try {
        const sumRes = await fetch(`/api/cases/${caseName}/filtered-summary/`);
        if (sumRes.ok) summaryPayload = await sumRes.json();
      } catch { /* summary is optional, ignore */ }
    }

    // Clear video src BEFORE swapping data so the URL watcher fires with a
    // clean slate. Setting data first triggers the watcher early; then
    // nulling videoSrc aborts the load and loadedmetadata never fires,
    // leaving videoReady stuck at false and the loading screen stuck forever.
    videoSrc.value = null;

    // Choose primary view: filtered if it loaded, otherwise raw.
    // Object.freeze tells Vue's proxy to skip traversing nested arrays.
    if (filtPayload) {
      data.value = Object.freeze(filtPayload);
      rawOverlayResults.value = Object.freeze(rawPayload.results);
      filterInfo.value = filtPayload._filter ?? null;
    } else {
      data.value = Object.freeze(rawPayload);
      rawOverlayResults.value = null;
      filterInfo.value = null;
    }
    dataReady.value = true;
    if (isPredictionMode) videoReady.value = true;

    filteredSummary.value = summaryPayload;
    activeCaseName.value = caseName;

    buildFrameSetChunked(rawPayload.results).then((set) => {
      rawFrameSet.value = set;
    });

    return data.value;
  }

  return {
    // state
    data, rawOverlayResults, dataReady, videoSrc, videoReady, activeCaseName,
    filterInfo, rawFrameSet, filteredSummary, isLoading, hasFilteredOverlay,
    // derived
    frameMap, partStartTs, partStartFrame,
    // actions
    fetchCase,
  };
}
