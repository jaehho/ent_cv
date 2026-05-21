// Case data — owns the fetched detections, the video readiness flags, and
// the derived per-frame / per-part lookups.
//
// On case load this composable fetches BOTH raw detections (required) and
// filtered detections (optional — exists only when postprocess was run via
// CLI). Raw is ALWAYS the primary view — it's the model's actual output and
// what the user is most interested in seeing. Filtered, when present, is
// layered on as an annotation indicating which raw detections survived
// postprocessing (drawn as a thin teal inset stroke by the overlay canvas).
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
  // `data` is the primary view: raw detections, always. The UI's bars, stats,
  // jump frames, class percentages, and labeled bounding boxes all read from
  // here, because the page is about visualizing the model's behavior.
  const data            = shallowRef(null);

  // Filtered results, populated only when filtered_detections.json exists on
  // disk. Drawn as a small teal inset stroke on each filtered bbox to mark
  // "kept by filter". `null` when no filter file exists (nothing to annotate).
  const filteredOverlayResults = shallowRef(null);

  const dataReady       = ref(false);
  const videoSrc        = ref(null);
  const videoReady      = ref(false);
  const activeCaseName  = ref(null);
  const filterInfo      = ref(null);             // _filter block from filtered_detections.json (null when no filter)
  const filteredFrameSet = shallowRef(null);     // Set<number> of frames present in the filtered file (drives "Changed" jump filter)
  const filteredSummary = ref(null);             // class_time_sec etc from filtered_summary.json (null when no filter)

  const isLoading = computed(() => !dataReady.value || !videoReady.value);

  // True when a filtered file was loaded (i.e. there's something to annotate
  // raw boxes with on the overlay canvas).
  const hasFilteredOverlay = computed(() => filteredOverlayResults.value !== null);

  // Sparse lookup: frame number → result entry, on the primary (raw) view.
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

    // Raw is ALWAYS primary. Filtered, when present, is the annotation layer.
    // Object.freeze tells Vue's proxy to skip traversing nested arrays.
    data.value = Object.freeze(rawPayload);
    filteredOverlayResults.value = filtPayload ? Object.freeze(filtPayload.results) : null;
    filterInfo.value = filtPayload?._filter ?? null;
    dataReady.value = true;
    if (isPredictionMode) videoReady.value = true;

    filteredSummary.value = summaryPayload;
    activeCaseName.value = caseName;

    // Frame set is over the filtered side now — `changedFrames` over in the
    // visualizer uses it to mark frames where raw and filtered disagree.
    if (filtPayload) {
      buildFrameSetChunked(filtPayload.results).then((set) => {
        filteredFrameSet.value = set;
      });
    } else {
      filteredFrameSet.value = null;
    }

    return data.value;
  }

  return {
    // state
    data, filteredOverlayResults, dataReady, videoSrc, videoReady, activeCaseName,
    filterInfo, filteredFrameSet, filteredSummary, isLoading, hasFilteredOverlay,
    // derived
    frameMap, partStartTs, partStartFrame,
    // actions
    fetchCase,
  };
}
