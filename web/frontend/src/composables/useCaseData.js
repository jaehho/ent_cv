// Case data — owns the fetched detections, the video readiness flags, and
// the derived per-frame / per-part lookups.
//
// On case load this composable fetches BOTH raw detections (required) and
// filtered detections (optional — exists only when postprocess was run via
// CLI). The active view (which the overlay paints and stats derive from) is
// chosen by the caller-supplied viewMode ref — defaulting to 'filtered' so
// the post-processed detections (carrying in_use flags etc.) are what the
// user sees first. Flipping to 'raw' shows the model's unfiltered output for
// debugging; this happens without re-fetching since both payloads are kept
// in memory.
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

/**
 * @param {import("vue").Ref<"raw"|"filtered">} [viewMode] - which detection
 *   set the overlay/stats track. Defaults to a local ref initialized to
 *   "filtered" if the caller doesn't supply one.
 */
export function useCaseData(viewMode = ref("filtered")) {
  // Raw payload (Object.frozen). Holds the universal context everyone reads:
  // classes list, total_frames, fps, parts, has_prediction_frames, _filter.
  // The results array inside is the raw side of the detections.
  const data            = shallowRef(null);

  // Filtered detections array (just the results, not the full payload).
  // Populated only when filtered_detections.json exists on disk; null
  // otherwise. When viewMode === "filtered" and this is non-null, the
  // overlay and per-frame stats source from here instead of data.results.
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

  // Sparse lookup: frame number → result entry. Follows viewMode — filtered
  // when the toggle is set and a filtered payload was loaded; otherwise raw.
  // Stays raw if filtered isn't available so the viewer still works on
  // cases without a postprocess run.
  const frameMap = computed(() => {
    if (!data.value) return new Map();
    const useFiltered = viewMode.value === "filtered" && filteredOverlayResults.value;
    const results = useFiltered ? filteredOverlayResults.value : data.value.results;
    return new Map(results.map(r => [r.frame, r]));
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

    // Issue all three requests concurrently. Summary may 404 when no filter
    // was run; .catch maps that to null so it never throws.
    const t0 = performance.now();
    const rawFetch  = fetch(`/api/cases/${caseName}/detections/`);
    const filtFetch = fetch(`/api/cases/${caseName}/detections/?mode=filtered`);
    const sumFetch  = fetch(`/api/cases/${caseName}/filtered-summary/`).catch(() => null);

    const [rawRes, filtRes] = await Promise.all([rawFetch, filtFetch]);
    const tHeaders = performance.now();
    if (!rawRes.ok) throw new Error(`HTTP ${rawRes.status}`);

    // Parse all three bodies in parallel. Each .json() still parses on the main
    // thread, but starting them together lets the browser interleave network
    // reads while parses are queued, instead of waiting body-then-parse twice.
    const [rawPayload, filtPayload, summaryPayload] = await Promise.all([
      rawRes.json(),
      filtRes.ok ? filtRes.json() : Promise.resolve(null),
      sumFetch.then(r => (r && r.ok ? r.json() : null)).catch(() => null),
    ]);
    const tParsed = performance.now();
    // Surface case-switch breakdown so we can see whether network or parse
    // dominates next time the user reports slowness.
    console.info(
      `[case ${caseName}] headers ${(tHeaders - t0).toFixed(0)}ms · parse ${(tParsed - tHeaders).toFixed(0)}ms · total ${(tParsed - t0).toFixed(0)}ms`
    );

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
