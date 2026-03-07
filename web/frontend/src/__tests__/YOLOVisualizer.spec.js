/**
 * Unit tests for YOLOVisualizer.vue — BUG-01 and BUG-02
 *
 * BUG-01: Video starts at first detection frame instead of frame 0
 * BUG-02: Canvas overlay shows stale frame after case load
 *
 * Root cause: nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))
 * Fix: nextTick(() => seekToFrame(0))
 *
 * Strategy: Mount the component with heavy stubs. Access internal reactive
 * state via wrapper.getCurrentComponent().setupState (Vue 3 test-utils
 * pattern for script setup components). Note: setupState auto-unwraps refs,
 * so state.currentFrame returns the number directly (not a ref).
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { nextTick } from "vue";
import YOLOVisualizer from "../components/YOLOVisualizer.vue";

// ── Router mock ──────────────────────────────────────────────────────────────
vi.mock("vue-router", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

// ── Fetch mock ───────────────────────────────────────────────────────────────
// results[0].frame = 200 — this is the key: the bug causes seekToFrame(200),
// the fix causes seekToFrame(0).
const MOCK_DETECTIONS = {
  classes: ["instrument"],
  results: [{ frame: 200, detections: [] }],
  total_frames: 1000,
  fps: 30,
  case_name: "test-case",
  video_files: ["video.mp4"],
  part_start_frames: { "video.mp4": 0 },
};

// ── HTMLVideoElement mock ────────────────────────────────────────────────────
Object.defineProperty(window.HTMLVideoElement.prototype, "play", {
  configurable: true,
  value: vi.fn().mockResolvedValue(undefined),
});
Object.defineProperty(window.HTMLVideoElement.prototype, "pause", {
  configurable: true,
  value: vi.fn(),
});

// ── Canvas mock ───────────────────────────────────────────────────────────────
HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
  clearRect: vi.fn(),
  drawImage: vi.fn(),
  fillRect: vi.fn(),
  fillText: vi.fn(),
  strokeRect: vi.fn(),
  beginPath: vi.fn(),
  moveTo: vi.fn(),
  lineTo: vi.fn(),
  stroke: vi.fn(),
  fill: vi.fn(),
  arc: vi.fn(),
  save: vi.fn(),
  restore: vi.fn(),
  scale: vi.fn(),
  translate: vi.fn(),
  measureText: vi.fn(() => ({ width: 50 })),
  createImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
  putImageData: vi.fn(),
  setLineDash: vi.fn(),
}));

// ── ResizeObserver mock ───────────────────────────────────────────────────────
global.ResizeObserver = vi.fn(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}));

// ── requestAnimationFrame mock ────────────────────────────────────────────────
global.requestAnimationFrame = vi.fn((cb) => setTimeout(cb, 0));
global.cancelAnimationFrame = vi.fn();

// ── Helper: mount YOLOVisualizer ─────────────────────────────────────────────
function mountVisualizer(id = "test-case") {
  return mount(YOLOVisualizer, {
    props: { id, username: "testuser" },
    attachTo: document.body,
    global: {
      stubs: {
        teleport: true,
      },
    },
  });
}

// ── Helper: get internal setup state ─────────────────────────────────────────
// In Vue 3 script setup, setupState auto-unwraps refs.
// So state.currentFrame returns the number value directly.
function getSetupState(wrapper) {
  return wrapper.getCurrentComponent().setupState;
}

// ── Helper: get the raw ref (not unwrapped) from setupContext ─────────────────
// We need the raw ref to be able to SET values on it in tests (for case-switch test).
// Access via the component's internal _: setupContext.exposed is undefined for
// script setup without defineExpose. Use the internal proxy refs map instead.
function getRawRef(wrapper, name) {
  const instance = wrapper.getCurrentComponent();
  // In Vue 3 internals, refs live in the setup context's refs map
  return instance.refs?.[name] ?? instance.setupState?.[name];
}

describe("YOLOVisualizer loadCase — BUG-01 and BUG-02", () => {
  beforeEach(() => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ ...MOCK_DETECTIONS }),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts at frame 0, not the first detection frame (200)", async () => {
    // BUG-01: When a case loads with results[0].frame = 200,
    // currentFrame should be 0 after loadCase completes.
    // The buggy code calls seekToFrame(200); the fix calls seekToFrame(0).
    // seekToFrame sets currentFrame.value = frame — so we assert currentFrame === 0.
    const wrapper = mountVisualizer("test-case");

    // Wait for fetch, state updates, and nextTick chain to complete
    await flushPromises();

    const state = getSetupState(wrapper);
    // setupState auto-unwraps: state.currentFrame is the number, not the ref
    expect(state.currentFrame).toBe(0);

    wrapper.unmount();
  });

  it("resets zoomLevel, panOffset, and playbackRate on case switch", async () => {
    // BUG-01: Switching cases must reset all view state refs.
    // The fix adds these resets to loadCase's reset block.
    const wrapper = mountVisualizer("case-one");
    await flushPromises();

    const state = getSetupState(wrapper);

    // Directly mutate the internal component state to simulate the user
    // having zoomed/panned/changed speed while viewing case-one.
    // We access the underlying Vue internal refs via the component instance.
    const internalRefs = wrapper.getCurrentComponent().setupContext?.expose
      ?? wrapper.getCurrentComponent();

    // Use the setupState proxy — assignments go through the ref's .value setter
    // because Proxy traps set() and routes to the ref automatically
    state.zoomLevel = 3;
    state.panOffset = 0.5;
    state.playbackRate = 2;

    // Confirm state was mutated (proves proxy write-through works)
    expect(state.zoomLevel).toBe(3);
    expect(state.panOffset).toBe(0.5);
    expect(state.playbackRate).toBe(2);

    // Now switch to a second case — this calls loadCase("case-two")
    // loadCase is accessible in setupState since it's defined in setup()
    await state.loadCase("case-two");
    await flushPromises();

    // After the switch, the reset block should have restored defaults
    expect(state.zoomLevel).toBe(1);
    expect(state.panOffset).toBe(0);
    expect(state.playbackRate).toBe(1);

    wrapper.unmount();
  });

  it("schedules draw for frame 0 via seekToFrame(0), not seekToFrame(200)", async () => {
    // BUG-02: The canvas overlay draw chain must be seeded with frame 0.
    // seekToFrame(frame) sets currentFrame.value = frame, then calls scheduleDraws(1).
    // After loadCase, currentFrame must be 0 proving the draw was for the correct frame.
    const wrapper = mountVisualizer("test-case");
    await flushPromises();

    const state = getSetupState(wrapper);

    // currentFrame is set by seekToFrame(frame) — it must be 0, not 200
    expect(state.currentFrame).toBe(0);

    wrapper.unmount();
  });
});

describe("YOLOVisualizer loading screen — LOAD-01 and LOAD-02", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("LOAD-01: isLoading is true immediately after mount, before fetch resolves", () => {
    // Use a manually-controlled promise so fetch never resolves during this test.
    let resolveHeld;
    vi.spyOn(global, "fetch").mockReturnValue(
      new Promise((resolve) => {
        resolveHeld = resolve;
      })
    );

    const wrapper = mountVisualizer("test-case");
    const state = getSetupState(wrapper);

    // Synchronous check — fetch has not resolved, so loading must be true
    expect(state.isLoading).toBe(true);

    wrapper.unmount();
    // resolveHeld is intentionally not called — test is done
  });

  it("LOAD-01: isLoading remains true after fetch resolves but before loadedmetadata fires", async () => {
    // Fetch resolves immediately, but jsdom never fires loadedmetadata automatically.
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ ...MOCK_DETECTIONS }),
    });

    const wrapper = mountVisualizer("test-case");
    await flushPromises();

    const state = getSetupState(wrapper);

    // dataReady should be true, but videoReady is still false — so isLoading === true
    expect(state.isLoading).toBe(true);

    wrapper.unmount();
  });

  it("LOAD-02: isLoading becomes false after videoReady is set", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ ...MOCK_DETECTIONS }),
    });

    const wrapper = mountVisualizer("test-case");
    await flushPromises();

    const state = getSetupState(wrapper);

    // Simulate loadedmetadata by writing videoReady via setupState proxy
    state.videoReady = true;

    expect(state.isLoading).toBe(false);

    wrapper.unmount();
  });

  it("LOAD-02: isLoading becomes false in prediction mode once dataReady is set (no video event needed)", async () => {
    vi.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({ ...MOCK_DETECTIONS }),
    });

    const wrapper = mountVisualizer("test-case");
    const state = getSetupState(wrapper);

    // Switch to prediction mode before triggering loadCase
    state.videoMode = "prediction";
    await state.loadCase("test-case");
    await flushPromises();

    // In prediction mode, no video element is used — loading should clear immediately
    expect(state.isLoading).toBe(false);

    wrapper.unmount();
  });
});

// ── TRANS-01 and TRANS-02: ResizeObserver + dynamic square sizing ──────────
const MOCK_DETECTIONS_WITH_SUMMARY = {
  ...MOCK_DETECTIONS,
  classes: ["instrument_a", "instrument_b"],
  results: [
    { frame: 0, detections: [{ class_id: 0, x1: 0, y1: 0, x2: 10, y2: 10, confidence: 0.9 }] },
    { frame: 1, detections: [{ class_id: 1, x1: 0, y1: 0, x2: 10, y2: 10, confidence: 0.9 }] },
  ],
  total_frames: 100,
};

const MOCK_FILTERED_SUMMARY = {
  transition_matrix: {
    instrument_a: { instrument_a: 0, instrument_b: 3 },
    instrument_b: { instrument_a: 2, instrument_b: 0 },
  },
  classes: ["instrument_a", "instrument_b"],
  total_transitions: 5,
};

describe("YOLOVisualizer — TRANS-01 and TRANS-02", () => {
  beforeEach(() => {
    // Reset mock call count between tests
    ResizeObserver.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  function mountWithFilteredSummary() {
    vi.spyOn(global, "fetch").mockImplementation((url) => {
      if (url.includes("filtered-summary")) {
        return Promise.resolve({
          ok: true,
          json: async () => MOCK_FILTERED_SUMMARY,
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ ...MOCK_DETECTIONS_WITH_SUMMARY }),
      });
    });
    return mountVisualizer("test-case");
  }

  it("TRANS-01a: ResizeObserver constructor is called at least once on mount and observe() is called with a DOM element", async () => {
    const wrapper = mountWithFilteredSummary();
    await flushPromises();

    // ResizeObserver should have been constructed
    expect(ResizeObserver).toHaveBeenCalled();

    // observe() should have been called on the instance with a DOM element
    const instance = ResizeObserver.mock.results[ResizeObserver.mock.results.length - 1].value;
    expect(instance.observe).toHaveBeenCalled();
    const observedEl = instance.observe.mock.calls[0][0];
    expect(observedEl).toBeTruthy();
    expect(observedEl instanceof Element).toBe(true);

    wrapper.unmount();
  });

  it("TRANS-01b: disconnect() is called on the observer instance when component unmounts", async () => {
    const wrapper = mountWithFilteredSummary();
    await flushPromises();

    const instance = ResizeObserver.mock.results[ResizeObserver.mock.results.length - 1].value;

    wrapper.unmount();

    expect(instance.disconnect).toHaveBeenCalled();
  });

  it("TRANS-02a: when ResizeObserver fires width=300, transitionMatrix.squareSize equals 300 (below 320 cap)", async () => {
    const wrapper = mountWithFilteredSummary();
    await flushPromises();

    // Set filterMode to filtered and inject filteredSummary so transitionMatrix computed returns non-null
    const state = getSetupState(wrapper);
    state.filterMode = "filtered";
    state.filteredSummary = MOCK_FILTERED_SUMMARY;
    await nextTick();

    // Fire the ResizeObserver callback manually
    const observerCallback = ResizeObserver.mock.calls[ResizeObserver.mock.calls.length - 1][0];
    observerCallback([{ contentRect: { width: 300 } }]);
    await nextTick();

    expect(state.transitionMatrix.squareSize).toBe(300);

    wrapper.unmount();
  });

  it("TRANS-02b: when ResizeObserver fires width=400, transitionMatrix.squareSize is capped at 320", async () => {
    const wrapper = mountWithFilteredSummary();
    await flushPromises();

    const state = getSetupState(wrapper);
    state.filterMode = "filtered";
    state.filteredSummary = MOCK_FILTERED_SUMMARY;
    await nextTick();

    const observerCallback = ResizeObserver.mock.calls[ResizeObserver.mock.calls.length - 1][0];
    observerCallback([{ contentRect: { width: 400 } }]);
    await nextTick();

    expect(state.transitionMatrix.squareSize).toBe(320);

    wrapper.unmount();
  });

  it("TRANS-02c: transitionMatrix.cellSize is always >= 14 regardless of class count", async () => {
    const wrapper = mountWithFilteredSummary();
    await flushPromises();

    const state = getSetupState(wrapper);
    state.filterMode = "filtered";
    state.filteredSummary = MOCK_FILTERED_SUMMARY;
    await nextTick();

    // Fire with a narrow width that would make cells tiny without the floor
    const observerCallback = ResizeObserver.mock.calls[ResizeObserver.mock.calls.length - 1][0];
    observerCallback([{ contentRect: { width: 28 } }]); // 28 / 2 classes = 14 exactly
    await nextTick();

    expect(state.transitionMatrix.cellSize).toBeGreaterThanOrEqual(14);

    wrapper.unmount();
  });
});
