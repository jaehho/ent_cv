# Architecture Research

**Domain:** Vue 3 video player with canvas overlay and responsive matrix plot
**Researched:** 2026-03-07
**Confidence:** HIGH (based on direct codebase inspection + established Vue 3 patterns)

---

## Standard Architecture

### System Overview

```
Browser
├── Vue Router (/cases/:id → YOLOVisualizer.vue, props: true)
│
└── YOLOVisualizer.vue  [single large component, ~1900 lines]
    │
    ├── State layer (component-local reactive refs)
    │   ├── data          shallowRef  — frozen API payload (fps, results, classes…)
    │   ├── currentFrame  ref(0)      — single source of truth for playback position
    │   ├── videoSrc      ref(null)   — current part URL fed to <video>
    │   └── activeCaseName ref(null) — case currently loaded
    │
    ├── Derived layer (computed)
    │   ├── frameMap           Map<frame → result entry>
    │   ├── currentDetections  filtered detections for currentFrame
    │   ├── currentPartVideoUrl → drives video part switching watcher
    │   ├── transitionMatrix   → fixed cellSize derived from panel width constant
    │   └── filteredSparseMap  → drives raster/minimap canvas draws
    │
    ├── Canvas draw layer (imperative, RAF-batched)
    │   ├── drawOverlay()  — bounding boxes on <canvas ref="overlayRef">
    │   ├── drawRaster()   — timeline heatmap on <canvas ref="rasterRef">
    │   └── drawMinimap()  — overview bar on <canvas ref="minimapRef">
    │
    └── DOM layer (template)
        ├── <video ref="videoRef">  playsinline, preload="auto", muted
        ├── <canvas ref="overlayRef">  position:absolute over video
        ├── Transitions matrix  rendered as flex divs with inline cellSize
        └── Right panel  fixed rightPanelWidth px, no ResizeObserver
```

### Component Responsibilities

| Component | Responsibility | Communication |
|-----------|---------------|---------------|
| `CasePicker.vue` | Lists available cases, navigates to `/cases/:id` | Router push |
| `YOLOVisualizer.vue` | All visualization — video, overlays, timelines, stats | Props: `id` (case name); Emits: `logout` |
| `LoginForm.vue` | Session auth form | Emits: `login` |

**Single-component design is intentional.** The PROJECT.md constraint "no new dependencies, prefer CSS and Vue reactive solutions" means decomposition is not in scope. All fixes apply within `YOLOVisualizer.vue`.

---

## Current Data Flow

### Case Load Sequence (with identified bugs)

```
Route /cases/:id
    ↓  props.id received
onMounted → loadCase(props.id)
    ↓
fetch /api/cases/{id}/detections/
    ↓
data.value = Object.freeze(parsed)        ← shallowRef, deep reactivity skipped
currentFrame.value = 0                    ← reset happens here
videoSrc.value = null                     ← cleared here
    ↓
nextTick → seekToFrame(parsed.results[0]?.frame ?? 0)
    ↓  [async, fires after DOM settles]
currentFrame.value = frame                ← set to first detection frame (~40-500)
videoRef.currentTime = ...                ← seeks video to match
    ↓
watch([currentDetections, videoRef, overlayRef]) fires
    ↓
drawOverlay() runs against currentFrame (now ~40-500, not 0)
```

**Bug root cause:** `loadCase` sets `currentFrame.value = 0` synchronously, then
`nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))` sets it to the first
detection frame. This is intentional — it seeks to the first frame that has data.
The bug is that when the video element has not yet loaded metadata, `drawOverlay()`
reads `vid.videoWidth` as 0 and returns early. After metadata loads, the overlay
watcher (`currentDetections, videoRef, overlayRef`) fires and draws correctly — but
`videoRef.value` may not be available yet (video is rendered conditionally on
`videoSrc` being non-null). The watcher fires when `videoRef` becomes non-null
(after `videoSrc` is set by the `currentPartVideoUrl` watcher), but the video
element has no dimensions yet at that instant, so `drawOverlay()` bails.

The `loadedmetadata` guard in the overlay watcher is supposed to handle this:
```js
if (vid && !vid.videoWidth) {
  vid.addEventListener("loadedmetadata", () => scheduleDraws(1), { once: true });
}
```
This only runs when `videoRef` is already non-null when the watcher fires. If the
watcher fires while `videoRef` is still null (before `videoSrc` is set), it takes
the else branch and calls `scheduleDraws(1)` immediately — which is a no-op because
`drawOverlay()` checks `!canvas || !vid` and returns. The `loadedmetadata` listener
is never attached.

### Transitions Matrix — No Responsiveness

```
transitionMatrix computed:
  const cellSize = Math.max(20, Math.min(32, Math.floor(200 / classes.length)));
  // 200 is a hardcoded pixel constant — not derived from panel width
```

`rightPanelWidth` ref exists and changes on drag, but `transitionMatrix` does not
depend on it. Cell sizes are fixed. The panel can narrow below the matrix width,
causing horizontal overflow, not responsive reflow.

---

## Architectural Patterns in This Codebase

### Pattern 1: RAF-Batched Imperative Canvas Draws

**What:** A bitmask flag (`_drawFlags`) accumulates draw requests. A single
`requestAnimationFrame` callback drains them once per frame.

**When to use:** Multiple reactive dependencies (frame number, zoom, filter mode)
would otherwise each trigger redundant redraws in the same tick.

**Current implementation:** `scheduleDraws(flags)` in YOLOVisualizer.vue:593–617.
Flags: 1=overlay, 2=raster, 4=minimap. Watchers call `scheduleDraws` with the
appropriate bitmask.

**Implication for overlay fix:** Adding a new trigger path (e.g. a `ResizeObserver`
on the video wrapper or a `loadedmetadata` event) should call
`scheduleDraws(1)` — it is already the canonical way to request an overlay redraw.

### Pattern 2: shallowRef + Object.freeze for Large Payloads

**What:** Detection data (`data`) is held in a `shallowRef` and deep-frozen after
fetch. Vue's reactivity proxy does not traverse into frozen objects.

**When to use:** API payloads with thousands of nested detection records would cause
significant overhead if Vue recursively made every property reactive.

**Implication:** Consumers must access `data.value.results` directly; computed props
that derive from it re-compute only when `data.value` itself (the reference) changes.

### Pattern 3: Component-Local State with No External Store

**What:** All reactive state lives inside `YOLOVisualizer.vue` as `ref`/`shallowRef`
declarations. No Pinia/Vuex.

**When to use:** Fine for a single-page viewer component where state does not need to
be shared across routes.

**Implication for navigation bug:** Because the component is NOT destroyed on
`/cases/:id → /cases/:id2` navigation (Vue Router reuses the same component instance
when the route path structure is identical), `watch(props.id)` would be needed to
reload state. Currently, the component is only loaded via `onMounted → loadCase(props.id)`.
If navigation goes through CasePicker → back → different case, the route component
IS destroyed/recreated (CasePicker → YOLOVisualizer is a component change). This
means the "stale frame on initial load" is NOT a navigation-between-cases problem —
it is a within-single-load problem with the async seekToFrame timing.

---

## Fix Approaches

### Fix 1: Frame State Initialization (canvas overlay at wrong frame on load)

**Problem:** The overlay draws before the video element has loaded its metadata
(videoWidth === 0), so `drawOverlay()` returns early. By the time metadata loads,
no re-draw is triggered.

**Root cause path:**
1. `loadCase` → `nextTick(seekToFrame)` → sets `currentFrame` → triggers overlay watcher
2. Watcher sees `videoRef` is null (videoSrc not yet set → video not mounted)
3. Watcher calls `scheduleDraws(1)` (else branch); `drawOverlay()` runs, finds `!vid`, returns
4. Later: `currentPartVideoUrl` watcher fires → sets `videoSrc` → video mounts → `videoRef` becomes non-null
5. Watcher fires again for `videoRef` change, `!vid.videoWidth` → attaches `loadedmetadata` listener — THIS should work
6. But: `loadedmetadata` fires → `scheduleDraws(1)` → `drawOverlay()` — should now have dimensions

The frame display showing ~40-500 instead of 0 means `seekToFrame` is being called
with `parsed.results[0].frame` (the first detection frame number, not 0) and is
succeeding. The issue is the user SEES a stale high frame number on initial render,
not 0. The actual overlay draw is correct once the video loads — the frame counter
just shows a non-zero starting frame because that IS what seekToFrame sets.

**Concrete fix approach:**
- In `loadCase`, change `nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))`
  to `nextTick(() => seekToFrame(0))` if the intent is to start at frame 0.
- OR: accept that starting at first-detection-frame is intentional, but ensure
  `currentFrame.value = 0` is not set before the `seekToFrame` call creates confusion.
- For the overlay not drawing on initial video load: add an explicit `loadedmetadata`
  listener in the `currentPartVideoUrl` watcher (line 1667–1683) that calls
  `scheduleDraws(1)` after the seek — because `seekAndPlay` already seeks, and the
  existing `seeked` event listener (`onSeeked`) fires after seeking and calls
  `scheduleDraws(1)`. Confirm the `onSeeked` listener is attached before `seekAndPlay`
  is called. Currently `onSeeked` is attached in the `watch([videoRef, data])` watcher
  (line 1686) which fires when `videoRef` becomes non-null — this should happen before
  the `loadedmetadata` fires. Verify ordering.

**Recommended fix (minimal, high confidence):**
In `loadCase` (line 1090), the `nextTick` call seeks to `parsed.results[0]?.frame ?? 0`.
The frame counter shows this value immediately. If the intent is frame 0: change to
`nextTick(() => seekToFrame(0))`. If the intent is first-detection frame (acceptable
behavior), document it and ensure the overlay draws by confirming the `onSeeked`
event path fires reliably after the `watch([videoRef, data])` listener is attached.

### Fix 2: Transitions Matrix — Square Aspect Ratio and Responsiveness

**Problem:** `cellSize` is derived from the constant `200` regardless of
`rightPanelWidth`. The matrix overflows the panel when it narrows below matrix width.

**Concrete fix approach using ResizeObserver:**

1. Add a `ref` for the right panel's transitions container:
   ```js
   const transitionsContainerRef = ref(null);
   const transitionsContainerWidth = ref(200); // default
   ```

2. In `onMounted`, attach a `ResizeObserver`:
   ```js
   let _resizeObserver = null;
   if (transitionsContainerRef.value) {
     _resizeObserver = new ResizeObserver(entries => {
       const w = entries[0]?.contentRect.width;
       if (w) transitionsContainerWidth.value = w;
     });
     _resizeObserver.observe(transitionsContainerRef.value);
   }
   ```
   Disconnect in `onUnmounted`.

3. Change `transitionMatrix` computed to use `transitionsContainerWidth.value`:
   ```js
   // Available width minus row-label column (~70px) and padding (~28px)
   const availableW = transitionsContainerWidth.value - 98;
   const cellSize = Math.max(14, Math.min(32, Math.floor(availableW / classes.length)));
   ```

4. The matrix is square by construction (same `cellSize` for row height and column
   width), so a fixed `cellSize` derived from available width gives a square result.
   No CSS `aspect-ratio` needed — the grid is inherently square when all cells are
   square.

**Alternative (pure CSS, no JS):**
Set the matrix grid to `display: grid; grid-template-columns: repeat(N, 1fr)` with
`aspect-ratio: 1` on each cell. This requires replacing the current `v-for` inline
style `width/height: cellSize` with a CSS grid approach and a computed column count.
This is simpler but requires changing the template structure more significantly.
ResizeObserver approach is more surgical.

**Build order:** Fix 1 (frame state) is independent of Fix 2 (matrix responsiveness).
They can be implemented in either order.

---

## Component Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| Router → YOLOVisualizer | `props.id` (case name string) | Component remounts on case change via CasePicker navigation |
| YOLOVisualizer → Django API | `fetch()` calls, plain HTTP | All in `loadCase`, `reloadData`, `runPostprocess` |
| YOLOVisualizer → `<video>` | `videoRef.value.src`, `.currentTime`, `.play()`, `.pause()` | Imperative DOM manipulation via ref |
| YOLOVisualizer → `<canvas>` | `getContext('2d')`, direct pixel ops | Three canvases: overlay, raster, minimap |
| YOLOVisualizer → localStorage | `yolo-visualizer-custom-order` key | Class sort order persistence |

---

## Anti-Patterns

### Anti-Pattern 1: Relying on Watch Order for Initialization Correctness

**What people do:** Chain async operations across multiple watchers and nextTick
calls where the correctness depends on watchers firing in a specific order.

**Why it's wrong:** Vue's watcher scheduling is well-defined per-tick, but when
async gaps (fetch, nextTick) are introduced between state mutations, the order
of side effects becomes harder to reason about. The current frame-initialization
sequence has exactly this fragility.

**Do this instead:** Consolidate initialization into a single async `loadCase`
function that awaits the video ready state explicitly before setting `currentFrame`,
rather than chaining `nextTick` → `seekToFrame` → watcher → DOM event → draw.

### Anti-Pattern 2: Hardcoded Pixel Constants in Computed Properties

**What people do:** Use magic numbers (`200`, `32`, `20`) as size constraints in
computed properties that compute layout-dependent values.

**Why it's wrong:** When the containing panel resizes, the computed does not
invalidate (the panel width is not a reactive dependency), so the layout becomes
stale.

**Do this instead:** Use a `ResizeObserver` to write panel width into a reactive
ref, then use that ref as a dependency in the computed. `ResizeObserver` is
supported in all modern browsers and requires no polyfill for this project's
target environment.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `currentPartVideoUrl` watcher → video element | Imperative: sets `src`, attaches `loadedmetadata` listener | Key initialization path; overlay draw depends on video dimensions being available |
| `watch([videoRef, data])` → video events | Attaches `play`, `pause`, `ended`, `seeked` listeners | Must attach before video starts playing to catch `seeked` after initial load |
| `scheduleDraws(flags)` → RAF | Bitmask accumulation + single RAF flush | All canvas redraws must go through this; don't call `drawOverlay()` directly |

### Scaling Considerations

This is an internal tool with O(1) concurrent users. Scaling is not a concern.
The performance constraint is rendering speed for videos with tens of thousands
of detection frames — already addressed by the sparse map optimization and RAF
batching in the existing code.

---

## Roadmap Implications

**Phase structure for the active milestone:**

1. **Fix frame/overlay initialization** — Diagnose the exact sequencing of
   `loadedmetadata`, `videoRef` availability, and `currentFrame` assignment.
   The fix is likely 3-5 lines. Risk: low (well-understood Vue 3 patterns).

2. **Make transitions matrix responsive** — Add `ResizeObserver` to the right panel's
   transitions container. Update `transitionMatrix` computed to use observed width.
   Risk: low (`ResizeObserver` is straightforward; no async concerns).

3. **Scrollbar styling** — Pure CSS (`::webkit-scrollbar` + `scrollbar-width: thin`).
   Apply globally or scoped to `.right-panel` and timeline areas. Risk: trivial.

4. **Loading screen** — Add a `loading` ref to `YOLOVisualizer.vue`, set true at
   start of `loadCase`, false after data is set and video is ready. Conditionally
   render an overlay. Risk: low.

**All fixes are self-contained within `YOLOVisualizer.vue` except scrollbar styling
(which may be global CSS). No new files needed. No backend changes needed.**

---

*Architecture research for: ENT CV Web Viewer — Vue 3 video + canvas overlay synchronization*
*Researched: 2026-03-07*
