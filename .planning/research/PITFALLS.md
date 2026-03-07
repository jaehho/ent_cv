# Pitfalls Research

**Domain:** Vue 3 video viewer UI polish — loading states, custom scrollbars, canvas/video sync, ResizeObserver
**Researched:** 2026-03-07
**Confidence:** HIGH (patterns verified against Vue 3 reactivity model and browser APIs; codebase-specific analysis from YOLOVisualizer.vue)

## Critical Pitfalls

### Pitfall 1: Loading state that clears before the video is actually ready

**What goes wrong:**
A `isLoading` flag is set to `false` after the fetch resolves and `data` is populated. But the `<video>` element hasn't fired `loadedmetadata` or `canplay` yet, so the overlay canvas has no dimensions and `currentFrame` is initialized while `videoRef.value` is `null` or has zero dimensions. The UI flickers briefly showing stale/blank overlay data.

**Why it happens:**
Developers conflate "data fetched" with "video ready." In this codebase, `data` is set from the API response, but the video element is conditionally rendered via `v-if="data"` — so the DOM node doesn't exist until after the reactive update settles. Even after it exists, the browser hasn't decoded the video stream yet.

**How to avoid:**
Track two separate flags: `dataReady` (API fetch done) and `videoReady` (video fired `loadedmetadata`). Drive `isLoading` from `dataReady && videoReady`. Wire `videoReady` inside the `watch([videoRef, data], ...)` block that already exists in the component, adding a `loadedmetadata` listener that sets the flag. Use `nextTick` after setting `data` before attaching the listener, since `videoRef` won't exist until the DOM updates.

**Warning signs:**
- Canvas draws immediately on load but shows wrong frame or is blank
- `currentFrame` is non-zero on initial render despite no user interaction
- Loading spinner disappears but video still shows a buffering state

**Phase to address:**
Frame-state initialization bug fix phase (same phase as the `currentFrame` reset bug — they share a root cause).

---

### Pitfall 2: Stale `currentFrame` on case load due to missing reactive reset

**What goes wrong:**
When navigating to a new case (or reloading the same case), `currentFrame` retains its previous value. The canvas draw function runs immediately using this stale frame number, rendering detections for frame ~40-500 instead of frame 0. This is the documented bug in PROJECT.md.

**Why it happens:**
`currentFrame` is a `ref(0)` initialized once at component setup. The component is likely kept alive across case navigations (Vue Router reuses the component instance). Even if recreated, the `watch([videoRef, data])` that syncs frame state starts ticking before `seekToFrame(0)` is called. Line 1083 (`currentFrame.value = 0`) resets the value, but any draw watchers triggered by `data` changing will fire before this reset runs, using the old value.

**How to avoid:**
Reset `currentFrame.value = 0` as the very first statement inside the `watch([videoRef, data])` handler, before any async work. Also cancel any in-flight `requestVideoFrameCallback` before resetting, since RVFC callbacks may still be queued from the previous playback session and will overwrite `currentFrame` with a stale media time.

**Warning signs:**
- On case load, the frame counter shows a non-zero value for a brief moment
- Canvas overlay renders a detection bounding box that doesn't match what the video shows
- Bug only manifests on second and subsequent case loads, not the first

**Phase to address:**
Frame-state initialization bug fix phase.

---

### Pitfall 3: Canvas dimensions are 0x0 on first draw because the element isn't laid out yet

**What goes wrong:**
`canvas.clientWidth` and `canvas.clientHeight` return 0 when the canvas element has just been inserted into the DOM but the browser hasn't performed a layout pass. `canvas.width = 0 * dpr` clears any content and the draw does nothing visible. Subsequent redraws may not be triggered because no watched reactive dependencies changed.

**Why it happens:**
The draw function reads `canvas.clientWidth` synchronously inside a `watchEffect` or `watch` that fires immediately after `data` is set. At that moment the canvas is in the DOM but not yet painted. This is especially likely when `v-if` conditionally renders the entire video+canvas block.

**How to avoid:**
Gate draw calls on `canvas.clientWidth > 0`. If zero, defer with `requestAnimationFrame(() => drawOverlay())` rather than calling immediately. The existing `scheduleDraws` helper in YOLOVisualizer is the right pattern — ensure it is always used rather than calling draw functions directly. Also add a `ResizeObserver` on the canvas wrapper that triggers a redraw, so the first layout paint fires a draw.

**Warning signs:**
- Canvas appears blank on initial load but redraws correctly after window resize
- `canvas.width` is set to 0 in DevTools Elements panel immediately after case load
- Overlay appears after the user clicks play or interacts with the UI

**Phase to address:**
Frame-state initialization bug fix phase (canvas/video sync on load).

---

### Pitfall 4: ResizeObserver callback fires before Vue has updated the DOM

**What goes wrong:**
A `ResizeObserver` is attached to a container in `onMounted`. When panel width changes (drag resize), the observer fires synchronously with the new layout size, but the Vue template that depends on the same reactive state hasn't re-rendered yet. Drawing to the canvas with new dimensions while Vue's virtual DOM is still being patched causes a visual mismatch — the canvas is sized for the new width but shows content computed from the old width.

**Why it happens:**
`ResizeObserver` callbacks run after layout but may run before the browser's next paint and before Vue's async queue flushes. If the resize is triggered by a reactive state change (e.g., `leftPanelWidth`), the observer callback may race with Vue's watcher queue.

**How to avoid:**
Inside the `ResizeObserver` callback, always call `nextTick(() => redraw())` rather than calling redraw directly. This ensures Vue has flushed its reactive updates before the canvas draws. Alternatively, debounce the observer callback (16ms) to coalesce rapid resize events during drag and avoid redundant draws.

**Warning signs:**
- Canvas content appears misaligned by one frame during panel drag
- Console shows "ResizeObserver loop limit exceeded" warning
- Resize handler calls a function that reads a Vue ref that hasn't updated yet

**Phase to address:**
Transitions matrix responsive square resize phase.

---

### Pitfall 5: Forgetting to disconnect ResizeObserver on component unmount

**What goes wrong:**
A `ResizeObserver` attached in `onMounted` continues to fire after the component is destroyed (e.g., user navigates back to case picker). The callback attempts to access `ref.value` which is now `null`, causing a TypeError. In long sessions this causes a memory leak.

**Why it happens:**
`ResizeObserver` is a raw browser API — Vue has no automatic cleanup for it. Developers remember to clean up Vue watchers (which have `onCleanup` callbacks) but forget raw observers.

**How to avoid:**
Always pair `ResizeObserver` creation with `onUnmounted(() => observer.disconnect())`. The existing codebase already uses `onUnmounted` for `cancelVideoFrameCallback` — follow the same pattern. Guard the callback body with `if (!ref.value) return` as a secondary defense.

**Warning signs:**
- TypeErrors in console after navigating away from the visualizer
- Memory profiler shows canvas/DOM nodes are retained after navigation
- Observer fires on unmounted component

**Phase to address:**
Transitions matrix responsive square resize phase.

---

### Pitfall 6: CSS custom scrollbars break in Firefox when using only `::-webkit-scrollbar`

**What goes wrong:**
`::-webkit-scrollbar`, `::-webkit-scrollbar-thumb`, and `::-webkit-scrollbar-track` are Chromium/Safari-only pseudo-elements. Firefox ignores them entirely and shows the default OS scrollbar. The site aesthetic is inconsistent across browsers.

**Why it happens:**
Webkit scrollbar pseudo-elements are non-standard and were never implemented by Firefox. Firefox uses the standard `scrollbar-width` and `scrollbar-color` CSS properties instead.

**How to avoid:**
Use both approaches together. Set `scrollbar-width: thin` and `scrollbar-color: <thumb> <track>` for Firefox (these are standard CSS). Add the `::-webkit-scrollbar` block for Chromium. Apply globally in `main.css` or a dedicated `scrollbar.css` imported in `main.js`, targeting `*` or specific overflow containers. PROJECT.md already identifies this dual approach as the intended decision — just ensure both property sets are written.

**Warning signs:**
- Scrollbar styling works in Chrome dev environment but looks wrong in Firefox
- Only `::-webkit-scrollbar` styles exist in the CSS without `scrollbar-width`/`scrollbar-color`

**Phase to address:**
Custom scrollbar styling phase.

---

### Pitfall 7: Global scrollbar CSS overrides nested elements unexpectedly

**What goes wrong:**
Applying `::-webkit-scrollbar` styles to `*` or `html, body` changes scrollbar appearance on every scrollable element, including browser-native dropdowns, date pickers, and any third-party components. In a Vue SFC setup, styles in `<style>` without `scoped` are truly global and affect the entire document.

**Why it happens:**
The `*` selector combined with `::-webkit-scrollbar` has extremely broad specificity. In SFCs, non-scoped styles are injected into the document head and affect all elements.

**How to avoid:**
Scope the custom scrollbar CSS to specific class selectors (`.classes-panel`, `.case-list`) rather than `*`. For the global default, apply to `html` only. This gives consistent styling on known containers without surprising overrides.

**Warning signs:**
- Unexpected scrollbar styling in modal dialogs or dropdowns
- Browser autocomplete lists look different from their native appearance

**Phase to address:**
Custom scrollbar styling phase.

---

### Pitfall 8: Square aspect ratio maintained with CSS `aspect-ratio` breaks when parent has explicit height

**What goes wrong:**
The transitions matrix container is given a fixed or percentage height by the panel layout. `aspect-ratio: 1` on the inner canvas/div then conflicts with the height constraint — the browser resolves the conflict by cropping rather than shrinking to maintain the square. The result is a non-square or overflowing element.

**Why it happens:**
`aspect-ratio` works as a ratio constraint, but it is overridden when both `width` and `height` are explicitly set. If the parent sets `height: 100%` and the child tries to infer width from `aspect-ratio`, the math may produce a taller-than-wide result that is then clipped.

**How to avoid:**
Drive the square from `width` only. Set `width: 100%` on the container and use `aspect-ratio: 1 / 1` with no explicit `height`. If the panel uses flex layout, set `flex-shrink: 0` on the container. For the canvas element specifically, use a `ResizeObserver` on the container to read `offsetWidth` and set both `canvas.width` and `canvas.height` to that value in pixels — this is more reliable than pure CSS for canvas elements because canvas has its own internal width/height that CSS `aspect-ratio` does not control.

**Warning signs:**
- Chart appears rectangular in DevTools even though `aspect-ratio: 1` is set
- Chart overflows its panel container
- Resizing the panel makes the chart taller but not wider

**Phase to address:**
Transitions matrix responsive square resize phase.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Single `isLoading` flag for fetch + video ready | Simpler code | Loading screen disappears before video is playable; frame state bugs | Never — split into two flags |
| Calling draw functions directly instead of via `scheduleDraws` | Fewer function calls | Canvas drawn before layout, shows blank or stale content | Never |
| Attaching `ResizeObserver` without cleanup | Less boilerplate | Memory leak and TypeErrors after navigation | Never |
| Using only `::-webkit-scrollbar` without Firefox properties | Works in Chrome dev | Inconsistent in Firefox production | Never for public-facing UI |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `requestVideoFrameCallback` | Not cancelling before `data` changes, leaving stale callbacks queued | Cancel with `cancelVideoFrameCallback` before resetting state; guard callback body with stale-check |
| `loadedmetadata` event | Attaching listener before `videoRef.value` exists (after `v-if` renders) | Use `nextTick` after reactive update that creates the element, or attach inside the `watch([videoRef, data])` handler |
| Canvas draw on `watchEffect` | Runs immediately before DOM is laid out | Gate on `canvas.clientWidth > 0`; defer with `requestAnimationFrame` if zero |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `ResizeObserver` triggering on every pixel during panel drag | Jank during drag; excessive canvas redraws | Debounce observer callback to 16ms | Any drag interaction |
| Drawing all three canvases (overlay, raster, minimap) on every frame tick | High CPU during playback | Only redraw canvases when their inputs actually changed; raster only redraws on zoom/pan, not every frame | High-resolution video, 60fps playback |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Loading screen clears before video can play | User clicks play on a blank video; disorienting | Hold loading state until `canplaythrough` or `loadedmetadata` fires |
| No loading state at all during postprocess | User clicks "Run Filter" with no feedback; double-clicks cause duplicate requests | Disable the button and show inline spinner; existing `postprocessing` ref can drive this |
| Frame counter shows wrong frame on load | Overlays don't match what user sees; erodes trust in detection data | Reset `currentFrame` to 0 synchronously when `data` changes |

## "Looks Done But Isn't" Checklist

- [ ] **Loading screen:** Verify the loading flag waits for `loadedmetadata`, not just the fetch. Test by throttling network in DevTools — loading screen should persist until video is buffered enough to play.
- [ ] **Custom scrollbars:** Test in Firefox — `::-webkit-scrollbar` does nothing there. Verify `scrollbar-width` and `scrollbar-color` are also set.
- [ ] **Frame reset:** Open a case, scrub to frame 300, navigate back to case picker, open the same case again — frame counter must show 0 and overlay must be blank (no detections rendered).
- [ ] **Square transitions matrix:** Drag the panel divider to minimum and maximum width — the chart must remain square (equal pixel width and height) at both extremes.
- [ ] **ResizeObserver cleanup:** Navigate from visualizer to case picker and back multiple times; verify no accumulating console errors.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Loading state cleared too early | LOW | Add second flag, wire to `loadedmetadata`; one-line change in watch handler |
| Stale `currentFrame` on load | LOW | Add `currentFrame.value = 0` reset at top of data watch handler; cancel RVFC callbacks |
| Canvas 0x0 on first draw | LOW | Add `if (!canvas.clientWidth) return requestAnimationFrame(draw)` guard |
| ResizeObserver not disconnected | LOW | Add `onUnmounted` cleanup next to existing cleanup block |
| Scrollbar CSS Firefox gap | LOW | Add two CSS properties (`scrollbar-width`, `scrollbar-color`) alongside existing webkit block |
| Square aspect ratio broken | MEDIUM | Switch from CSS-only to ResizeObserver-driven canvas sizing; read `offsetWidth` and set both dimensions |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Loading state clears before video ready | Loading screen implementation | Throttle network; spinner must persist until video fires `loadedmetadata` |
| Stale `currentFrame` on case load | Frame-state initialization bug fix | Navigate to case, scrub to frame 300, navigate back, reload — must show frame 0 |
| Canvas 0x0 on first draw | Frame-state initialization bug fix | Load a case; canvas overlay must render correctly without any user interaction |
| ResizeObserver races Vue DOM update | Transitions matrix responsive resize | Drag panel while video plays; chart must not flicker or misalign |
| ResizeObserver not disconnected | Transitions matrix responsive resize | Navigate back and forth 5 times; no TypeErrors in console |
| Webkit-only scrollbar CSS | Custom scrollbar styling | Test scrollbar appearance in Firefox; must match site aesthetic |
| Global scrollbar CSS unintended scope | Custom scrollbar styling | Inspect all scrollable containers after applying global styles |
| Square aspect ratio broken by height constraints | Transitions matrix responsive resize | Test at minimum and maximum panel widths; chart pixel dimensions must be equal |

## Sources

- Vue 3 documentation on `watch`, `watchEffect`, `nextTick`, and lifecycle hooks: https://vuejs.org/guide/essentials/watchers.html
- MDN `ResizeObserver` documentation: https://developer.mozilla.org/en-US/docs/Web/API/ResizeObserver
- MDN CSS `scrollbar-width` (Firefox standard): https://developer.mozilla.org/en-US/docs/Web/CSS/scrollbar-width
- MDN CSS `scrollbar-color`: https://developer.mozilla.org/en-US/docs/Web/CSS/scrollbar-color
- MDN `::-webkit-scrollbar` (non-standard, Chromium/Safari only): https://developer.mozilla.org/en-US/docs/Web/CSS/::-webkit-scrollbar
- MDN `HTMLVideoElement.requestVideoFrameCallback()`: https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback
- MDN `HTMLMediaElement: loadedmetadata` event: https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/loadedmetadata_event
- MDN CSS `aspect-ratio` property: https://developer.mozilla.org/en-US/docs/Web/CSS/aspect-ratio
- Codebase analysis: `/home/jaeho/ent_cv/web/frontend/src/components/YOLOVisualizer.vue` (direct inspection)

---
*Pitfalls research for: Vue 3 video viewer UI polish (ENT CV)*
*Researched: 2026-03-07*
