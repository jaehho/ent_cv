# Stack Research

**Domain:** Vue 3 SPA polish — loading states, custom scrollbars, canvas/video synchronization
**Researched:** 2026-03-07
**Confidence:** HIGH (loading/scrollbar), HIGH (canvas sync — existing code already uses the right API)

## Context: What Exists

The existing `YOLOVisualizer.vue` already uses `requestVideoFrameCallback` (RVFC) with a `requestAnimationFrame` fallback. `currentFrame` is `ref(0)`. The component gates rendering with `v-if="data"` — no loading state is shown while data is fetching or while the video loads its metadata. The frame counter bug (stale initial frame) is a state-reset issue on case load, not a sync API choice.

No new dependencies should be added per the project constraint. All solutions below are pure CSS or Vue reactive primitives.

---

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Vue 3 `v-if` / `v-else` | 3.4 (already in use) | Show loading overlay vs. main UI | Zero overhead; already the gating mechanism (`v-if="data"`). Extend it with a loading ref rather than adding a library. |
| CSS `@keyframes` shimmer | N/A | Skeleton pulse animation | No dependency. A single `@keyframes` block in `<style>` covers all skeleton elements. Standard pattern in 2025. |
| `scrollbar-width` + `scrollbar-color` | CSS Level 5 (standard) | Cross-browser scrollbar thickness and color | Supported in all modern browsers as of Chrome 121 / Firefox 64 / Safari 16. Baseline 2024 "widely available". |
| `::-webkit-scrollbar` pseudo-elements | Non-standard but universal | Fine-grained scrollbar shape for Chromium/WebKit | Chrome/Edge/Safari require this for radius, padding, and per-part color control. Both rule sets coexist safely — browsers ignore the one they don't understand. |
| `ResizeObserver` | Web API (built-in) | Detect panel width changes for square plot resizing | Already available; no import needed. Fires synchronously before paint in modern Chrome. Use with `requestAnimationFrame` guard to avoid layout thrashing. |
| `requestVideoFrameCallback` | Baseline 2024 | Canvas overlay frame sync | Already used in the codebase. Fires at the video's frame rate, not display refresh rate. This is the correct API for frame-accurate bounding box overlay. |

### Supporting Libraries

None recommended. Project constraint prohibits new dependencies and all three problem areas are solvable with CSS and native Vue/Web APIs.

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Browser DevTools > Layers | Verify canvas positioning stays in sync with video layout | Check that `position: absolute` overlay matches video's `object-fit: contain` letterbox rect |
| ESLint + `eslint-plugin-vue` (already configured) | Catch reactive side effects in watchers | Existing setup is sufficient |

---

## Approaches by Problem Area

### 1. Loading Screen

**Pattern: Reactive `isLoading` ref with `v-if`/`v-else` in the existing component gate.**

The current gate is `v-if="data"`. Extend it to also require `!isLoading`:

```js
// In setup()
const isLoading = ref(true);
// Set false after data fetch AND video loadedmetadata both resolve
```

Show a skeleton layout (header bar, left panel placeholder, video area placeholder) while `isLoading` is true. Animate with a shimmer sweep using a single `@keyframes`.

**Shimmer pattern (pure CSS, no library):**

```css
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position:  200% 0; }
}
.skeleton {
  background: linear-gradient(90deg, #1a1a24 25%, #2a2a38 50%, #1a1a24 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 4px;
}
```

**Why this over a library:** `vue-skeletor` and similar add ~15-30kB for what is a `<div>` with a CSS animation. The existing site already has a dark color scheme (`#1a1a24` area) — matching the shimmer manually is 5 lines of CSS.

**What loading must cover:** Two async phases must BOTH complete before hiding the loader:
1. `fetch /api/cases/<id>/detections/` (the `data` ref)
2. Video `loadedmetadata` event fires on the `<video>` element

Gate on both, not just `data`. Otherwise the overlay jumps from skeleton directly to a blank canvas frame.

**Confidence: HIGH** — standard Vue 3 reactive pattern, no library risk.

---

### 2. Custom Scrollbar CSS

**Pattern: Two rule sets in `<style>` (or global CSS). Both must exist.**

```css
/* Standard (Firefox, Chrome 121+, Safari 16+) */
.scrollable {
  scrollbar-width: thin;
  scrollbar-color: #4ecdc4 #1a1a24; /* thumb track — match site accent */
}

/* WebKit pseudo-elements (Chrome, Edge, Safari — all versions) */
.scrollable::-webkit-scrollbar        { width: 6px; }
.scrollable::-webkit-scrollbar-track  { background: #1a1a24; border-radius: 3px; }
.scrollable::-webkit-scrollbar-thumb  { background: #4ecdc4; border-radius: 3px; }
.scrollable::-webkit-scrollbar-thumb:hover { background: #38b2ac; }
```

**Apply globally** (e.g., in `src/style.css` or `App.vue` `<style>`) so the classes section, case picker scroll container, and any future scrollable areas all inherit it without per-component declarations.

**Why both rule sets:** `scrollbar-width`/`scrollbar-color` became Baseline "widely available" only in 2024 (Chrome 121, Feb 2024). WebKit pseudo-elements are non-standard but implemented in all Chromium-based browsers for all versions. They do not conflict — browsers silently skip rules they don't understand. Using only the standard properties omits the radius/hover-color control that makes the scrollbar feel custom.

**What NOT to do:** Do not use a JavaScript scrollbar library (OverlayScrollbars, SimpleBar, etc.). They intercept native scroll events, add DOM nodes, and cause accessibility issues. Native CSS is sufficient and has zero runtime cost.

**Confidence: HIGH** — MDN documents both rule sets as the canonical cross-browser approach.

---

### 3. Canvas Overlay Synchronization and Frame State Reset

#### 3a. Frame Reset Bug

**Problem:** `currentFrame` is not reset to `0` when a new case loads. The watcher at line 1083 does `currentFrame.value = 0` but the `videoRef` watcher that re-attaches RVFC fires asynchronously, so the first RVFC callback can read `video.currentTime` before the video has seeked to frame 0, yielding a stale frame number.

**Fix pattern:** In the `loadedmetadata` handler (or `seeked` handler after the initial seek), explicitly set `currentFrame.value = 0` immediately before the video is allowed to play, and call `scheduleDraws(1)` after. The `seeked` event is the correct gate — it fires after `video.currentTime` is actually positioned.

**Confidence: HIGH** — `seeked` firing after `currentTime` is set is documented browser behavior.

#### 3b. Canvas Overlay Position Sync (Existing Approach is Correct)

The existing code reads `canvas.clientWidth`/`canvas.clientHeight` and corrects for `object-fit: contain` letterboxing. This is the right approach. The residual mismatch on initial load is caused by the same frame reset bug — the canvas draws detections for the wrong frame, not for wrong coordinates.

**Do not switch** to `requestAnimationFrame` loops for overlay drawing. The existing pattern (draw on `seeked`, draw on `currentFrame` change via `watch`) is correct and avoids unnecessary repaints.

#### 3c. Square Aspect Ratio for Transitions Matrix

**Pattern: `ResizeObserver` on the panel container → update a reactive `plotSize` ref → bind to both `width` and `height` of the canvas.**

```js
const plotSize = ref(300);
const panelRef = ref(null);

onMounted(() => {
  const ro = new ResizeObserver(entries => {
    const width = entries[0].contentRect.width;
    // rAF guard prevents layout thrashing inside ResizeObserver callback
    requestAnimationFrame(() => { plotSize.value = width; });
  });
  ro.observe(panelRef.value);
  onUnmounted(() => ro.disconnect());
});
```

```html
<canvas :width="plotSize" :height="plotSize" ref="matrixCanvas" />
```

**Why `requestAnimationFrame` inside `ResizeObserver`:** ResizeObserver callbacks may fire mid-layout in some browsers. Wrapping the reactive update in `rAF` defers it to the next paint, preventing the "ResizeObserver loop limit exceeded" warning and canvas flicker.

**Confidence: HIGH** — `ResizeObserver` is Baseline widely available; the `rAF` guard is the recommended pattern per the CSSWG spec discussion.

---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| Pure CSS skeleton (`@keyframes` shimmer) | `vue-skeletor` or `vue-loading-skeleton` | Only if skeleton shapes need to auto-match typography across many component types — not the case here, shapes are static |
| Native CSS scrollbar properties | OverlayScrollbars JS library | Only if scroll container needs custom scrollbar that overlays content (not needed here — native `thin` width is fine) |
| `requestVideoFrameCallback` (already in use) | `timeupdate` event for frame sync | `timeupdate` fires at ~4Hz when paused and is rate-limited; RVFC is frame-accurate. Do not regress to `timeupdate`. |
| `ResizeObserver` for plot resize | CSS `aspect-ratio: 1/1` on the canvas | `aspect-ratio` controls CSS display size but not the canvas `width`/`height` attributes — the bitmap would still be wrong resolution. Use `ResizeObserver` to set both. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| JS scrollbar libraries (OverlayScrollbars, SimpleBar) | Add runtime DOM mutation, break native scroll accessibility, require cleanup on unmount | Native `scrollbar-width` + `::-webkit-scrollbar` CSS |
| `vue-skeletor` / `vue-loading-skeleton` | 15-30kB for a `<div>` + CSS animation; the project prohibits new dependencies | `@keyframes shimmer` in `<style>` |
| `timeupdate` for canvas frame sync | Rate-limited, not frame-accurate; regresses what RVFC already provides | `requestVideoFrameCallback` (already in use) |
| `setInterval` for canvas resize polling | Wastes CPU, introduces 16ms+ lag | `ResizeObserver` |
| CSS `aspect-ratio: 1/1` alone for the matrix canvas | Controls layout box but not canvas bitmap dimensions — drawing will be stretched | `ResizeObserver` + set `canvas.width = canvas.height = measuredWidth` |

---

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| `scrollbar-width` / `scrollbar-color` CSS | Chrome 121+, Firefox 64+, Safari 16+ | Baseline "widely available" as of 2024. Pre-Chrome 121 Chrome users see unstyled scrollbar — acceptable fallback. |
| `::-webkit-scrollbar` | Chrome all versions, Edge (Chromium), Safari all versions | Non-standard but stable. Firefox ignores it silently. |
| `requestVideoFrameCallback` | Baseline 2024 (Chrome 83+, Firefox 132+, Safari 15.4+) | Already used in codebase with rAF fallback. |
| `ResizeObserver` | Baseline widely available (Chrome 64+, Firefox 69+, Safari 13.1+) | No polyfill needed. |

---

## Sources

- [MDN: requestVideoFrameCallback](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback) — Browser support (Baseline 2024), frame sync semantics — HIGH confidence
- [MDN: CSS Scrollbars Styling](https://developer.mozilla.org/en-US/docs/Web/CSS/Guides/Scrollbars_styling) — Standard properties and cross-browser strategy — HIGH confidence
- [Chrome for Developers: Scrollbar Styling](https://developer.chrome.com/docs/css-ui/scrollbar-styling) — Chrome 121 adoption of standard properties — HIGH confidence
- [LearnVue: Vue Skeleton Loading with Suspense](https://learnvue.co/articles/vue-skeleton-loading) — `v-if`/`v-else` skeleton pattern — MEDIUM confidence (community source, pattern is standard)
- [CSSWG drafts issue #9717](https://github.com/w3c/csswg-drafts/issues/9717) — ResizeObserver + rAF guard for canvas flicker — MEDIUM confidence (spec discussion, not final doc)

---

*Stack research for: Vue 3 SPA UI polish (loading states, scrollbars, canvas/video sync)*
*Researched: 2026-03-07*
