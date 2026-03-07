# Project Research Summary

**Project:** ENT CV Web Viewer — UI Polish Milestone
**Domain:** Vue 3 SPA video review tool (surgical instrument detection)
**Researched:** 2026-03-07
**Confidence:** HIGH

## Executive Summary

This milestone is a UI polish pass on an existing, working Vue 3 + Django application. The core product — YOLO-based surgical instrument detection overlaid on video — already functions. The work is entirely frontend, confined almost entirely to `YOLOVisualizer.vue` and global CSS. No backend changes, no new dependencies, and no new files are required. The project explicitly prohibits adding packages, so every solution is implemented with native browser APIs, Vue 3 reactive primitives, and pure CSS.

The recommended approach is to tackle fixes in dependency order: first resolve the shared root cause of the frame-state and canvas-alignment bugs (they are the same async sequencing problem), then add the loading screen that depends on that same initialization pathway, then layer in the independent cosmetic changes (scrollbar styling, matrix responsiveness). All six deliverables are low-complexity, low-risk, and achievable with well-documented browser APIs.

The primary risk is the async initialization sequence in `loadCase` — multiple watchers, `nextTick` calls, and DOM events must fire in the right order. The research identified the exact failure path: `drawOverlay()` runs before `videoRef` is non-null or before `video.videoWidth` is non-zero, producing a blank or stale overlay. Fixing this requires consolidating draw triggers through the existing `scheduleDraws()` helper and gating the loading flag on both the API response and the `loadedmetadata` video event, not just one of them.

## Key Findings

### Recommended Stack

No new stack choices are needed. All six features are implemented using capabilities already present in the codebase or available as zero-dependency browser APIs. Vue 3's `ref`, `watch`, `onMounted`, and `onUnmounted` cover state management. `ResizeObserver` (Baseline widely available) handles panel dimension tracking. `requestVideoFrameCallback` is already in use for frame-accurate overlay sync. CSS `scrollbar-width`/`scrollbar-color` plus `::-webkit-scrollbar` pseudo-elements cover cross-browser scrollbar styling without any library.

**Core technologies:**
- Vue 3 reactive refs (`ref`, `shallowRef`, `computed`): state management — already in use, no changes needed
- `ResizeObserver` (Web API): observe panel width changes for matrix responsiveness — no import, universally supported
- `requestVideoFrameCallback` (Web API): frame-accurate canvas sync — already integrated with `requestAnimationFrame` fallback
- CSS `scrollbar-width` + `scrollbar-color`: Firefox-standard scrollbar styling — add alongside existing `::-webkit-scrollbar` block
- CSS `::-webkit-scrollbar` pseudo-elements: Chromium/Safari scrollbar styling — already used, needs Firefox counterpart
- CSS `@keyframes` shimmer: skeleton/loading animation — 5 lines of CSS, no library

### Expected Features

**Must have (table stakes — all P1, all this milestone):**
- Loading indicator while API fetch + video `loadedmetadata` both resolve — blank screen signals broken page
- Frame 0 reset on case load — stale frame on load breaks the core value proposition
- Canvas overlay pixel-accurate to video on first render — misaligned boxes erode trust immediately
- Scrollable content in case list and classes panel — clipped content is jarring
- Custom scrollbar styling matching dark theme — default OS scrollbars clash visually
- Transitions matrix square aspect ratio — non-square symmetric matrix is visually misleading

**Should have (differentiator — post-validation, v1.x):**
- `requestVideoFrameCallback` swap for playback sync — only if overlay lag is user-reported at normal speeds

**Defer (v2+):**
- Skeleton loading screens (layout-matched placeholders) — spinner sufficient for internal tool
- Keyboard shortcuts for frame stepping — ergonomic improvement, not blocking

### Architecture Approach

All changes live inside the single `YOLOVisualizer.vue` component (~1900 lines) with one exception: scrollbar CSS goes in global stylesheet. The component uses a well-established RAF-batched canvas draw pattern (`scheduleDraws(flags)` bitmask) — all new draw triggers must route through this helper, never call draw functions directly. State follows a strict reactive hierarchy: `data` (shallowRef, frozen API payload) → `computed` derivations → imperative canvas draws. The frame-state bug and canvas-alignment bug share one root cause: draw triggers fire before the video element exists in the DOM or before it has loaded its dimensions.

**Major components:**
1. `YOLOVisualizer.vue` — all visualization logic; contains video, canvas overlays, timeline, stats, and matrix
2. `CasePicker.vue` — case list navigation; needs `overflow-y: auto` and custom scrollbar class
3. Global CSS (`src/style.css` or `App.vue`) — scrollbar rules applied once, inherited everywhere

### Critical Pitfalls

1. **Loading flag cleared after fetch, not after video ready** — track two flags (`dataReady` + `videoReady`); drive `isLoading` from both; wire `videoReady` to `loadedmetadata` inside the existing `watch([videoRef, data])` handler
2. **Stale `currentFrame` on case load** — reset `currentFrame.value = 0` as the first statement in the data watch handler, before any async work; cancel any queued `requestVideoFrameCallback` before resetting
3. **Canvas dimensions 0x0 on first draw** — gate all draw calls on `canvas.clientWidth > 0`; if zero, defer with `requestAnimationFrame`; always use `scheduleDraws()`, never call draw functions directly
4. **`ResizeObserver` not disconnected on unmount** — always pair observer creation with `onUnmounted(() => observer.disconnect())`; the codebase already does this for `cancelVideoFrameCallback` — follow the same pattern
5. **Webkit-only scrollbar CSS missing Firefox counterpart** — `scrollbar-width: thin` and `scrollbar-color` must accompany every `::-webkit-scrollbar` block; test in Firefox before closing the task

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Frame State and Canvas Initialization Fix
**Rationale:** This is the root cause shared by two reported bugs (stale frame, misaligned overlay). It also unblocks Phase 2 because the loading screen depends on knowing when the video is actually ready. Fix the initialization sequence first so everything downstream is built on a stable foundation.
**Delivers:** Correct frame 0 on case load; canvas overlay aligned and rendered on first draw without user interaction
**Addresses:** Frame 0 reset (P1), canvas overlay pixel accuracy (P1)
**Avoids:** Pitfalls 2 (stale frame), 3 (canvas 0x0), and the anti-pattern of relying on watch order for initialization correctness

### Phase 2: Loading Screen
**Rationale:** Depends on the initialization fix from Phase 1 — the loading flag must gate on `loadedmetadata`, which is the same event path fixed in Phase 1. Implementing loading before the initialization fix would require revisiting the loading gate immediately after.
**Delivers:** Skeleton or spinner shown during fetch + video init; hidden only when both are ready
**Uses:** Vue `ref` + `v-if`/`v-else`; CSS `@keyframes shimmer` (optional, for skeleton variant); `loadedmetadata` event path from Phase 1
**Avoids:** Pitfall 1 (loading cleared before video ready)

### Phase 3: Custom Scrollbar Styling
**Rationale:** Fully independent of Phases 1 and 2. Pure CSS change with no JavaScript. Can be done at any point; placed here because it is the lowest-risk change and provides visible polish.
**Delivers:** Dark-themed thin scrollbars in both Chromium and Firefox; consistent with site accent colors
**Uses:** CSS `scrollbar-width: thin`, `scrollbar-color`, `::-webkit-scrollbar` pseudo-elements applied globally
**Avoids:** Pitfall 6 (webkit-only CSS), Pitfall 7 (overly broad selector scope)

### Phase 4: Transitions Matrix Responsive Square Resize
**Rationale:** Independent of all prior phases but placed last because it introduces the only new JavaScript pattern in this milestone (`ResizeObserver`) and has the most pitfalls to avoid (races with Vue DOM updates, missing cleanup). Doing it after simpler phases reduces risk of overlapping changes.
**Delivers:** Transitions matrix that maintains square aspect ratio as panel width changes; correct pixel dimensions for canvas element (not just CSS box)
**Uses:** `ResizeObserver` on transitions container ref; reactive `transitionsContainerWidth` ref as dependency in `transitionMatrix` computed; `nextTick` inside observer callback to avoid Vue DOM race; `onUnmounted` cleanup
**Avoids:** Pitfalls 4 (ResizeObserver races Vue), 5 (observer not disconnected), 8 (CSS aspect-ratio broken by height constraints)

### Phase Ordering Rationale

- Phase 1 before Phase 2: Loading screen correctness depends on the same `loadedmetadata` event path that Phase 1 establishes
- Phase 3 independent: Pure CSS, zero JavaScript, zero risk; could be parallel with any other phase
- Phase 4 last: Introduces the only new JavaScript browser API pattern (`ResizeObserver`) and has the most cleanup requirements; isolating it reduces debugging surface area

### Research Flags

Phases with standard patterns (skip additional research):
- **Phase 3 (Scrollbar CSS):** MDN documents the exact dual rule set required; implementation is mechanical
- **Phase 2 (Loading Screen):** Standard Vue 3 `ref` + `v-if` pattern; no unknowns

Phases that may benefit from reading existing code carefully before implementing:
- **Phase 1 (Frame State Fix):** The async watcher ordering in `loadCase` is subtle — read the exact current sequence at lines 1083, 1667–1686 before writing a fix to avoid introducing a new ordering bug
- **Phase 4 (ResizeObserver):** Verify where the transitions container is in the template and confirm `rightPanelWidth` reactive ref exists before adding a second observer

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | No new stack; all APIs are Baseline widely available; verified against MDN |
| Features | HIGH | Requirements are explicit and scoped; no ambiguity about what ships this milestone |
| Architecture | HIGH | Based on direct codebase inspection of YOLOVisualizer.vue; root cause analysis is specific |
| Pitfalls | HIGH | Pitfalls derived from actual code paths in the component, not hypothetical scenarios |

**Overall confidence:** HIGH

### Gaps to Address

- **Exact `loadCase` async sequence:** The architecture research identified the likely root cause but notes that confirming the ordering of `watch([videoRef, data])` vs `currentPartVideoUrl` watcher firing relative to `nextTick(seekToFrame)` requires reading lines 1083–1090 and 1667–1686 in sequence. Confirm before writing Phase 1 fix.
- **Matrix container DOM structure:** The ResizeObserver approach for Phase 4 assumes there is a wrapping element for the transitions matrix with a stable ref. Verify the template structure before attaching the observer.
- **Postprocess loading state:** PITFALLS.md notes the "Run Filter" button has no loading feedback during postprocessing. This was not in the original scope but is a low-effort addition (the `postprocessing` ref already exists). Flag for roadmapper to include or explicitly defer.

## Sources

### Primary (HIGH confidence)
- MDN: `HTMLVideoElement.requestVideoFrameCallback()` — frame sync semantics, browser support
- MDN: CSS Scrollbars Styling — standard properties and cross-browser strategy
- Chrome for Developers: Scrollbar Styling — Chrome 121 adoption of `scrollbar-width`/`scrollbar-color`
- MDN: `ResizeObserver` — API reference, cleanup requirements
- MDN: `HTMLMediaElement: loadedmetadata event` — event timing relative to video element lifecycle
- Vue 3 docs: `watch`, `watchEffect`, `nextTick`, lifecycle hooks
- Direct codebase inspection: `web/frontend/src/components/YOLOVisualizer.vue`

### Secondary (MEDIUM confidence)
- LearnVue: Vue Skeleton Loading with Suspense — `v-if`/`v-else` skeleton pattern
- CSSWG drafts issue #9717 — ResizeObserver + rAF guard for canvas flicker
- CSS-Tricks: The Current State of Styling Scrollbars in CSS
- ishadeed: Custom Scrollbars in CSS

---
*Research completed: 2026-03-07*
*Ready for roadmap: yes*
