# Phase 1: Frame State Fix - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix the async initialization sequence so the viewer always starts at frame 0 with a correct canvas overlay on every case load. Both BUG-01 and BUG-02 share a single root cause: `loadCase` calls `nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))` which seeks to the first detection frame (40–500), not frame 0.

Scope: `YOLOVisualizer.vue` only. No backend changes. No new state management patterns.

</domain>

<decisions>
## Implementation Decisions

### Starting position
- Always open at frame 0 (video start), even if there are no detections there
- Never auto-jump to the first detection frame on load
- The `nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))` call must be removed or changed to seek to 0

### Canvas at frame 0
- A blank canvas overlay at frame 0 is acceptable — user can scrub or play to see detections
- No auto-scroll or auto-seek to first detection frame after load

### Case switching — full reset
When switching between cases, reset ALL of the following:
- `currentFrame` → 0
- `zoomLevel` → 1 (default)
- `panOffset` → 0 (start of timeline)
- `playbackRate` → 1 (normal speed, also set on the video element)
- Minimap viewport position → beginning

These resets should happen in `loadCase()` alongside the existing resets (filter mode, class visibility, etc.).

### Claude's Discretion
- Exact ordering of resets within `loadCase` relative to `data.value = ...` assignment
- Whether to keep or remove the `nextTick` wrapper (could simply change the argument to 0)
- How to handle the `videoRef.value.playbackRate` reset timing (video element may not exist yet)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `seekToFrame(frame)` at line 1190 — sets `currentFrame.value`, calls `scheduleDraws(1)`, and handles multi-part video seeking. Can be called with `0` instead of the first detection frame.
- `scheduleDraws(flags)` at line 606 — triggers overlay/raster/minimap redraws. Already called in the `loadedmetadata` listener path.

### Established Patterns
- State resets in `loadCase` (lines 1068–1090): `filteredSummary`, `filterInfo`, `filterMode`, `ppResult`, `ppError`, `activeCaseName`, `enabledClasses`, `jumpFilterClassIds`, `videoSrc`, `currentFrame` are all reset before the `nextTick`. This is the right place to add `zoomLevel`, `panOffset`, `playbackRate` resets.
- The `currentPartVideoUrl` watcher (around line 1667) attaches `loadedmetadata` → `seekAndPlay` and calls `scheduleDraws(1)` — this handles the canvas draw after video loads. The frame-counter fix does NOT need to touch this watcher.
- `playbackRate` is set on `videoRef.value.playbackRate` in `setRate()` at line 1272–1273. On case switch, `videoSrc.value = null` is set before the new src, so video element may not exist; reset `playbackRate` ref only in `loadCase`, and let the existing `setRate` path sync it to the element when the video loads.

### Integration Points
- `loadCase` is called from the case picker interaction handler
- `currentFrame`, `zoomLevel`, `panOffset`, `playbackRate` are all `ref()`s at lines 541, 545, 546, 560
- Canvas overlay draw: `drawOverlay()` at line 1388, triggered via `scheduleDraws(1)` — already fires on `seeked` event (line 1733)

</code_context>

<specifics>
## Specific Ideas

- The fix is surgical: in `loadCase`, change `nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))` to `nextTick(() => seekToFrame(0))` and add resets for `zoomLevel`, `panOffset`, `playbackRate` alongside the existing `currentFrame.value = 0` reset.
- The canvas mismatch (BUG-02) should resolve automatically once the frame counter is correct — the existing `seeked`/`loadedmetadata` draw chain is sound.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-frame-state-fix*
*Context gathered: 2026-03-07*
