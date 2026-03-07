# Phase 1: Frame State Fix - Research

**Researched:** 2026-03-07
**Domain:** Vue 3 reactive state / async initialization sequencing in YOLOVisualizer.vue
**Confidence:** HIGH

## Summary

Phase 1 is a surgical fix to a single file: `web/frontend/src/components/YOLOVisualizer.vue`. The root cause of both BUG-01 and BUG-02 is one line in `loadCase()` (line 1090):

```js
nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0));
```

This seeks to the first detection frame (which can be 40–500+) instead of frame 0. The fix is to change the argument to `0`. Additionally, `loadCase()` must reset `zoomLevel`, `panOffset`, and `playbackRate` refs alongside the existing `currentFrame.value = 0` reset on line 1083, so case-switching is fully clean.

BUG-02 (canvas mismatch) resolves automatically once BUG-01 is fixed: the `seeked` event fires after `seekToFrame(0)` which already triggers `scheduleDraws(1)` → `drawOverlay()`. The existing draw chain is sound and requires no changes.

The `videoRef.value.playbackRate` element property must NOT be set in `loadCase()` because `videoSrc.value = null` clears the video element before the new source loads. Only the `playbackRate` ref needs resetting here; the existing `setRate()` path syncs it to the element when the video is ready.

**Primary recommendation:** Change line 1090 from `seekToFrame(parsed.results[0]?.frame ?? 0)` to `seekToFrame(0)`, and add `zoomLevel.value = 1`, `panOffset.value = 0`, `playbackRate.value = 1` to the reset block in `loadCase()` (lines 1069–1083).

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Always open at frame 0 (video start), even if there are no detections there
- Never auto-jump to the first detection frame on load
- The `nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0))` call must be removed or changed to seek to 0
- A blank canvas overlay at frame 0 is acceptable — user can scrub or play to see detections
- No auto-scroll or auto-seek to first detection frame after load
- When switching between cases, reset ALL of: `currentFrame` → 0, `zoomLevel` → 1, `panOffset` → 0, `playbackRate` → 1 (also set on the video element), minimap viewport position → beginning
- These resets should happen in `loadCase()` alongside the existing resets

### Claude's Discretion
- Exact ordering of resets within `loadCase` relative to `data.value = ...` assignment
- Whether to keep or remove the `nextTick` wrapper (could simply change the argument to 0)
- How to handle the `videoRef.value.playbackRate` reset timing (video element may not exist yet)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BUG-01 | On case load, video starts at frame 0 (fix: `loadCase` must seek to frame 0, not the first detection frame) | `seekToFrame(0)` already exists and works correctly; only the argument passed to it is wrong |
| BUG-02 | Canvas overlay renders correct detections for frame 0 on initial load (dependent on BUG-01 fix and ensuring draw fires after video seeks) | `seekToFrame()` already calls `scheduleDraws(1)` on line 1196; `seeked` event also triggers a draw at line 1733; no additional draw wiring needed |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vue 3 | already installed | Reactivity (`ref`, `nextTick`, `watch`) | Project framework |
| @vue/test-utils | ^2.4.0 | Component mounting in tests | Already used in `App.spec.js` |
| vitest | ^2.0.0 | Test runner | Already configured via `npm test` |

No new dependencies required. This fix uses only existing Vue 3 primitives.

**Installation:** None needed.

## Architecture Patterns

### Relevant Code Locations (all in `YOLOVisualizer.vue`)

| Symbol | Line | Role |
|--------|------|------|
| `loadCase()` | 1060 | Async function — entry point for the fix |
| Reset block | 1069–1083 | Where `currentFrame`, `filterMode`, etc. are reset; add new resets here |
| `nextTick(() => seekToFrame(...))` | 1090 | The bug — change argument from `parsed.results[0]?.frame ?? 0` to `0` |
| `seekToFrame(frame)` | 1190 | Sets `currentFrame.value`, calls `scheduleDraws(1)`, seeks video element |
| `scheduleDraws(flags)` | 606 | RAF-based draw: flag 1 = overlay, 2 = raster, 4 = minimap |
| `zoomLevel` ref | 545 | Timeline zoom; must reset to 1 |
| `panOffset` ref | 546 | Timeline pan; must reset to 0 |
| `playbackRate` ref | 560 | Playback speed; must reset to 1 |
| `setRate(r)` | 1271 | Sets both `playbackRate.value` and `videoRef.value.playbackRate` — used for normal rate changes |
| `currentPartVideoUrl` watcher | 1667 | Loads new video src and calls `seekAndPlay` on `loadedmetadata` |
| Video sync watcher | 1686 | Attaches RVFC/rAF loop; fires on `[videoRef, data]` changes |

### Pattern: Reset then Seek

The established pattern in `loadCase()` is: reset reactive refs first (lines 1069–1083), then set `videoSrc.value = null`, then `nextTick()` for any DOM-dependent operations. The `playbackRate` ref reset fits cleanly in the reset block. The `videoRef.value.playbackRate` element property does NOT need setting here because `videoSrc.value = null` destroys the current video; the `setRate` path will re-apply it when the new video loads if needed.

For `panOffset`, the minimap viewport is derived from this ref reactively — resetting it to `0` automatically repositions the minimap.

### Anti-Patterns to Avoid
- **Setting `videoRef.value.playbackRate` in `loadCase()`:** The video element may not exist at reset time (src is cleared). Set only the ref; let `setRate` sync to the element.
- **Removing the `nextTick` wrapper entirely:** `seekToFrame(0)` still needs `nextTick` if any DOM reconciliation must complete first. Keeping the wrapper with argument `0` is the safest change.
- **Touching the `currentPartVideoUrl` watcher (line 1667):** That watcher handles src-switching and `loadedmetadata`; it is correct and must not be modified for this phase.
- **Touching the video sync watcher (line 1686):** It only runs while playing; no change needed.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Redrawing canvas after seek | Custom event listener | Existing `scheduleDraws(1)` call inside `seekToFrame()` | Already fires on line 1196; adding another listener creates duplicate draws |
| Syncing playback rate to element | Manual watcher | Existing `setRate()` at line 1271 | Already handles both ref and element; reuse it if post-load rate sync is ever needed |

## Common Pitfalls

### Pitfall 1: Setting element properties before element exists
**What goes wrong:** Calling `videoRef.value.playbackRate = 1` in `loadCase()` throws because `videoSrc.value = null` clears the video element before the new src loads.
**Why it happens:** The video element is conditionally rendered based on `videoSrc`.
**How to avoid:** Reset only the `playbackRate` ref in `loadCase()`; the element property is synced by `setRate()` or the watcher chain when the element is ready.
**Warning signs:** Console error "Cannot set properties of null" on case switch.

### Pitfall 2: Double-draw on initial load
**What goes wrong:** Adding a new `scheduleDraws` call in `loadCase` in addition to the existing one inside `seekToFrame` causes the overlay to draw twice.
**Why it happens:** `seekToFrame(0)` already calls `scheduleDraws(1)` on line 1196.
**How to avoid:** Do not add any additional draw scheduling calls; rely solely on `seekToFrame`.

### Pitfall 3: Wrong nextTick placement
**What goes wrong:** Moving resets to after `nextTick` means they fire after the new video source begins loading, potentially leaving stale state visible for one frame.
**How to avoid:** Keep all ref resets (including `zoomLevel`, `panOffset`, `playbackRate`) before `videoSrc.value = null` and the `nextTick` call, matching the existing pattern.

## Code Examples

### Current buggy line (line 1090)
```js
// Source: YOLOVisualizer.vue line 1090 (current)
nextTick(() => seekToFrame(parsed.results[0]?.frame ?? 0));
```

### Fixed line
```js
// Fixed: always seek to frame 0
nextTick(() => seekToFrame(0));
```

### Reset block after fix (lines 1069–1090 region)
```js
filteredSummary.value = null;
filterInfo.value = null;
filterMode.value = 'raw';
ppResult.value = null;
ppError.value = null;
activeCaseName.value = caseName;
enabledClasses.value = new Set(parsed.classes.map((_, i) => i));
jumpFilterClassIds.value = new Set();
// ... customOrder ...
videoSrc.value = null;
currentFrame.value = 0;
zoomLevel.value = 1;      // NEW
panOffset.value = 0;      // NEW
playbackRate.value = 1;   // NEW (ref only — element synced by setRate/watcher chain)

buildFrameSetChunked(parsed.results).then((set) => { rawFrameSet.value = set; });
nextTick(() => seekToFrame(0));   // CHANGED: was parsed.results[0]?.frame ?? 0
```

## State of the Art

No framework or pattern changes needed. This is a logic correction, not an architectural change.

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| `seekToFrame(firstDetectionFrame)` | `seekToFrame(0)` | Video opens at true start; BUG-01 resolved |
| No zoom/pan/rate reset on case switch | Reset `zoomLevel`, `panOffset`, `playbackRate` refs | Clean state on every case load |

## Open Questions

1. **Does `panOffset` reset fully reset minimap viewport?**
   - What we know: `panOffset` is a `ref(0)` that the minimap reads reactively.
   - What's unclear: Whether the minimap has any internal scroll state separate from `panOffset`.
   - Recommendation: Treat `panOffset.value = 0` as sufficient; verify visually during implementation.

2. **Is `playbackRate` reset visible to the user if the previous case had a non-1x rate?**
   - What we know: The UI rate selector displays `playbackRate.value`.
   - What's unclear: Whether the selector updates reactively or requires a user interaction.
   - Recommendation: Since `playbackRate` is a `ref`, the UI will update reactively. The element rate syncs on next video load via the existing `setRate` call path. Confirm during testing.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | vitest ^2.0.0 + @vue/test-utils ^2.4.0 |
| Config file | vite.config.js (vitest block) |
| Quick run command | `cd /home/jaeho/ent_cv/web/frontend && npm test` |
| Full suite command | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BUG-01 | `loadCase()` seeks to frame 0, not first detection frame | unit | `cd web/frontend && npx vitest run --reporter=verbose src/__tests__/YOLOVisualizer.spec.js` | ❌ Wave 0 |
| BUG-02 | Canvas overlay receives draw call for frame 0 after case load | unit | same file | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cd /home/jaeho/ent_cv/web/frontend && npm test`
- **Per wave merge:** `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `web/frontend/src/__tests__/YOLOVisualizer.spec.js` — covers BUG-01 and BUG-02; needs mock for `fetch`, `videoRef`, `nextTick`, and `scheduleDraws`

Note: `YOLOVisualizer.vue` is a large single-file component with direct DOM refs and fetch calls. Unit tests will require substantial mocking. A lightweight integration smoke test (mount + stub video + call `loadCase` + assert `currentFrame.value === 0`) is the most practical approach with `@vue/test-utils`.

## Sources

### Primary (HIGH confidence)
- Direct read of `YOLOVisualizer.vue` lines 537–610, 1060–1094, 1190–1208, 1271–1274, 1667–1703 — all state refs, `loadCase`, `seekToFrame`, `setRate`, `currentPartVideoUrl` watcher
- `.planning/phases/01-frame-state-fix/01-CONTEXT.md` — locked decisions and code context from discussion phase
- `.planning/REQUIREMENTS.md` — BUG-01 and BUG-02 definitions

### Secondary (MEDIUM confidence)
- `web/frontend/src/__tests__/App.spec.js` — confirms test infrastructure pattern (vitest + @vue/test-utils, stub-based mounting)
- `web/frontend/package.json` — confirms vitest ^2.0.0 and @vue/test-utils ^2.4.0 installed

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — code read directly, no external lookups needed
- Architecture: HIGH — fix scope confirmed by reading actual lines, cross-referenced with CONTEXT.md
- Pitfalls: HIGH — derived from code structure (videoRef null timing) and Vue 3 reactivity mechanics

**Research date:** 2026-03-07
**Valid until:** Stable indefinitely — pure local code analysis, no external API dependencies
