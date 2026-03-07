---
phase: 01-frame-state-fix
plan: 01
subsystem: ui
tags: [vue, vitest, vue-test-utils, tdd, bugfix]

requires: []
provides:
  - loadCase() seeks to frame 0 on every case load (not first detection frame)
  - zoomLevel, panOffset, playbackRate refs reset on case switch
  - Unit test suite for loadCase BUG-01 and BUG-02 behaviors
affects: []

tech-stack:
  added: []
  patterns:
    - "TDD with vitest + @vue/test-utils mount-based approach for YOLOVisualizer"
    - "Access script setup internal state in tests via wrapper.getCurrentComponent().setupState (auto-unwraps refs)"

key-files:
  created:
    - web/frontend/src/__tests__/YOLOVisualizer.spec.js
  modified:
    - web/frontend/src/components/YOLOVisualizer.vue

key-decisions:
  - "Do not call setRate() or set videoRef.value.playbackRate in loadCase — video element does not exist at reset time; reset only the ref"
  - "Frame 0 internally displays as 'Frame 1' in the UI (line 115: :value='currentFrame + 1') — this is correct 1-indexed display behavior, no change needed"
  - "Mount-based test approach used (not composable extraction) — setupState proxy write-through enables ref mutation in tests"

patterns-established:
  - "loadCase reset block order: null-out filteredSummary/filterInfo/filterMode/ppResult/ppError → set activeCaseName/enabledClasses/jumpFilterClassIds → null videoSrc → reset currentFrame/zoomLevel/panOffset/playbackRate → buildFrameSetChunked → nextTick(seekToFrame)"

requirements-completed: [BUG-01, BUG-02]

duration: ~15min
completed: 2026-03-07
---

# Phase 01-01: Frame State Fix Summary

**Fixed loadCase() async init so every case opens at frame 0 with correct canvas overlay, and case switching resets zoomLevel/panOffset/playbackRate refs**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-07T02:54:00Z
- **Completed:** 2026-03-07T08:00:00Z
- **Tasks:** 3 (2 auto + 1 human-verify)
- **Files modified:** 2

## Accomplishments

- Replaced `seekToFrame(parsed.results[0]?.frame ?? 0)` with `seekToFrame(0)` — video no longer jumps to first detection frame on load
- Added `zoomLevel.value = 1`, `panOffset.value = 0`, `playbackRate.value = 1` resets to loadCase reset block — view state now correctly clears between cases
- Created 3-test unit suite covering BUG-01 and BUG-02 using vitest + @vue/test-utils mount-based approach; all pass GREEN

## Task Commits

Each task was committed atomically:

1. **Task 1: Write failing tests for BUG-01 and BUG-02** - `bf5d67a` (test)
2. **Task 2: Fix loadCase() — seek to frame 0, reset state refs** - `5f60383` (fix)
3. **Task 3: Verify fix in browser** - human-approved (no commit)

## Files Created/Modified

- `web/frontend/src/__tests__/YOLOVisualizer.spec.js` - 3 unit tests asserting frame 0 start and view state reset on case load/switch
- `web/frontend/src/components/YOLOVisualizer.vue` - Two changes in loadCase(): seek target fixed to 0, three ref resets added

## Decisions Made

- **Do not call setRate() in loadCase**: The video element does not exist at reset time (videoSrc is null). Only the `playbackRate` ref is reset; `videoRef.value.playbackRate` will sync when the video loads via the existing watcher.
- **Frame display is 1-indexed by design**: The UI binds `:value="currentFrame + 1"` (line 115), so frame 0 correctly shows as "Frame 1". Confirmed by human during verification — no change needed.
- **Mount-based tests**: Full component mount with stubs used instead of composable extraction. The `setupState` proxy supports write-through (assigning to `state.zoomLevel` routes through the ref's `.value` setter), enabling the case-switch test to mutate state and verify reset.

## Deviations from Plan

None — plan executed exactly as written. Both changes were precisely as specified in the plan's action blocks.

## Issues Encountered

- Initial test approach used `state.currentFrame.value` (wrong — setupState auto-unwraps refs). Fixed to `state.currentFrame` (the unwrapped number). Tests then correctly showed RED before the fix.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- BUG-01 and BUG-02 are resolved and verified in browser
- Test infrastructure for YOLOVisualizer is established (mount pattern + setupState access)
- Ready for remaining phases (Phase 2 onward)

---
*Phase: 01-frame-state-fix*
*Completed: 2026-03-07*
