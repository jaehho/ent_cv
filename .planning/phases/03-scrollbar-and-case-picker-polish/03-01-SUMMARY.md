---
phase: 03-scrollbar-and-case-picker-polish
plan: 01
subsystem: ui
tags: [css, scrollbar, vue, vitest]

# Dependency graph
requires: []
provides:
  - Global dark-themed scrollbar CSS (webkit + Firefox) in index.html
  - CasePicker overflow-y: auto fix — case list scrolls past viewport
  - CasePicker.spec.js unit tests (PICK-01)
affects: [future UI phases, YOLOVisualizer (classes panel scrollbar appearance)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Global CSS rules go in index.html inline <style> block — not in scoped SFC styles (pseudo-elements need unscoped rules)"
    - "Test scoped Vue CSS in vitest via ?raw import to inspect SFC source text"

key-files:
  created:
    - web/frontend/src/__tests__/CasePicker.spec.js
  modified:
    - web/frontend/index.html
    - web/frontend/src/components/CasePicker.vue

key-decisions:
  - "Scrollbar CSS placed in index.html inline style block — scoped SFC styles drop pseudo-element rules due to data-v-xxx attribute injection"
  - "CasePicker test asserts overflow-y via ?raw import of SFC source — jsdom does not apply scoped <style> blocks"
  - "align-items changed from center to flex-start on .upload-root — center + min-height: 100vh splits overflow above/below, top half unreachable with body overflow: hidden"
  - "height:100vh (not min-height:100vh) on .upload-root — min-height lets element grow beyond viewport requiring body scroll (disabled); fixed height creates bounded internal scroll container"

patterns-established:
  - "Global pseudo-element CSS (scrollbars, selections) must live in index.html, not in component <style scoped>"
  - "Use ?raw import in vitest to verify scoped CSS properties that jsdom cannot evaluate"

requirements-completed: [SCROLL-01, SCROLL-02, PICK-01]

# Metrics
duration: 8min
completed: 2026-03-07
---

# Phase 3 Plan 01: Scrollbar and Case Picker Polish Summary

**Dark-themed 6px scrollbars via dual-API CSS (webkit + Firefox scrollbar-color) and CasePicker flex overflow fix enabling scroll past viewport**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-07T03:51:00Z
- **Completed:** 2026-03-07T03:53:00Z
- **Tasks:** 3 auto tasks completed (Task 3 = human-verify checkpoint)
- **Files modified:** 3

## Accomplishments
- Global scrollbar CSS in index.html — thin dark scrollbar in both Chromium (webkit pseudo-elements) and Firefox (scrollbar-width/scrollbar-color Baseline 2022)
- CasePicker.vue .upload-root changed to align-items: flex-start + overflow-y: auto — case list is fully scrollable when it exceeds viewport height
- CasePicker.spec.js test suite — two tests covering presence of .upload-root and overflow-y contract

## Task Commits

Each task was committed atomically:

1. **Task 0: CasePicker test scaffold (RED)** - `3c63bc6` (test)
2. **Task 0 refinement: raw SFC source assertion** - `693d9a7` (test)
3. **Task 1: Global scrollbar CSS** - `26f0403` (feat)
4. **Task 2: CasePicker overflow layout fix** - `54fa96b` (feat)
5. **Task 3 fix: height:100vh root cause fix (post-checkpoint)** - `e6ffa74` (fix)

_Task 3 was a human-verify checkpoint — user reported case picker not scrollable; root cause fixed and re-verified._

## Files Created/Modified
- `web/frontend/index.html` — Added webkit scrollbar rules + Firefox scrollbar-width/color
- `web/frontend/src/components/CasePicker.vue` — align-items: flex-start, overflow-y: auto, margin: 0 auto on .upload-center
- `web/frontend/src/__tests__/CasePicker.spec.js` — Two-test suite for PICK-01

## Decisions Made
- Scrollbar CSS in index.html inline block (not SFC scoped) — pseudo-elements cannot carry scoped data-v attribute
- CasePicker test uses `?raw` SFC import to check source for overflow-y rule — avoids jsdom scoped-style limitation
- align-items: flex-start rather than center — center mode splits overflow equally, making top half unreachable under body overflow: hidden

## Deviations from Plan

**1. [Rule 1 - Bug] Test assertion approach changed during Task 0**
- **Found during:** Task 0 verification run
- **Issue:** Plan suggested `element.style.overflowY` check, but this returns empty string for scoped CSS in jsdom — test would never turn GREEN after fix
- **Fix:** Switched to `?raw` import and `toContain("overflow-y: auto")` — directly checks SFC source text
- **Files modified:** web/frontend/src/__tests__/CasePicker.spec.js
- **Verification:** Test fails RED before Task 2 fix, GREEN after
- **Committed in:** 693d9a7

**2. [Rule 1 - Bug] Root cause fix: min-height vs height on .upload-root (post-checkpoint)**
- **Found during:** Task 3 human-verify — user reported case picker still not scrollable after Task 2
- **Issue:** `min-height: 100vh` lets `.upload-root` grow beyond viewport, but body scroll is disabled (`overflow: hidden`). The element had no bounded height so `overflow-y: auto` had no effect.
- **Fix:** Changed to `height: 100vh` — creates a bounded container so `overflow-y: auto` scrolls internally, independent of body scroll
- **Files modified:** web/frontend/src/components/CasePicker.vue
- **Verification:** Full vitest suite green (16/16); browser-verified by user
- **Committed in:** e6ffa74

---

**Total deviations:** 2 auto-fixed (Rule 1 x2 — wrong test assertion, incorrect CSS height approach in plan)
**Impact on plan:** Both fixes essential for correctness. No scope creep.

## Issues Encountered
- jsdom does not apply Vue scoped `<style>` blocks — documented in test file comments. Resolved via `?raw` import pattern.
- Original Task 2 plan instruction used `min-height: 100vh` which cannot scroll when `body { overflow: hidden }` — fixed post-checkpoint via Rule 1.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All requirements SCROLL-01, SCROLL-02, PICK-01 satisfied and browser-verified
- Plan fully complete — no remaining blockers

---
*Phase: 03-scrollbar-and-case-picker-polish*
*Completed: 2026-03-07*
