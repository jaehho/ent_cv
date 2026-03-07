# ENT CV Web Viewer

## What This Is

ENT CV is a surgical instrument detection system built on YOLO, with a Django + Vue 3 web viewer for reviewing predictions. The viewer plays raw surgical videos with YOLO bounding box overlays, instrument timelines, and a class/transition analysis panel. v1.0 delivered a polish pass: frame-state bug fixes, loading screen, custom scrollbars, and a responsive transitions matrix.

## Core Value

The viewer must accurately show instrument detections synchronized to video playback — the overlay and frame state must always match what's playing.

## Requirements

### Validated

- ✓ Case picker lists available cases — existing
- ✓ Video playback with YOLO bounding box overlays — existing
- ✓ Instrument class timeline panel (scrollable list) — existing
- ✓ Transitions matrix panel (correlation-matrix-style plot) — existing
- ✓ On-demand postprocessing via UI — existing
- ✓ Session-based authentication — existing
- ✓ Adjustable panel widths — existing
- ✓ Bug fix: frame counter and canvas overlay reset to frame 0 on case load — v1.0
- ✓ Loading screen displayed while case data fetches and video initializes — v1.0
- ✓ Case picker page scrollable when cases overflow viewport — v1.0
- ✓ Custom scrollbar styling matching site aesthetic applied globally — v1.0
- ✓ Transitions matrix maintains 1:1 square aspect ratio and auto-resizes when panel width changes — v1.0

### Active

(None — defining next milestone)

### Out of Scope

- New ML pipeline features — not part of web viewer polish
- Authentication changes — existing session auth is sufficient
- New API endpoints — all v1.0 fixes were frontend-side
- `requestVideoFrameCallback` adoption (SYNC-01) — deferred to v2, current timeupdate works post-bug-fix

## Context

**Shipped v1.0 on 2026-03-07.** 4 phases, 4 plans. ~2,956 LOC in modified files (YOLOVisualizer.vue is the main component at 2,287 LOC). 21 unit tests passing across 4 test files.

**Tech stack:** Vue 3 (no TypeScript), Django 6, Vite 5, Vitest + @vue/test-utils for frontend tests.

**Known tech debt from v1.0:**
- Phase 01 VERIFICATION.md missing (administrative gap only — fix confirmed working)
- Scrollbar visual rendering and ResizeObserver real-drag behavior require human browser confirmation (visual-only, not programmatically testable)
- Transitions matrix renders at 1×1px briefly on first paint before ResizeObserver fires first callback (cosmetic flash)

## Constraints

- **Tech stack**: Vue 3 (no TypeScript), Django 6, Vite 5 — no framework changes
- **No new dependencies**: Prefer CSS and Vue reactive solutions
- **Frontend-only**: All v1.0 requirements were frontend changes; backend untouched

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Frontend-only scope | All issues are in YOLOVisualizer.vue and CasePicker.vue | ✓ Good — backend untouched, all fixes isolated |
| CSS custom scrollbar in index.html | Scoped SFC pseudo-elements get data-v attribute injection, breaking ::-webkit-scrollbar | ✓ Good — unscoped global rules work correctly |
| height:100vh (not min-height) on CasePicker .upload-root | min-height lets element grow beyond viewport, body overflow:hidden prevents scroll; bounded height enables internal overflow-y:auto scroll | ✓ Good — root cause fix verified by user |
| videoReady set in seekAndPlay (loadedmetadata callback), not watcher entry | Ensures video src has actually loaded metadata before clearing loading overlay | ✓ Good — LOAD-02 satisfied |
| watch(matrixContainerRef) instead of onMounted guard | .right-panel is inside v-if="data", so ref is null at onMounted time; watcher fires reactively | ✓ Good — correct for conditional template refs |
| Do not call setRate() in loadCase | videoRef is null at reset time; reset only the playbackRate ref, let existing watcher sync to video element | ✓ Good — avoids null reference error |
| Mount-based tests with setupState proxy | Composable extraction would require refactoring; setupState write-through enables ref mutation in tests | ✓ Good — all 21 tests pass |
| ?raw SFC import in vitest for scoped CSS assertions | jsdom does not apply scoped Vue <style> blocks; raw source inspection is the correct workaround | ✓ Good — PICK-01 test correctly turns RED/GREEN |

---
*Last updated: 2026-03-07 after v1.0 milestone*
