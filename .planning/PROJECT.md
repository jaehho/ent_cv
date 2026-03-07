# ENT CV Web Viewer

## What This Is

ENT CV is a surgical instrument detection system built on YOLO, with a Django + Vue 3 web viewer for reviewing predictions. The viewer plays raw surgical videos with YOLO bounding box overlays, instrument timelines, and a class/transition analysis panel. This project focuses on polish and bug fixes to the web viewer.

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

### Active

- [ ] Loading screen displayed while case data fetches and video initializes
- [ ] Case picker page scrollable when cases overflow viewport
- [ ] Custom scrollbar styling matching site aesthetic applied globally (especially classes section in YOLOVisualizer)
- [ ] Transitions matrix maintains 1:1 square aspect ratio and auto-resizes when panel width changes
- [ ] Bug fix: frame counter and canvas overlay reset to frame 0 on case load (currently initializes to a stale frame ~40-500)
- [ ] Bug fix: canvas overlay position/scale matches video on initial load (related to frame counter bug)

### Out of Scope

- New ML pipeline features — not part of this milestone
- Authentication changes — existing session auth is sufficient
- New API endpoints — all fixes are frontend-side

## Context

The YOLOVisualizer is a single large Vue 3 component (`web/frontend/src/components/YOLOVisualizer.vue`). The frame counter bug is a state initialization issue: when a case loads, the frame counter and canvas overlay state are not reset to 0 despite the video element being at frame 1. This causes the overlay to render detections for the wrong frame on initial display.

The transitions section renders a square correlation-matrix-style plot. Currently it does not respond to panel width changes — it should maintain a square aspect ratio and resize automatically as the adjustable panel changes width.

Scrollbar styling is currently default browser styling. The classes section (instrument list) in YOLOVisualizer is where the mismatch with site aesthetics is most visible.

## Constraints

- **Tech stack**: Vue 3 (no TypeScript), Django 6, Vite 5 — no framework changes
- **No new dependencies**: Prefer CSS and Vue reactive solutions
- **Frontend-only**: All active requirements are frontend changes; backend untouched

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Frontend-only scope | All issues are in YOLOVisualizer.vue and CasePicker.vue | — Pending |
| CSS custom scrollbar | Native CSS ::-webkit-scrollbar + scrollbar-width for cross-browser | — Pending |

---
*Last updated: 2026-03-07 after initialization*
