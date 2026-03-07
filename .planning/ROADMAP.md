# Roadmap: ENT CV Web Viewer Polish

## Overview

A focused polish pass on the existing Vue 3 + Django surgical instrument detection viewer. Four phases deliver in dependency order: fix the frame initialization root cause first, then add the loading screen that depends on it, then layer in independent cosmetic improvements (scrollbars, case picker), and finally make the transitions matrix responsive.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Frame State Fix** - Reset frame counter and canvas overlay to frame 0 on every case load (completed 2026-03-07)
- [x] **Phase 2: Loading Screen** - Show a loading indicator while API fetch and video initialization both complete (completed 2026-03-07)
- [x] **Phase 3: Scrollbar and Case Picker Polish** - Apply custom scrollbar styling globally and make the case picker scrollable (completed 2026-03-07)
- [ ] **Phase 4: Transitions Matrix Responsive Resize** - Maintain square aspect ratio as panel width changes via ResizeObserver

## Phase Details

### Phase 1: Frame State Fix
**Goal**: The viewer always starts at frame 0 with a correctly positioned canvas overlay on every case load
**Depends on**: Nothing (first phase)
**Requirements**: BUG-01, BUG-02
**Success Criteria** (what must be TRUE):
  1. When a case loads, the video starts at frame 0 — no stale frame from a previous case or initial state
  2. The canvas overlay renders the correct bounding boxes for frame 0 immediately on first display, without requiring any user interaction
  3. Switching between cases resets frame state cleanly each time
**Plans**: 1 plan

Plans:
- [ ] 01-01-PLAN.md — Fix loadCase() seek target and reset zoomLevel/panOffset/playbackRate refs (BUG-01, BUG-02)

### Phase 2: Loading Screen
**Goal**: Users see a loading indicator from case selection until the viewer is ready to play, never a blank or broken page
**Depends on**: Phase 1
**Requirements**: LOAD-01, LOAD-02
**Success Criteria** (what must be TRUE):
  1. Selecting a case immediately shows a loading screen — the viewer content is hidden until ready
  2. The loading screen is visible for the full duration of both the API fetch and video initialization
  3. The viewer (video + overlays) appears only after both the API response is received and the video loadedmetadata event has fired
**Plans**: 1 plan

Plans:
- [ ] 02-01-PLAN.md — Add dataReady/videoReady/isLoading gate + spinner overlay in YOLOVisualizer (LOAD-01, LOAD-02)

### Phase 3: Scrollbar and Case Picker Polish
**Goal**: Scrollable areas use custom dark-themed scrollbars and the case picker handles overflow correctly
**Depends on**: Phase 2
**Requirements**: SCROLL-01, SCROLL-02, PICK-01
**Success Criteria** (what must be TRUE):
  1. Scrollbars in the classes panel and all other scrollable areas display a thin, dark-themed style consistent with the site aesthetic
  2. The custom scrollbar styling is visible in both Chromium-based browsers and Firefox
  3. The case picker page is scrollable when the number of cases exceeds the viewport height — no cases are clipped or unreachable
**Plans**: 1 plan

Plans:
- [ ] 03-01-PLAN.md — Add global scrollbar CSS + fix CasePicker overflow layout (SCROLL-01, SCROLL-02, PICK-01)

### Phase 4: Transitions Matrix Responsive Resize
**Goal**: The transitions matrix always renders as a square and reflows correctly as the panel is resized
**Depends on**: Phase 3
**Requirements**: TRANS-01, TRANS-02
**Success Criteria** (what must be TRUE):
  1. Dragging the panel divider to change the right panel width causes the transitions matrix to resize and remain square
  2. The transitions matrix is never distorted into a non-square rectangle at any panel width
  3. When the component is unmounted, the ResizeObserver is cleaned up with no console errors
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Frame State Fix | 1/1 | Complete   | 2026-03-07 |
| 2. Loading Screen | 1/1 | Complete    | 2026-03-07 |
| 3. Scrollbar and Case Picker Polish | 1/1 | Complete    | 2026-03-07 |
| 4. Transitions Matrix Responsive Resize | 0/? | Not started | - |
