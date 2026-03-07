# Requirements: ENT CV Web Viewer Polish

**Defined:** 2026-03-07
**Core Value:** The viewer must accurately show instrument detections synchronized to video playback — the overlay and frame state must always match what's playing.

## v1 Requirements

Requirements for this polish milestone. All are frontend-only changes.

### Bug Fixes

- [x] **BUG-01**: On case load, video starts at frame 0 (fix: `loadCase` must seek to frame 0, not the first detection frame)
- [x] **BUG-02**: Canvas overlay renders correct detections for frame 0 on initial load (dependent on BUG-01 fix and ensuring draw fires after video seeks)

### Loading

- [ ] **LOAD-01**: A loading screen is displayed from the moment a case is selected until both the API response is received AND the video `loadedmetadata` event has fired
- [ ] **LOAD-02**: The loading screen is hidden and the viewer is shown only when the video is ready to play from frame 0

### Scrollbars

- [ ] **SCROLL-01**: Custom scrollbar styling (matching site aesthetic) is applied globally via CSS, covering both Chromium/WebKit (`::-webkit-scrollbar`) and Firefox (`scrollbar-width` + `scrollbar-color`)
- [ ] **SCROLL-02**: The classes section in YOLOVisualizer and all other scrollable areas display the custom scrollbar

### Case Picker

- [ ] **PICK-01**: The case picker page is scrollable when the case list overflows the viewport

### Transitions Matrix

- [ ] **TRANS-01**: The transitions matrix panel uses a `ResizeObserver` on its container to detect panel width changes
- [ ] **TRANS-02**: The transitions matrix always renders at a 1:1 square aspect ratio, resizing automatically as the panel width changes

## v2 Requirements

### Video Sync

- **SYNC-01**: Adopt `requestVideoFrameCallback` for frame-accurate canvas overlay sync (currently `timeupdate` — works once frame-reset bug is fixed, but rVFC is more accurate)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Backend API changes | All issues are frontend-only |
| New npm dependencies | Pure CSS + Vue reactivity sufficient |
| ML pipeline changes | Not part of this milestone |
| Auth changes | Existing session auth is sufficient |
| Postprocess button loading state | Low priority, not requested |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BUG-01 | Phase 1 | Complete |
| BUG-02 | Phase 1 | Complete |
| LOAD-01 | Phase 2 | Pending |
| LOAD-02 | Phase 2 | Pending |
| SCROLL-01 | Phase 3 | Pending |
| SCROLL-02 | Phase 3 | Pending |
| PICK-01 | Phase 3 | Pending |
| TRANS-01 | Phase 4 | Pending |
| TRANS-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-07*
*Last updated: 2026-03-07 after initial definition*
