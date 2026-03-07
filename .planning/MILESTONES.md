# Milestones

## v1.0 Web Viewer Polish (Shipped: 2026-03-07)

**Phases completed:** 4 phases, 4 plans, 0 tasks

**Key accomplishments:**
- Fixed loadCase() frame-0 initialization — video and canvas overlay now start at frame 0 on every case load (BUG-01, BUG-02)
- Dual-gate loading screen (API fetch + video loadedmetadata) with spinner overlay in YOLOVisualizer (LOAD-01, LOAD-02)
- Global dark-themed scrollbar CSS in index.html covering Chromium (::-webkit-scrollbar) and Firefox (scrollbar-width/color) (SCROLL-01, SCROLL-02)
- CasePicker bounded-height overflow fix (height:100vh + overflow-y:auto) enabling internal scroll past viewport (PICK-01)
- ResizeObserver-driven transitions matrix maintaining 1:1 square aspect ratio at all panel widths via watch(matrixContainerRef) (TRANS-01, TRANS-02)

**Known gaps:**
- Phase 01 (Frame State Fix) missing VERIFICATION.md — gsd-verifier was not run. BUG-01/BUG-02 confirmed working by integration checker and human browser verification in SUMMARY.md.

---

