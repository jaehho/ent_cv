# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Web Viewer Polish

**Shipped:** 2026-03-07
**Phases:** 4 | **Plans:** 4 | **Sessions:** 1

### What Was Built
- Frame-0 initialization fix — loadCase() now seeks to frame 0 and resets zoomLevel/panOffset/playbackRate on every case load
- Dual-gate loading screen — isLoading computed gates viewer behind both API fetch (dataReady) and video loadedmetadata (videoReady)
- Global dark-themed scrollbar CSS via index.html inline style block (Chromium + Firefox dual-API)
- CasePicker overflow fix — bounded height:100vh container with overflow-y:auto for internal scrolling
- ResizeObserver-driven transitions matrix — watch(matrixContainerRef) fires after v-if renders, drives dynamic squareSize

### What Worked
- TDD approach: write failing tests first, then implement — each phase had RED tests that turned GREEN with the fix
- Reading existing code before planning — understanding the loadCase() async sequencing prevented introducing new bugs
- Checking template ref reactivity before attaching observers — identified the v-if="data" null ref problem before implementation
- Parallel phase execution pattern — each phase small enough to complete in one focused session (~8-20 min each)

### What Was Inefficient
- Phase 01 skipped the gsd-verifier — VERIFICATION.md was never created, requiring a known-gap note in the milestone audit
- CasePicker fix required a post-checkpoint iteration (min-height → height:100vh) — the plan's original instruction used min-height which doesn't work with body overflow:hidden; would have been caught by researching the existing body CSS first
- Test assertion approach in Phase 03 required refinement (element.style.overflowY → ?raw SFC import) — jsdom limitation should be documented upfront for scoped CSS tests

### Patterns Established
- **Global pseudo-element CSS goes in index.html**: Scoped SFC `<style>` blocks get `data-v-xxx` attribute injection that breaks `::-webkit-scrollbar` and similar pseudo-elements
- **Use `?raw` SFC import in vitest for scoped CSS assertions**: jsdom cannot evaluate scoped styles; raw source text inspection is the correct workaround
- **watch(templateRef) instead of onMounted for v-if elements**: Template refs inside `v-if` are null at onMounted time; a watcher on the ref fires reactively when the condition becomes true
- **Dual-gate loading pattern**: Two boolean refs (dataReady, videoReady) combined in one computed (isLoading) — clean, extendable to N async conditions
- **Mount-based tests with setupState proxy**: Full component mount + `wrapper.getCurrentComponent().setupState` for ref access; setupState auto-unwraps refs so use `state.myRef` not `state.myRef.value`

### Key Lessons
1. **Always check what the video element's null state is before writing to it** — videoRef is null during loadCase() because the viewer is gated behind v-if; any direct DOM access during reset must be null-guarded or deferred
2. **CSS properties that depend on a bounded container need height, not min-height** — min-height lets the element grow unbounded; overflow-y:auto only activates when content exceeds a fixed height
3. **Run gsd-verifier for every phase** — even when the SUMMARY documents human browser verification, the formal VERIFICATION.md provides the audit trail needed for milestone completion without gaps

### Cost Observations
- Model mix: 100% sonnet (claude-sonnet-4-6)
- Sessions: 1 day sprint
- Notable: All 4 phases completed in a single day with 49 commits; TDD kept each phase focused and fast

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 1 day | 4 | Initial baseline — TDD + mount-based Vue tests established |

### Cumulative Quality

| Milestone | Tests | Files Modified | Zero-Dep Additions |
|-----------|-------|----------------|-------------------|
| v1.0 | 21 | 5 | 0 |

### Top Lessons (Verified Across Milestones)

1. Read existing component code before planning to understand async sequencing and null states
2. Global pseudo-element CSS must be unscoped (index.html, not SFC `<style scoped>`)
