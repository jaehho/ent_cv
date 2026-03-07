# Feature Research

**Domain:** Video review web app (surgical instrument detection viewer — Vue 3 + Django)
**Researched:** 2026-03-07
**Confidence:** HIGH (active requirements are well-scoped; UX patterns are stable)

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist in any competent video review tool. Missing these makes the app feel broken.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Loading indicator while case data fetches | Any data-fetching app shows a spinner or skeleton; blank screen during load signals a broken page | LOW | Show during API fetch + video `loadeddata` event; hide when both resolve |
| Correct initial frame state (frame 0) | Video always opens at the beginning; overlay showing wrong frame on load breaks the core value proposition | LOW | Clear frame counter state on route/case change; listen for `loadedmetadata` or `seeked` event to reset canvas |
| Canvas overlay pixel-perfect to video | If bounding boxes don't align to the video on first render, trust in the system collapses | MEDIUM | Canvas must read `video.videoWidth/Height` and `getBoundingClientRect()` after layout; resize observer needed |
| Scrollable content when items overflow viewport | Case list or instrument classes that clip without scroll is a jarring, unpolished experience | LOW | `overflow-y: auto` with min-height on container; applies to CasePicker grid and classes panel |
| Scrollbars that don't visually clash with the app | Browser-default scrollbars in a dark themed UI create obvious inconsistency | LOW | CSS `::-webkit-scrollbar` + `scrollbar-width: thin` + `scrollbar-color` for cross-browser; apply globally via `:root` or `body` |
| Aspect-ratio-correct analysis plots | A non-square correlation/transition matrix plot is visually misleading for a symmetric matrix | LOW | CSS `aspect-ratio: 1 / 1` on the plot container; `ResizeObserver` or reactive panel width to trigger re-render |

### Differentiators (Competitive Advantage)

Features that go beyond expectation for a surgical annotation review tool.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Frame-accurate overlay using `requestVideoFrameCallback` | `timeupdate` fires ~4x/sec and can miss frames; rVFC fires on every compositor frame, keeping overlay and video in lockstep | MEDIUM | Replace `timeupdate` listener with `video.requestVideoFrameCallback()`; fall back to `requestAnimationFrame` if unsupported (Firefox) |
| Progressive loading feedback (skeleton vs. spinner) | Skeleton screens that match the layout reduce perceived load time vs. a generic spinner | MEDIUM | Requires layout-matched placeholder markup; high effort for an internal tool |
| Loading progress bar at top of page (nprogress-style) | Surgical context: reviewers run many cases back-to-back; a thin top bar signals progress without blocking the view | LOW | CSS-only top bar animating width; no library needed |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Custom scrollbar that hides completely until hover (overlay scrollbars) | Looks sleek; modern macOS aesthetic | Hides affordance; users may not know content is scrollable; accessibility concern | Thin always-visible scrollbar (`scrollbar-width: thin`); subtle color matching background |
| Heavy animation/transition on loading state | Feels polished | Adds perceived latency; animation runs while user is waiting; annoying if they see it 50 times/day | Instant fade-in (100ms) when content is ready; no loading animation drawn out |
| Per-scrollbar library (OverlayScrollbars, SimpleBar) | Cross-browser consistency | New dependency; PROJECT.md explicitly says "no new dependencies" | CSS `scrollbar-width` + `::-webkit-scrollbar` covers Chrome/Edge/Firefox/Safari adequately |
| Animated skeleton loaders | Reduces perceived load time | Significant markup overhead for an internal single-user tool; effort outweighs benefit | Single centered spinner with text label ("Loading case…") is sufficient |
| Canvas drawn via `requestAnimationFrame` loop | Simple mental model | Wastes GPU time drawing identical frames; causes canvas to lag video by up to one rAF cycle | Draw only on `timeupdate` or `seeked` + `requestVideoFrameCallback` for playback |

## Feature Dependencies

```
Frame-state reset (frame 0 on load)
    └──required by──> Canvas overlay alignment on load
                          └──required by──> Core value (overlay matches video)

Loading indicator
    └──depends on──> Knowing when BOTH API fetch AND video loadeddata have resolved

Aspect-ratio square plot
    └──depends on──> Panel width being available as reactive data (already exists via leftPanelWidth)

Custom scrollbar (global)
    └──enables──> Classes panel scroll UX (most visible location)
    └──enables──> CasePicker scroll UX

requestVideoFrameCallback (differentiator)
    └──replaces──> timeupdate listener (not additive — swap, don't layer)
    └──conflicts with──> any frame counter logic based on Math.floor(currentTime * fps)
                         (must recalculate frame from rVFC metadata.mediaTime instead)
```

### Dependency Notes

- **Frame-state reset requires canvas alignment fix:** The frame counter bug and the canvas misalignment are the same root cause. Fix once, fix both. Don't address canvas size in isolation without also resetting frame counter state.
- **Loading indicator depends on both fetch + video ready:** API data resolving alone is not enough — video `loadeddata` must also fire before hiding the loader, otherwise the video element renders blank while the overlay appears.
- **Square plot depends on reactive panel width:** The panel resize mechanism already exists. The plot just needs to bind to that reactive value to trigger a CSS or Chart.js resize.

## MVP Definition

### Launch With (this milestone)

These are the active requirements from PROJECT.md — all must ship together because they share the frame-state root cause.

- [ ] Loading screen — display while fetch + video init resolve; hide when both ready
- [ ] Frame 0 reset on case load — clear `currentFrame` state; re-derive from `seeked`/`loadedmetadata` event
- [ ] Canvas overlay position/scale fix — resize observer or `loadedmetadata` handler triggers canvas dimension recalculation
- [ ] Custom scrollbar styling (global) — `scrollbar-width: thin` + `scrollbar-color` + `::-webkit-scrollbar` block in global CSS
- [ ] CasePicker scrollable — `overflow-y: auto` on case grid container
- [ ] Transitions matrix square aspect ratio — `aspect-ratio: 1` on plot wrapper; reactive to panel width

### Add After Validation (v1.x)

- [ ] `requestVideoFrameCallback` for playback sync — only if users report overlay lag during fast playback; adds complexity for a problem that may not be noticeable at normal speeds

### Future Consideration (v2+)

- [ ] Skeleton loading screens — only if load time grows beyond ~2s; currently likely fast enough for a spinner
- [ ] Keyboard shortcuts for frame stepping — ergonomic for surgical review sessions; deferred to avoid scope creep

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Frame 0 reset + canvas alignment | HIGH (core value broken without it) | LOW | P1 |
| Loading indicator | HIGH (blank screen = broken perception) | LOW | P1 |
| Custom scrollbar (global) | MEDIUM (aesthetic polish) | LOW | P1 |
| CasePicker scrollable | MEDIUM (usability when cases > viewport) | LOW | P1 |
| Square aspect ratio plot | MEDIUM (data integrity perception) | LOW | P1 |
| rVFC frame sync (differentiator) | LOW-MEDIUM (latent; not user-reported) | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

This is an internal surgical review tool, not a consumer product. Relevant reference class is annotation review UIs (CVAT, Label Studio, Roboflow) and video annotation platforms.

| Feature | CVAT / Label Studio | Roboflow Annotate | Our Approach |
|---------|---------------------|-------------------|--------------|
| Loading state | Full-screen spinner with label text | Skeleton + progress bar | Simple centered spinner with "Loading case…" text — sufficient for internal tool |
| Scrollbar styling | Default or thin native | Thin, matches dark theme | Thin, dark-themed via CSS; no library |
| Canvas overlay sync | Per-frame draw on seek events | rVFC or timeupdate | Fix timeupdate-based draw to reset correctly on load; rVFC as future upgrade |
| Plot aspect ratio | Enforced via chart library config | N/A | CSS `aspect-ratio: 1` + ResizeObserver |

## Sources

- [MDN: HTMLVideoElement.requestVideoFrameCallback()](https://developer.mozilla.org/en-US/docs/Web/API/HTMLVideoElement/requestVideoFrameCallback)
- [web.dev: Perform efficient per-video-frame operations](https://web.dev/requestvideoframecallback-rvfc/)
- [Chrome for Developers: Scrollbar styling](https://developer.chrome.com/docs/css-ui/scrollbar-styling)
- [MDN: ::-webkit-scrollbar](https://developer.mozilla.org/en-US/docs/Web/CSS/::-webkit-scrollbar)
- [CSS-Tricks: The Current State of Styling Scrollbars in CSS](https://css-tricks.com/the-current-state-of-styling-scrollbars-in-css/)
- [ishadeed: Custom Scrollbars in CSS](https://ishadeed.com/article/custom-scrollbars-css/)

---
*Feature research for: ENT CV Web Viewer — UI polish milestone*
*Researched: 2026-03-07*
