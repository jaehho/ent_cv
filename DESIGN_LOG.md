# Design Log

Running, append-only log of design decisions and non-trivial changes. Newest on top.

## 2026-05-20 — Raw-vs-filtered comparison: parked

**Context.** Web viewer's overlay should let the user understand what the postprocess filter changes between the model's raw predictions and the kept "filtered" predictions. Cycled through several visual treatments, none landed.

**Tried (newest → oldest).**

- **Two-canvas vertical compare stack + collapsible side panels** (commits `d6e965a`, `7168c9c`). Single `<video>` decode feeding two stacked canvases via `ctx.drawImage(video, …)`; each canvas paints the full frame + its own bbox layer (raw on top, filtered on bottom). Side panels could collapse to 36 px icon strips to give the stack more vertical real estate.
  - **User reaction:** *"videos are too small."* Even with both side panels collapsed, the two stacked panes can't approach a single-video size, and the smaller view loses surgical detail.
- **Side-by-side half-clip split** (commit `0ae02d7`). Single canvas, clipped into left/right halves: raw on the left, filtered on the right, divider down the middle, per-half count badges.
  - **User reaction:** *"most of the detections are on the bottom right of the video so this doesn't really work."* The clip cuts off detection regions that cluster in one corner; you only see half the action on each side.
- **Raw primary + thin teal inset annotation** (commit `ac16e00`). Raw boxes drawn at full opacity + label (the "primary" signal); filtered boxes drawn as a thin 4 px teal inset stroke inside each kept bbox. Inline SVG visual guide in the filter-info card explained the symbols.
  - **User reaction:** *"it still feels unintuitive or hard to quickly understand and compare what a raw prediction was vs what the processed prediction is."* When raw == filtered (common case), the inset reads as "box inside a box" cognitive load. When they differ slightly, the rendering is busy.
- **Always-on raw underneath + filtered primary on top** (commit `7ef60e0`). Filtered boxes drawn solid + labeled at full opacity, raw boxes drawn dim (0.35 α) + dashed beneath. Replaced the broken Raw/Filtered toggle.
  - **User reaction:** *"raw vs filtered overlays look the same to me."* When the filter rejects nothing on a given frame the two layers collapse visually; dim-dashed isn't distinct enough.
- **Pre-existing Raw/Filtered view toggle** (state before `7ef60e0`). One-at-a-time mode toggle in the header.
  - **User reaction:** *"I don't think the Raw and filtered view toggles are working."* Wasn't actually broken — the user couldn't tell whether they'd switched modes because in most cases raw ≈ filtered.

**Outcome.** No visualization landed. All overlapping-layer designs failed because most raw detections survive the filter unchanged, so any layered design looks redundant; all spatial-split designs failed because either (a) detections cluster in one frame region so half is empty, or (b) splitting the frame shrinks each view too much.

**Decision.** Park the comparison feature. Revert visualization code to single-view (raw boxes only). Keep:
- The fetch-both-on-load data flow in `useCaseData.js` — raw + filtered fetched in parallel, exposed via `filteredOverlayResults` / `filteredClassStats` for whatever the next attempt looks like.
- The filter-info-card showing filter method + parameters (text-only).
- The collapsible side panels (useful regardless of compare mode).
- The "(filtered: N%)" secondary stat in the class panel (a number, not visual noise).

Drop the visualization code paths: `drawComparePane`, `.compare-pane` template, `.video-el--source-only` hidden positioning, teal inset stroke in `drawSingleOverlay`.

**Next time we revisit:** the principle is that the comparison should make a *difference* visible, not show two redundant overlays. Promising directions to try later:
- Diff-only highlight mode (only rejections appear, in red — most frames have nothing to draw).
- Hold-to-flip key (single view, press R to swap to the other view, photographer-style).
- Status icons on each raw box (✓ / ✗) instead of a second bbox layer.
- Frame-level diff strip in the raster timeline (already present as `changedFrames`) made more prominent so the user can jump straight to interesting frames.
