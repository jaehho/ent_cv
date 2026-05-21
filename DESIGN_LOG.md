# Design Log

Running, append-only log of design decisions and non-trivial changes. Newest on top.

## 2026-05-21 — Raw / Filtered view-mode toggle

**Context.** After wiring in the in-use signal, the box labels never showed "IN USE" — even on cases known to have `in_use=true` in `filtered_detections.json`. Root cause: the overlay was painting raw detections only, with filtered loaded as a side annotation (change markers, timeline shading). There was no UI to actually look at the filtered side directly. This was a regression from yesterday's revert of the side-by-side comparison: that revert killed the *dual* view but accidentally also killed *any* way to view filtered alone.

**Decision.** Add a Raw / Filtered toggle in the player-controls "Playback" section (between Rate and Jump). Defaults to **Filtered** so the post-processed view (which carries `in_use` and all downstream signals) is what users see first. Both payloads stay loaded; flipping is instant — no re-fetch.

- **Composable**: `useCaseData` now accepts a `viewMode` ref. `frameMap` (the central per-frame lookup that drives the overlay, stats, raster, and jump filters) switches its source between raw and filtered based on this ref. Falls back to raw if no filtered file exists, so cases without a postprocess run still work.
- **UI**: `PlayerControls.vue` gained a `view-toggle` block with two buttons. Filtered is disabled (with a tooltip) when no `filtered_detections.json` was loaded for the case.
- **No projection hack**: rejected the alternative of projecting `in_use` flags from the filtered side onto raw boxes (class-level or IoU-matched). The toggle is the right primitive — once filtered is viewable directly, in_use naturally appears on the actual boxes it belongs to.

**Carryover from earlier today.** The dashed/dimmed in-use treatments were swapped for an "IN USE" label suffix on the box (per the user's preference for a more legible signal). Timeline still dims idle frames at 35% alpha.

**Files.** `web/frontend/src/composables/useCaseData.js` (viewMode param, frameMap swap, header comment correction); `web/frontend/src/components/PlayerControls.vue` (view-block UI + props/emit); `web/frontend/src/components/YOLOVisualizer.vue` (`detectionViewMode` ref, prop wiring).

## 2026-05-21 — Web viewer: gesture-based timeline zoom, scrub-tracking playhead, matrix label fix, case-load gzip + instrumentation

**Context.** Four issues addressed in one pass:

1. *Zoom existed but was invisible* — only scroll-on-raster and `+/-/0` keys, both hidden in the kbd hints bar. Users didn't realize the timeline could zoom.
2. *Auto-pan only fired during playback* — manual scrub / arrow / jump could leave the playhead off-screen with no way to recover except a minimap click.
3. *Transitions matrix y-axis labels misaligned* — `minHeight: cellSize` allowed wrapped labels (e.g. "Energy-based hemostasis without suction") to grow taller than the corresponding grid row, drifting visually off-row.
4. *Case switching reported as slow* — wanted to see where the time actually goes before deciding on a structural fix.

**Decisions.**

- **Zoom UX — gestures over chrome.** First attempt was a small in/out/reset button toolbar in the raster's top-right corner. User rejected it as not intuitive enough. Replaced with a video-editor-style gesture set:
  - Raster `Shift+drag` selects a region; release zooms into it. Translucent teal rectangle gives live feedback during the drag.
  - Raster `dblclick` resets zoom + pan.
  - Minimap viewport indicator becomes interactive: drag inside it to pan, drag either edge (±6px hit zone, `ew-resize` cursor) to rescale (resize-left also re-anchors panOffset; resize-right keeps the left edge fixed). Pressing outside the viewport recenters there and continues into a drag.
  - Mouse-wheel-at-cursor zoom on the raster is retained; `0` key resets; `+/-` keys removed in favor of the gestures.
  - Kbd-bar surfaces the new gestures (`Shift+drag`, `Dbl-click`, `Scroll`) instead of the old key list.
- **Smoothness pass (post-feedback).** First gesture build was choppy — only `Shift+drag` (which doesn't write any state during the drag) felt fast. Every other gesture wrote `panOffset`/`zoomLevel` per mousemove, which fired the minimap + raster redraw watchers and pegged the main thread on sparse-iteration repaints. Two changes made it smooth:
  - *Minimap viewport + playhead lifted to CSS divs.* Same trick the raster playhead already used. The minimap canvas now only paints static content (bars + changed-frame strip) and was dropped from the `panOffset`/`zoomLevel`/`currentFrame` watcher entirely. Pan/zoom/scrub now move two `position:absolute` divs via `left`/`width`/`translateX` — compositor-only, zero canvas redraw.
  - *drawRaster subsamples by pixel at low zoom.* The bars loop used to iterate every visible frame even when 30+ frames collapsed into one pixel (e.g. zoom=1 over 100k frames). New `step = max(1, ceil(0.5 / pxPerFrame))` keeps ~2 samples per pixel; at high zoom `step==1` and behavior is identical to before. Loop iterations dropped from O(visibleFrames) to O(pixelWidth) at the zoom levels where it actually mattered.
- **Tracking mode.** "Pan only when off-screen": when `currentFrame` leaves `[panOffset, panOffset + 1/zoom]`, recenter the viewport on the playhead. Active during playback, scrubbing, arrow-step, jump, and zoom changes — single rule. Mid-view scrubbing doesn't move the viewport (avoids jitter). During playback, a 2% leading margin scrolls a beat before the playhead actually exits to avoid edge flicker.
- **Matrix labels.** Switch from `minHeight` (grows with content) to fixed `height` + `whiteSpace: nowrap` + inner span with `text-overflow: ellipsis`. Full label still in `title` for hover. Trade-off: long names truncate visually; widen the right panel or hover for the full name. The previous behavior wasn't actually showing more text — it was showing wrapped text that no longer pointed at the right row.
- **Case-switch network/parse parallelization + instrumentation.** Started all three API calls (raw + filtered detections, filtered summary) concurrently before any await, and `Promise.all`'d the `.json()` parses too. Added a `console.info` per case-load: `headers Xms · parse Yms · total Zms`.
- **Case-switch gzip.** Added a JSON-only variant of `GZipMiddleware` (`api.middleware.JsonGZipMiddleware`) to the head of `MIDDLEWARE`. Detection JSON for a 1h surgery is 50–100MB — gzip compresses repetitive bbox/class/confidence JSON ~5–10×, cutting the wire portion of case-switch dramatically. Client-side parse cost unchanged.
  - *Why not vanilla `django.middleware.gzip.GZipMiddleware`*: it also compresses `StreamingHttpResponse`, which strips `Content-Length` on range requests and re-encodes already-compressed mp4 bytes. The browser saw the video stream as broken and dropped into a buffering loop instead of progressive playback. Gating compression on `Content-Type: application/json` lets us keep the JSON win without touching the video / image endpoints.

**Findings — case-switch slowness root cause.**
Detection files for real cases are 50–100MB JSON (e.g. `20251204_01/detections.json` is 100MB on disk). The browser's `Response.json()` parses on the main thread serially, so multiple ~100MB parses still block the UI for seconds even after gzip. Server-side, `views.detections()` ALSO walks every detection to remap class IDs and regroup by frame — proportional to payload size.

**Open / promising next steps for case-switch slowness** (gzip is in; bigger work still pending):
- *Columnar response format* — return a flat columnar payload (`frame[]`, `cls[]`, `conf[]`, `bbox[]` as typed arrays) instead of an array of objects-per-detection. Cuts both wire size and parse cost; gzip's redundancy advantage shrinks once we're not repeating JSON keys per detection. Bigger change.
- *Web Worker parse* — move `Response.json()` for the detection payloads into a worker so the main thread stays interactive during parse.
- *Skip re-fetch on remount* — currently the visualizer remounts whenever the user navigates back to picker and into another case. With a top-level case cache (keyed by case id) we'd avoid re-fetching/re-parsing for cases the user revisits in a session.

## 2026-05-21 — "Instrument in use" derived from patient-box distance

**Context.** Needed a frame-level "in use" signal for each instrument. Quantified as the minimum Euclidean distance between the instrument bounding box and a patient bounding box; below threshold (or overlapping) → in use.

**Decisions.**
- **Threshold unit.** Fraction of *frame* diagonal (default 0.02), not patient-box diagonal. Resolution-agnostic; loses zoom-invariance but the project's cameras are reasonably consistent and the math is simpler to reason about.
- **Patient source.** Raw (confidence-filtered only) detections, not temporally filtered. Patient detection is steady; using raw avoids edge cases where the temporal filter trims short patient runs and blinds the reference.
- **Carry-forward bound.** Last-seen patient box reused for up to `--patient-carry-sec` (default 3.0s). Beyond that, no reference → instrument cannot be in-use. Prevents stale anatomy boxes from yielding false positives across scene changes.
- **Frame dims source.** New: `predict.py` records `width`/`height` in `metadata.json` from `result.orig_shape`. Backfill via `scripts/backfill_metadata_dims.py` (ffprobe → in-place metadata write, idempotent). Runtime falls back to ffprobing the source video if metadata still lacks dims.

**Outputs.** Per-row `in_use` flag in `filtered_detections.json`; per-segment `in_use_fraction` / `in_use_spans` and a flat `in_use_segments` timeline in `filtered_segments.json`; per-class `in_use_time_sec` / `in_use_fraction` in `filtered_summary.json`. `_filter` block records all in-use params (threshold, carry, dims) so downstream consumers know how the flags were derived.

**Tests.** `tests/test_postprocess_inuse.py` — 16 unit tests covering `min_box_distance` (overlap, touching, separated, diagonal, symmetric) and `compute_in_use` (in-frame in/out, far/near, no-patient, carry-forward in/out of window, raw patient source override, non-instrument flag passthrough). `tests/` added to `[tool.pytest.ini_options].testpaths`.

## 2026-05-21 — Phase 5 IA cleanup: shipped and reverted same day

**Context.** Phase 5 of the web refactor plan called for two changes: (1) pull the time/frame readout out of the left panel into a new strip under the video; (2) collapse the keyboard-hints bar behind a `? Keyboard shortcuts` toggle.

**Why reverted.** Both were stylistic preferences I rationalized as IA wins, not actual fixes for the original *"hard to find controls"* complaint (which was already addressed in Phase 1 by removing the postprocess UI).

- **Time strip move.** The new strip was *smaller* (22 px vs the previous 28 px clock) and less prominent. The "~280 px eye travel" argument in the original commit was made up — the old time-display sat at the bottom of the left panel, roughly aligned with the bottom of the video, so the horizontal distance while glancing was minor.
- **Keyboard hints collapse.** Reclaimed only ~24 px of vertical chrome. For a single-user model-debugging tool, ambient shortcut visibility helps more than the tiny space gain hurts.

**Decision.** Phase 5 ships as a no-op. Code is unchanged from the end of Phase 4; this log entry exists so the lesson survives the squash.

**Lesson.** When the original phase plan was written, "single time/frame readout" was based on a vague intuition rather than a concrete pain point. Should have pushed back at planning time. Don't ship UI moves "because the plan said so" — needs a real complaint behind it.

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
