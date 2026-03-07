# Phase 4: Transitions Matrix Responsive Resize - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Make the transitions matrix always render as a square and reflow correctly as the right panel is resized by dragging the divider. A ResizeObserver on the matrix container div drives reactivity. No backend changes, no new npm dependencies.

</domain>

<decisions>
## Implementation Decisions

### Square enforcement strategy
- Matrix always fills available container width (not a fixed size that only shrinks)
- Square boundary enforced at the outer wrapper div: set both `width` and `height` on the matrix wrapper to the measured container width (capped at 320px max)
- Cell size is derived from this square size ÷ class count, so the grid always fits the square exactly
- Cell size floor: clamp at 14px minimum. If cells would go below 14px, the matrix section scrolls horizontally — do not shrink cells further
- Maximum size cap: 320px square — even if the panel is wider, the matrix does not exceed 320px

### ResizeObserver wiring
- Attach a `ref` to the transitions container div and observe `contentRect.width` in a ResizeObserver — this gives the actual usable width after padding, more accurate than `rightPanelWidth`
- Store measured width in a new reactive ref: `matrixContainerWidth`
- Set up in `onMounted`, disconnect in `onUnmounted` (satisfies success criterion: no console errors on unmount)
- `transitionMatrix` computed reads `matrixContainerWidth` to derive `cellSize` — all sizing logic stays in one reactive computed

### Claude's Discretion
- Initial value of `matrixContainerWidth` before ResizeObserver fires (e.g. `0` or `rightPanelWidth.value` as fallback)
- Whether to use `contentRect.width` or `borderBoxSize[0].inlineSize`
- Exact padding values to subtract (if any) before applying the 320px cap
- Whether the wrapper div uses `width: Xpx; height: Xpx` or `aspect-ratio: 1` CSS for the square constraint

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `rightPanelWidth` ref (line 588): already reactive to drag events via `startPanelDrag` — can serve as initial/fallback value for `matrixContainerWidth`
- `transitionMatrix` computed (lines 1021–1052): currently uses hardcoded `200` constant for cell sizing — this is the only change needed to the computed logic
- `onMounted` / `onUnmounted` hooks: already used in the component for video event listener setup — same pattern applies here

### Established Patterns
- All reactive state uses `ref()` / `computed()` — follow the same pattern for `matrixContainerWidth`
- Panel drag handler updates `rightPanelWidth.value` directly — ResizeObserver fires as a consequence of that, which then updates `matrixContainerWidth.value`
- No new npm packages allowed (REQUIREMENTS.md Out of Scope)

### Integration Points
- Matrix template (lines 470–510): the wrapper `div v-if="transitionMatrix"` gets a `ref` attribute and a `:style` binding for the square dimensions
- `transitionMatrix` computed (line 1050): replace `Math.floor(200 / classes.length)` with `Math.floor(squareSize / classes.length)` where `squareSize = Math.min(320, matrixContainerWidth.value)`
- ResizeObserver connects the DOM container ref → `matrixContainerWidth` ref → `transitionMatrix` computed → template re-render

</code_context>

<specifics>
## Specific Ideas

- The fix is additive: one new `ref` (`matrixContainerWidth`), one `templateRef` on the matrix container div, one ResizeObserver in `onMounted`/`onUnmounted`, and a one-line change in the `transitionMatrix` computed replacing the hardcoded `200` with the dynamic `squareSize`
- The outer wrapper div should have both `width` and `height` set to `squareSize + 'px'` to enforce the square — CSS `overflow: hidden` on this div prevents bleed if rounding causes a 1px error

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 04-transitions-matrix-responsive-resize*
*Context gathered: 2026-03-07*
