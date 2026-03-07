# Phase 4: Transitions Matrix Responsive Resize - Research

**Researched:** 2026-03-07
**Domain:** Vue 3 ResizeObserver integration, CSS square enforcement
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Matrix always fills available container width (not a fixed size that only shrinks)
- Square boundary enforced at the outer wrapper div: set both `width` and `height` on the matrix wrapper to the measured container width (capped at 320px max)
- Cell size is derived from this square size divided by class count, so the grid always fits the square exactly
- Cell size floor: clamp at 14px minimum. If cells would go below 14px, the matrix section scrolls horizontally — do not shrink cells further
- Maximum size cap: 320px square — even if the panel is wider, the matrix does not exceed 320px
- Attach a `ref` to the transitions container div and observe `contentRect.width` in a ResizeObserver
- Store measured width in a new reactive ref: `matrixContainerWidth`
- Set up in `onMounted`, disconnect in `onUnmounted`
- `transitionMatrix` computed reads `matrixContainerWidth` to derive `cellSize`
- No new npm dependencies

### Claude's Discretion
- Initial value of `matrixContainerWidth` before ResizeObserver fires (e.g. `0` or `rightPanelWidth.value` as fallback)
- Whether to use `contentRect.width` or `borderBoxSize[0].inlineSize`
- Exact padding values to subtract (if any) before applying the 320px cap
- Whether the wrapper div uses `width: Xpx; height: Xpx` or `aspect-ratio: 1` CSS for the square constraint

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRANS-01 | The transitions matrix panel uses a `ResizeObserver` on its container to detect panel width changes | ResizeObserver is a browser-native API; jsdom requires a mock (already present in the test file). Pattern: observe in `onMounted`, disconnect in `onUnmounted`. |
| TRANS-02 | The transitions matrix always renders at a 1:1 square aspect ratio, resizing automatically as the panel width changes | Inline `:style` binding on wrapper div sets `width` and `height` to the same `squareSize` value derived from `matrixContainerWidth`. `transitionMatrix` computed provides `cellSize = Math.floor(squareSize / classes.length)` (clamped at 14px floor). |
</phase_requirements>

## Summary

Phase 4 is a targeted, self-contained change to `YOLOVisualizer.vue`. The entire implementation is additive: one new `ref` (`matrixContainerWidth`), one template `ref` on the matrix container div, one `ResizeObserver` lifecycle in `onMounted`/`onUnmounted`, and a one-line change in the `transitionMatrix` computed replacing the hardcoded `200` with a dynamic `squareSize`. No new libraries, no backend changes, no CSS files touched.

The component already imports and uses `onMounted`/`onUnmounted` (lines 1909 and 1928). The existing test file already mocks `global.ResizeObserver` (line 72–76 of `YOLOVisualizer.spec.js`), so the test infrastructure requires no changes to mount correctly. New tests for TRANS-01 and TRANS-02 behavior are the only test additions needed.

The discretionary choices (initial value, `contentRect.width` vs `borderBoxSize`, CSS square strategy) are resolved below based on browser compatibility and the existing codebase patterns.

**Primary recommendation:** Use `contentRect.width` (broader support, simpler), initialize `matrixContainerWidth` to `0` so the matrix is hidden until the observer fires, and set `width` + `height` inline style on the wrapper div (not `aspect-ratio`) to match the existing inline-style pattern used throughout the template.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| ResizeObserver | Browser native | Observe element dimension changes | No polyfill needed for modern browsers; already mocked in test suite |
| Vue 3 `ref` / `computed` | Already in use | Reactive state + derived sizing | Matches all existing reactive patterns in the component |
| Vue 3 `onMounted` / `onUnmounted` | Already in use | Lifecycle setup/teardown | Already used at lines 1909/1928 — exact same pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `contentRect.width` | `borderBoxSize[0].inlineSize` | `borderBoxSize` is newer (Chrome 84+, Firefox 69+) but `contentRect` is universally supported. For this internal tool, either works; `contentRect` is simpler. |
| `width + height` inline style | `aspect-ratio: 1` CSS | `aspect-ratio` does not enforce a fixed pixel size — height grows with content. Explicit `width + height` in px guarantees the square at the measured size. |

## Architecture Patterns

### Recommended Project Structure

No new files. Changes are entirely within:
- `web/frontend/src/components/YOLOVisualizer.vue` (template + script)
- `web/frontend/src/__tests__/YOLOVisualizer.spec.js` (new test cases)

### Pattern 1: ResizeObserver with Vue 3 template ref

**What:** Observe a DOM element's width and write measurements into a reactive ref, so computed properties that depend on that ref re-evaluate automatically.

**When to use:** Any time a component needs to respond to its own or a child element's size changes that are driven by external forces (panel drag, window resize, flex reflow).

**Example:**
```javascript
// In <script setup>
const matrixContainerRef = ref(null)   // template ref
const matrixContainerWidth = ref(0)    // reactive measurement

let _matrixResizeObserver = null

onMounted(() => {
  // ... existing onMounted code stays ...
  if (matrixContainerRef.value) {
    _matrixResizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        matrixContainerWidth.value = entry.contentRect.width
      }
    })
    _matrixResizeObserver.observe(matrixContainerRef.value)
  }
})

onUnmounted(() => {
  // ... existing onUnmounted code stays ...
  if (_matrixResizeObserver) {
    _matrixResizeObserver.disconnect()
    _matrixResizeObserver = null
  }
})
```

**Template wiring:**
```html
<!-- line 470 area — add ref attribute -->
<div v-if="transitionMatrix" ref="matrixContainerRef"
  style="flex-shrink:0;border-top:1px solid #1a1a24;padding:10px 14px;max-height:40%;overflow-y:auto">
```

### Pattern 2: Square wrapper via inline style binding

**What:** A wrapper div inside the container holds only the matrix grid. Its `width` and `height` are set to the same computed `squareSize` value.

**When to use:** When the content must remain square regardless of container width, and the container width is measured dynamically.

**Example:**
```html
<!-- Inner wrapper — new div wrapping the flex row of labels + grid -->
<div :style="{ width: transitionMatrix.squareSize + 'px', height: transitionMatrix.squareSize + 'px', overflow: 'hidden' }">
  <!-- existing label + grid divs go here -->
</div>
```

The `squareSize` is returned from the `transitionMatrix` computed:
```javascript
const squareSize = Math.min(320, Math.max(0, matrixContainerWidth.value))
const cellSize = squareSize > 0
  ? Math.max(14, Math.floor(squareSize / classes.length))
  : 20  // fallback until observer fires
```

**Note on 14px floor and horizontal scroll:** The spec says cells clamp at 14px and the matrix section scrolls horizontally if needed. The matrix section wrapper already has `overflow-y:auto`. Add `overflow-x:auto` to that same div to enable horizontal scroll when `cellSize * classes.length > squareSize`.

### Pattern 3: Initial value before observer fires

**What:** `matrixContainerWidth` starts at `0`. The `transitionMatrix` computed already returns `null` when `filteredSummary` is absent, so the matrix is hidden until data loads. By the time `filteredSummary` is populated and the matrix renders, `onMounted` has already fired and the ResizeObserver will have called back at least once. If the callback hasn't fired yet, `squareSize` is `0` and `cellSize` falls back to `20` — a safe invisible state since the wrapper would be 0x0.

**Verdict:** Initialize to `0`. No flash of incorrect size occurs because `v-if="transitionMatrix"` gates rendering on data availability, and by the time data is available, `onMounted` and the first ResizeObserver callback have both completed.

### Anti-Patterns to Avoid

- **Observing `rightPanelWidth` with a `watch`:** The watch fires on the reactive ref change, not on the DOM reflow. ResizeObserver fires after the DOM reflow is complete, giving the actual rendered width including padding offsets. Use the observer, not the watch.
- **Subtracting padding manually:** Do not compute `contentRect.width - 28` (14px padding each side). `contentRect` already excludes padding from the reported width when `box-sizing: content-box` is in effect. Verify with the existing `* { box-sizing: border-box }` scoped style (line 1942) — with border-box, `contentRect.width` equals the width including padding minus the border. Test this in the browser and use what the observer actually reports.
- **Setting `height` only via `aspect-ratio`:** `aspect-ratio: 1` makes height proportional to width, but the element's height grows to fit its content if content overflows. Explicit `height: Xpx` + `overflow: hidden` is the reliable approach.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Element size tracking | `window.resize` listener + `getBoundingClientRect()` polling | `ResizeObserver` | `window.resize` misses panel drag events; `getBoundingClientRect` on resize only catches viewport changes, not container reflows |
| Square enforcement | CSS `padding-bottom: 100%` hack | Explicit `width + height` px binding | Percentage padding trick requires `position: relative` wrapper and `position: absolute` children — breaks the existing flex layout |

## Common Pitfalls

### Pitfall 1: ResizeObserver not attached because ref is null at onMounted
**What goes wrong:** `matrixContainerRef.value` is `null` at `onMounted` time because the `v-if="transitionMatrix"` condition is false — `filteredSummary` isn't loaded yet, so the div doesn't exist in the DOM.
**Why it happens:** `v-if` removes the element from the DOM; template refs for v-if'd elements are `null` when the condition is false.
**How to avoid:** Two options:
  1. Move the ref to a persistent ancestor div (one without `v-if`) and measure its width — the outer right-panel wrapper always exists.
  2. Use a `watch` on `transitionMatrix` (not null check) to attach the observer lazily after the div renders, then guard with `if (matrixContainerRef.value)`.
**Recommendation:** Option 1 — attach the ref to the always-present container div that wraps the full right panel content (or specifically to the transitions section's parent), not to the `v-if` div itself. This is simpler and more reliable.

**Confirmed:** The `v-if` is on the outer `<div>` at line 470. The ResizeObserver ref must go on a parent that is always in the DOM, or the component must watch for the element's existence.

### Pitfall 2: disconnect() called on null observer
**What goes wrong:** If `matrixContainerRef.value` was null at `onMounted` (v-if false), `_matrixResizeObserver` is never assigned. `onUnmounted` calling `_matrixResizeObserver.disconnect()` throws.
**How to avoid:** Guard with `if (_matrixResizeObserver)` before disconnect — shown in the code example above.

### Pitfall 3: `contentRect.width` vs `borderBoxSize` in jsdom
**What goes wrong:** jsdom's ResizeObserver mock returns whatever `contentRect.width` the mock is configured to return (0 by default). Tests that check sizing logic must configure the mock to call back with a specific width.
**How to avoid:** In tests, after mounting, manually invoke the ResizeObserver callback with a synthetic entry:
```javascript
const observerInstance = ResizeObserver.mock.results[0].value
observerInstance.observe.mock.calls  // verify it was called
// Trigger the callback manually:
const [callback] = ResizeObserver.mock.calls[0]
callback([{ contentRect: { width: 300 } }])
await nextTick()
```

### Pitfall 4: cellSize derived before squareSize is capped
**What goes wrong:** If `squareSize = matrixContainerWidth.value` without the 320px cap, a wide panel produces oversized cells.
**How to avoid:** Always apply `Math.min(320, matrixContainerWidth.value)` before dividing by `classes.length`.

## Code Examples

### Complete transitionMatrix computed (updated)
```javascript
// Source: derived from existing lines 1021-1052 + CONTEXT.md decisions
const transitionMatrix = computed(() => {
  const tm = filteredSummary.value?.transition_matrix;
  if (!tm || !data.value) return null;
  const classSet = new Set();
  for (const [from, targets] of Object.entries(tm)) {
    classSet.add(from);
    for (const to of Object.keys(targets)) classSet.add(to);
  }
  if (classSet.size === 0) return null;
  const allClasses = data.value.classes;
  const classes = allClasses.filter(c => classSet.has(c));
  let maxCount = 1;
  for (const targets of Object.values(tm)) {
    for (const count of Object.values(targets)) {
      if (count > maxCount) maxCount = count;
    }
  }
  const grid = classes.map(from =>
    classes.map(to => ({
      from, to,
      count: tm[from]?.[to] ?? 0,
      intensity: (tm[from]?.[to] ?? 0) / maxCount,
    }))
  );
  // Dynamic square size replaces hardcoded 200
  const squareSize = Math.min(320, Math.max(0, matrixContainerWidth.value));
  const cellSize = squareSize > 0
    ? Math.max(14, Math.floor(squareSize / classes.length))
    : Math.max(20, Math.min(32, Math.floor(200 / classes.length)));  // fallback
  return { classes, grid, cellSize, squareSize };
});
```

### ResizeObserver target: use a persistent ancestor
Because the `v-if="transitionMatrix"` div doesn't exist until data loads, attach the ref to its closest persistent ancestor. Inspect the template to find the right panel scroll container. The transitions section's parent is the right-panel `div` that wraps all right-panel sections (lines 400–512 area). Assign `ref="matrixContainerRef"` there. `contentRect.width` on that ancestor gives the available panel width, which is what the size cap operates on.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Vitest (embedded in vite.config.js) |
| Config file | `web/frontend/vite.config.js` (test block) |
| Quick run command | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js` |
| Full suite command | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRANS-01 | ResizeObserver is constructed and `observe()` is called on mount | unit | `npx vitest run src/__tests__/YOLOVisualizer.spec.js -t "TRANS-01"` | ❌ Wave 0 |
| TRANS-01 | `disconnect()` is called on unmount (no console errors) | unit | `npx vitest run src/__tests__/YOLOVisualizer.spec.js -t "TRANS-01"` | ❌ Wave 0 |
| TRANS-02 | `transitionMatrix.squareSize` equals `Math.min(320, measuredWidth)` | unit | `npx vitest run src/__tests__/YOLOVisualizer.spec.js -t "TRANS-02"` | ❌ Wave 0 |
| TRANS-02 | `transitionMatrix.cellSize` is always >= 14 | unit | `npx vitest run src/__tests__/YOLOVisualizer.spec.js -t "TRANS-02"` | ❌ Wave 0 |
| TRANS-02 | squareSize capped at 320 when panel is wider | unit | `npx vitest run src/__tests__/YOLOVisualizer.spec.js -t "TRANS-02"` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cd /home/jaeho/ent_cv/web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js`
- **Per wave merge:** `cd /home/jaeho/ent_cv/web/frontend && npx vitest run`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] New `describe("YOLOVisualizer — TRANS-01 and TRANS-02", ...)` block in `web/frontend/src/__tests__/YOLOVisualizer.spec.js` — covers TRANS-01 and TRANS-02

*(Framework and existing mocks are already in place — `global.ResizeObserver` is already mocked at line 72–76 of the spec file. No new test infrastructure needed.)*

## Open Questions

1. **Which persistent ancestor div should receive `ref="matrixContainerRef"`?**
   - What we know: The `v-if="transitionMatrix"` div (line 470) cannot hold the ref because it doesn't exist until data loads.
   - What's unclear: The exact line number and structure of the persistent right-panel scroll container. The template ends at line 516 — a read of lines 390–470 would confirm the right parent.
   - Recommendation: The planner should confirm by reading that block. The ref goes on the nearest persistent ancestor that represents the full available width for the transitions section.

2. **`contentRect.width` vs subtracting padding?**
   - What we know: The transitions container has `padding: 10px 14px` (line 471). With `box-sizing: border-box` (global scoped style at line 1942), `contentRect.width` on a border-box element reports width including padding.
   - What's unclear: Whether `contentRect.width` already reflects the inner content width or the full padded width.
   - Recommendation: In a border-box context, `contentRect` reports the width of the content area (excluding padding), so no manual subtraction is needed. The planner should verify by testing in the browser during implementation.

## Sources

### Primary (HIGH confidence)
- MDN ResizeObserver API — `contentRect.width` property, lifecycle (observe/disconnect), browser compatibility
- Vue 3 Composition API — `ref`, `computed`, `onMounted`, `onUnmounted` (verified against existing usage in the component)
- `/home/jaeho/ent_cv/web/frontend/src/components/YOLOVisualizer.vue` — direct read of lines 460–516, 519–600, 1015–1052, 1905–1939
- `/home/jaeho/ent_cv/web/frontend/src/__tests__/YOLOVisualizer.spec.js` — direct read confirming existing ResizeObserver mock and test patterns

### Secondary (MEDIUM confidence)
- ResizeObserver `borderBoxSize` vs `contentRect` — MDN documents both; `contentRect` universally supported, chosen for simplicity

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Vue 3 Composition API and ResizeObserver are both well-established; no third-party libraries involved
- Architecture: HIGH — implementation pattern is confirmed by direct reading of the existing component; lifecycle hooks already imported
- Pitfalls: HIGH — v-if ref null issue is a well-known Vue 3 gotcha, confirmed by the template structure read

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable APIs, no fast-moving dependencies)
