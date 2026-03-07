# Phase 3: Scrollbar and Case Picker Polish - Research

**Researched:** 2026-03-07
**Domain:** CSS scrollbar styling, Vue 3 layout / overflow
**Confidence:** HIGH

## Summary

Phase 3 is a pure CSS / layout task with no JavaScript logic changes. Two problems must be solved: (1) scrollable areas across the app use the browser-default scrollbar which is visually inconsistent with the dark theme; (2) CasePicker's root element uses `align-items: center` with `min-height: 100vh`, which clips the card list when content overflows the viewport because the flexbox centering prevents vertical scroll.

The scrollbar fix is applied globally in `index.html`'s inline `<style>` block (the only global CSS entry point) using two complementary APIs: `::-webkit-scrollbar` pseudo-elements for Chromium/Safari and `scrollbar-width` + `scrollbar-color` for Firefox. No new files, no npm packages, no Vue-level changes required.

The CasePicker overflow fix is a one-line CSS change: replace `align-items: center` with `align-items: flex-start` on `.upload-root`, or alternatively remove the flexbox centering and let the inner `.upload-center` center itself via `margin: auto`. The root element must also be allowed to scroll (`overflow-y: auto` or relying on body scroll once `body { overflow: hidden }` is scoped away from the picker route).

**Primary recommendation:** Add custom scrollbar CSS to `index.html` inline styles block; fix CasePicker scroll by making `.upload-root` scrollable and removing the `align-items: center` trap.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SCROLL-01 | Custom scrollbar styling (dark-themed) applied globally, covering Chromium (`::-webkit-scrollbar`) and Firefox (`scrollbar-width` + `scrollbar-color`) | Global CSS lives in `index.html` inline `<style>`; both APIs are native CSS, no library needed |
| SCROLL-02 | Classes section in YOLOVisualizer and all other scrollable areas display the custom scrollbar | `overflow-y: auto` already set on classes panel (line 397 of YOLOVisualizer) and jump-filter div (line 88); global rule covers them automatically |
| PICK-01 | Case picker page is scrollable when case list overflows the viewport | `.upload-root` has `align-items: center` which prevents overflow scroll; fix is layout-only in CasePicker.vue scoped `<style>` |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| CSS (native) | — | Scrollbar pseudo-elements and properties | Only option; no JS library needed |
| Vue 3 SFC `<style scoped>` | 3.x | Component-level layout fix for CasePicker | Already in use project-wide |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Inline `index.html` `<style>` | A global `src/style.css` imported in `main.js` | Either works. Project has no `src/style.css` yet; adding to `index.html` keeps surface minimal and matches existing pattern. |

**Installation:** None required.

---

## Architecture Patterns

### Global CSS Entry Point

The project has exactly one global CSS location: the inline `<style>` block in `index.html`. `main.js` has no CSS imports. Each Vue component uses `<style scoped>`. The correct place for scrollbar rules that must apply everywhere is `index.html`.

Current inline block (lines 7-11 of `index.html`):
```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, #app { height: 100%; }
body { overflow: hidden; }
```

Scrollbar rules appended here will cascade to all elements.

### Pattern 1: Dual-API Scrollbar Styling

**What:** Two CSS APIs must coexist — one for Chromium/Safari, one for Firefox.

**Chromium/WebKit:**
```css
/* Source: MDN https://developer.mozilla.org/en-US/docs/Web/CSS/::-webkit-scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a4a; }
```

**Firefox (standard spec, Baseline 2022):**
```css
/* Source: MDN https://developer.mozilla.org/en-US/docs/Web/CSS/scrollbar-width */
/* Source: MDN https://developer.mozilla.org/en-US/docs/Web/CSS/scrollbar-color */
* { scrollbar-width: thin; scrollbar-color: #2a2a3a transparent; }
```

Firefox ignores `::-webkit-scrollbar`. Chromium ignores `scrollbar-width`/`scrollbar-color` when `::-webkit-scrollbar` is present (as of Chrome 121, it restores support for the standard properties alongside pseudo-elements). Both sets of rules should coexist.

**When to use:** Always apply both. Order does not matter; they target different browser engines.

### Pattern 2: CasePicker Overflow Fix

**Problem root cause:** `.upload-root` is:
```css
.upload-root {
  min-height: 100vh;
  display: flex;
  align-items: center;  /* PROBLEM: prevents overflow scroll */
  justify-content: center;
}
```

When flex container uses `align-items: center` and the content height exceeds the viewport, the overflow is split above and below — the top half is clipped behind the viewport top and unreachable because `body { overflow: hidden }` prevents page scroll.

**Fix — two complementary changes:**

1. On `.upload-root`: change to `align-items: flex-start` so content starts at the top. Add `overflow-y: auto` so the root element itself scrolls. Retain `justify-content: center` for horizontal centering.
2. On `.upload-center`: add `margin: 40px auto` to preserve vertical breathing room at top.

This keeps the existing visual design intact for short case lists (centered appearance) while allowing scroll when content grows.

**Alternative approach:** Remove flexbox centering from `.upload-root` entirely and use `margin: 0 auto` on `.upload-center`. Same result.

**Note on `body { overflow: hidden }`:** This global rule in `index.html` is intentional — YOLOVisualizer is a full-viewport fixed layout that should never body-scroll. The CasePicker fix must therefore make the route's own container scrollable rather than relying on body scroll.

### Anti-Patterns to Avoid

- **Scoping scrollbar rules in a component `<style scoped>`:** Vue scoped styles add a data attribute selector. `::-webkit-scrollbar` pseudo-elements don't receive the scoped attribute and the rule is dropped silently. Always put scrollbar rules in global (unscoped) CSS.
- **Using `::-webkit-scrollbar` without the Firefox properties:** Results in SCROLL-01 failure on Firefox — the requirement explicitly requires both.
- **Fixing CasePicker by removing `body { overflow: hidden }`:** That rule is load-bearing for YOLOVisualizer. Fix at the component level only.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Custom scrollbar | JS-based scroll replacement (SimpleBar, OverlayScrollbars) | Native CSS only | REQUIREMENTS.md explicitly forbids new npm dependencies |
| Overflow detection | JS ResizeObserver to toggle scroll class | CSS `overflow-y: auto` | Pure CSS is sufficient; ResizeObserver is Phase 4's concern for the matrix |

---

## Common Pitfalls

### Pitfall 1: Scoped `<style>` Drops Scrollbar Rules

**What goes wrong:** Developer adds `::-webkit-scrollbar` inside `<style scoped>` — rules are silently ignored because Vue's scoping mechanism injects a `[data-v-xxx]` attribute that pseudo-elements can't carry.

**How to avoid:** Only place scrollbar rules in global CSS (`index.html` inline style or an unscoped `<style>` in `App.vue`).

**Warning signs:** Scrollbar unchanged in browser; DevTools shows the rule with a strikethrough or no selector match.

### Pitfall 2: CasePicker `align-items: center` Clips Content

**What goes wrong:** `align-items: center` on a flex container with `min-height: 100vh` causes overflow content to extend equally above and below center — the above portion is unreachable because `body { overflow: hidden }` suppresses body scroll.

**How to avoid:** Use `align-items: flex-start` + `overflow-y: auto` on the route root container. Do NOT touch body overflow.

### Pitfall 3: Firefox `scrollbar-color` Two-Value Syntax

**What goes wrong:** `scrollbar-color: #thumb` (single value) is invalid — Firefox requires exactly two values: thumb then track.

**Correct syntax:** `scrollbar-color: #2a2a3a transparent;`

---

## Code Examples

### Global Scrollbar Styles (append to `index.html` inline `<style>`)
```css
/* Source: MDN ::-webkit-scrollbar, scrollbar-width, scrollbar-color */
/* Chromium / WebKit */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3c3c4e; }

/* Firefox (standard spec) */
* { scrollbar-width: thin; scrollbar-color: #2a2a3a transparent; }
```

### CasePicker `.upload-root` Fix (in `CasePicker.vue` `<style scoped>`)
```css
/* Before */
.upload-root {
  min-height: 100vh; background: #06060a;
  display: flex; align-items: center; justify-content: center;
  ...
}

/* After */
.upload-root {
  min-height: 100vh; background: #06060a;
  display: flex; align-items: flex-start; justify-content: center;
  overflow-y: auto;
  ...
}
.upload-center { ... margin: 40px auto; }  /* replaces the flex centering vertical effect */
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `::-webkit-scrollbar` only (no Firefox) | `scrollbar-width` + `scrollbar-color` (Baseline 2022) alongside webkit pseudo-elements | Firefox 64+ (2018), standardized 2022 | Firefox now has first-class support; both APIs must coexist |
| `overflow: scroll` always shows scrollbar | `overflow: auto` only shows when content overflows | Always true | Use `auto` for aesthetic; scrollbar appears only when needed |

---

## Open Questions

1. **Chrome 121+ standard scrollbar properties**
   - What we know: Chrome 121 (Jan 2024) added support for `scrollbar-color` and `scrollbar-width` as standard properties alongside `::-webkit-scrollbar`.
   - What's unclear: Whether Chrome uses the webkit pseudo-elements OR the standard properties when both are defined. MDN notes that when `::-webkit-scrollbar` rules are present, standard properties are ignored in Chromium (as of documentation checked).
   - Recommendation: Keep both sets of rules. The webkit block controls Chromium; the standard block controls Firefox. This is the safe, established pattern.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Vitest (inline in vite.config.js) |
| Config file | vite.config.js (`test.environment: jsdom`) |
| Quick run command | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` |
| Full suite command | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SCROLL-01 | Global CSS contains `::-webkit-scrollbar` + `scrollbar-width` rules | manual-only | — | N/A |
| SCROLL-02 | Scrollable elements in YOLOVisualizer show styled scrollbar | manual-only | — | N/A |
| PICK-01 | CasePicker root is scrollable when content overflows | unit (DOM structure) | `npx vitest run src/__tests__/CasePicker.spec.js` | ❌ Wave 0 |

**Manual-only justification for SCROLL-01/02:** CSS pseudo-element rendering cannot be tested in jsdom (no layout engine). Visual verification in a real browser is the only reliable method.

### Sampling Rate
- **Per task commit:** `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose`
- **Per wave merge:** same
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `web/frontend/src/__tests__/CasePicker.spec.js` — covers PICK-01 (asserts `.upload-root` has `overflow-y: auto` in computed style OR checks component renders a scrollable container)

*(SCROLL-01 and SCROLL-02 are manual-only; no Wave 0 test file required for those.)*

---

## Sources

### Primary (HIGH confidence)
- MDN Web Docs — `::-webkit-scrollbar` https://developer.mozilla.org/en-US/docs/Web/CSS/::-webkit-scrollbar
- MDN Web Docs — `scrollbar-width` https://developer.mozilla.org/en-US/docs/Web/CSS/scrollbar-width
- MDN Web Docs — `scrollbar-color` https://developer.mozilla.org/en-US/docs/Web/CSS/scrollbar-color
- Direct code inspection of `index.html`, `CasePicker.vue`, `YOLOVisualizer.vue`, `main.js`

### Secondary (MEDIUM confidence)
- STATE.md decision log: "CSS approach: Native ::-webkit-scrollbar + scrollbar-width/scrollbar-color, no library" — confirms locked approach
- REQUIREMENTS.md Out of Scope: "New npm dependencies — Pure CSS + Vue reactivity sufficient"

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — locked by REQUIREMENTS.md and STATE.md decisions; confirmed against MDN
- Architecture: HIGH — confirmed by direct code inspection; global CSS location unambiguous
- Pitfalls: HIGH — webkit scoped-style issue is well-documented; CasePicker layout bug verified by reading the actual CSS

**Research date:** 2026-03-07
**Valid until:** Stable indefinitely (CSS scrollbar APIs, Vue 3 scoped styles behavior — no churn expected)
