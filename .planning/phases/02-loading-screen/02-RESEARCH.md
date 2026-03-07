# Phase 2: Loading Screen - Research

**Researched:** 2026-03-07
**Domain:** Vue 3 reactive state gating, HTMLVideoElement event lifecycle
**Confidence:** HIGH

## Summary

Phase 2 adds a loading screen that is visible from the moment a case is selected until both the API
fetch is complete and the video `loadedmetadata` event has fired. The viewer content is hidden
behind the loading screen while either condition is unmet. Once both are satisfied the loading
screen is removed and the viewer renders.

The current codebase has no loading state at all. The template root is `<div v-if="data">`, meaning
the viewer either renders or renders nothing — there is no in-between state. When a case is selected
`loadCase()` fires the fetch, sets `data.value`, and calls `nextTick(() => seekToFrame(0))`. The
video element appears in the DOM after `data` is set; the `currentPartVideoUrl` watcher fires and
attaches a `loadedmetadata` listener that calls `seekAndPlay`. The loading gate must span the period
from case selection (before fetch) through to `loadedmetadata` having fired.

The approach is: add two boolean refs (`dataReady` and `videoReady`), derive `isLoading` as a
computed, reset both refs to `false` at the top of `loadCase()`, set `dataReady = true` after the
API response is stored, and set `videoReady = true` inside the `loadedmetadata` callback in the
`currentPartVideoUrl` watcher. The template adds a `v-if="isLoading"` loading overlay and wraps
the existing `<div v-if="data">` as `v-else`. No new dependencies required.

**Primary recommendation:** Two boolean refs + one computed (`isLoading`) + a minimal spinner
overlay driven by `v-if`/`v-else` in the template. Wire `videoReady` inside the existing
`currentPartVideoUrl` watcher's `loadedmetadata` callback. Total change is under 25 lines.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LOAD-01 | A loading screen is displayed from the moment a case is selected until both the API response is received AND the video `loadedmetadata` event has fired | `dataReady` + `videoReady` refs reset in `loadCase()` before fetch; `isLoading = !dataReady || !videoReady` computed drives template gate |
| LOAD-02 | The loading screen is hidden and the viewer is shown only when the video is ready to play from frame 0 | `videoReady` set to `true` inside the `loadedmetadata` callback in `currentPartVideoUrl` watcher, which Phase 1 confirmed fires after video src is set and metadata is parsed |
</phase_requirements>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vue 3 `ref` / `computed` | already in use | Reactive boolean state for loading gate | Already the project's state primitive — no new deps |
| Vue `v-if` / `v-else` | already in use | Conditionally render loading overlay vs viewer | Standard Vue template conditional |
| CSS `@keyframes` | native | Spinner animation | Zero-dependency; 5 lines; matches project's "no new npm deps" constraint |

### No Alternatives to Consider

The project REQUIREMENTS.md explicitly marks "New npm dependencies" as out of scope. Every solution
is native CSS + Vue 3 primitives.

**Installation:** none required.

## Architecture Patterns

### Current Template Structure (relevant excerpt)

```
<template>
  <div v-if="data" class="app-root">   <!-- whole viewer -->
    ...
  </div>
  <!-- nothing rendered while data is null -->
</template>
```

### Recommended Loading-Gate Structure

```
<template>
  <!-- Loading overlay — shown from case selection until video ready -->
  <div v-if="isLoading" class="loading-screen">
    <div class="loading-spinner"></div>
    <p class="loading-label">Loading...</p>
  </div>

  <!-- Viewer — shown only after both fetch and loadedmetadata complete -->
  <div v-else-if="data" class="app-root">
    ...
  </div>
</template>
```

### Pattern: Two-Flag Loading Gate

**What:** Two boolean refs track the two async conditions independently. A computed derives the
combined gate.

**When to use:** When a ready state depends on two independent async events (API response + DOM
event) that fire in an unpredictable order.

**Example:**

```javascript
// Source: Vue 3 docs — ref, computed
const dataReady  = ref(false);
const videoReady = ref(false);
const isLoading  = computed(() => !dataReady.value || !videoReady.value);
```

Reset at the start of `loadCase()`:
```javascript
async function loadCase(caseName) {
  dataReady.value  = false;
  videoReady.value = false;
  try {
    const res    = await fetch(`/api/cases/${caseName}/detections/`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const parsed = await res.json();
    data.value = Object.freeze(parsed);
    // ... rest of loadCase reset block (unchanged) ...
    dataReady.value = true;          // condition 1 satisfied
    nextTick(() => seekToFrame(0));
  } catch (err) { ... }
}
```

Set `videoReady` inside the existing `currentPartVideoUrl` watcher:
```javascript
watch(currentPartVideoUrl, (newUrl) => {
  if (!newUrl || newUrl === videoSrc.value) return;
  // ... existing pause/videoSrc assignment ...
  nextTick(() => {
    if (!videoRef.value) return;
    const seekTs = currentPartTimestamp.value;
    const seekAndPlay = () => {
      videoRef.value.currentTime = seekTs;
      videoReady.value = true;       // condition 2 satisfied
      if (wasPlaying) videoRef.value.play();
    };
    videoRef.value.src = newUrl;
    videoRef.value.addEventListener('loadedmetadata', seekAndPlay, { once: true });
  });
});
```

### Pattern: Minimal Spinner CSS

**What:** A rotating border spinner — the simplest possible loading indicator for an internal tool.

**Example:**

```css
/* Source: MDN CSS animations */
.loading-screen {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100vh;
  background: #0f0f17;          /* matches existing app background */
  gap: 16px;
}

.loading-spinner {
  width: 40px;
  height: 40px;
  border: 3px solid #2a2a35;
  border-top-color: #4ecdc4;    /* matches existing accent color */
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.loading-label {
  color: #555;
  font-size: 13px;
  letter-spacing: 1px;
}
```

### Anti-Patterns to Avoid

- **Gating only on `data`:** The existing `v-if="data"` gate hides the viewer until the API
  response arrives but shows it immediately after — before `loadedmetadata` fires and before the
  video is seekable. This violates LOAD-02 because the viewer appears in a non-ready state.
- **Gating on `videoSrc`:** `videoSrc` is set synchronously inside `nextTick` in the `currentPartVideoUrl`
  watcher, but the video is not ready to play until `loadedmetadata` fires. Using `videoSrc` as the
  gate does not satisfy LOAD-02.
- **Setting `videoReady = true` after setting `videoSrc`, not after `loadedmetadata`:** The video
  element begins loading when `src` is assigned but metadata is not available until `loadedmetadata`
  fires. Setting `videoReady` too early would hide the loading screen before the video can seek.
- **Not resetting flags on case switch:** If `dataReady` and `videoReady` are not reset at the top
  of `loadCase()`, switching cases shows the viewer briefly with stale data before the new case
  loads. Reset both as the first statements before the `await fetch(...)` call.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loading animation | Custom JS-driven animation | CSS `@keyframes` | Browser handles compositing off main thread; GPU-accelerated; 5 lines |
| Async gate logic | Custom Promise.all pattern | Two refs + computed | Vue reactivity handles re-evaluation automatically; no subscriptions to manage |

**Key insight:** The existing Phase 1 `loadedmetadata` callback in `currentPartVideoUrl` is the
exact right hook for `videoReady`. No new event listener infrastructure is needed — just add
`videoReady.value = true` inside `seekAndPlay`, which already fires at the correct moment.

## Common Pitfalls

### Pitfall 1: `videoReady` Never Set for Prediction Mode

**What goes wrong:** `currentPartVideoUrl` returns `null` in prediction mode (`videoMode === 'prediction'`).
The watcher guard `if (!newUrl ...) return` short-circuits, so `loadedmetadata` never fires and
`videoReady` is never set — the loading screen stays forever.

**Why it happens:** The prediction mode path bypasses the video element entirely. There is no
`loadedmetadata` event to hook.

**How to avoid:** After setting `dataReady = true` in `loadCase()`, also check if `videoMode` is
`'prediction'` and set `videoReady = true` immediately in that branch. Alternatively, watch
`videoMode` and `currentPartVideoUrl` together. The simplest approach: if `currentPartVideoUrl`
computes to `null` (prediction mode), set `videoReady = true` along with `dataReady = true`.

**Warning signs:** Loading screen stays permanently when switching to prediction video mode.

### Pitfall 2: Loading Spinner Flashes on Fast API Responses

**What goes wrong:** On localhost the API responds in <50ms. The loading screen appears and
disappears so quickly it feels like a flash/glitch rather than a helpful indicator.

**Why it happens:** Both flags flip within a single microtask queue cycle when data is cached.

**How to avoid:** This is acceptable behavior — do not add artificial minimum-delay timers. The
spec says "from case selection until ready," not "for at least N milliseconds." For an internal
tool, a fast-disappearing spinner is correct behavior. No workaround needed.

### Pitfall 3: `isLoading` Computed Initialized to `false`

**What goes wrong:** If `dataReady` and `videoReady` initialize to `true` (or `isLoading`
initializes to `false`), there is no loading screen on the very first case load — the viewer
element attempts to render before `data` is set, which triggers the `v-else-if="data"` false branch
and shows nothing, but the loading screen is also absent, producing a blank screen.

**How to avoid:** Initialize `dataReady = ref(false)` and `videoReady = ref(false)` so
`isLoading` starts as `true`. The initial blank-screen state becomes the loading screen state.

### Pitfall 4: `v-if` / `v-else-if` on Sibling Elements at Root Level

**What goes wrong:** Vue 3 `<template>` can have multiple root nodes. `v-if` + `v-else` must be
on adjacent sibling elements — any intervening comment or element breaks the `v-else` linkage and
causes a compile warning.

**How to avoid:** Place the loading `<div>` and the viewer `<div>` as direct adjacent siblings
with no elements between them. Use `v-else-if="data"` (not bare `v-else`) to preserve the null
guard that existed before.

## Code Examples

### Full Loading Gate — Script Section

```javascript
// Source: Vue 3 docs — reactivity fundamentals
// Add alongside other refs at the top of <script setup>
const dataReady  = ref(false);
const videoReady = ref(false);
const isLoading  = computed(() => !dataReady.value || !videoReady.value);
```

### Reset in loadCase() — Before fetch

```javascript
// Source: direct codebase reading — loadCase() at line 1060
async function loadCase(caseName) {
  dataReady.value  = false;   // ADD: reset before fetch starts
  videoReady.value = false;   // ADD: reset before fetch starts
  try {
    const res = await fetch(`/api/cases/${caseName}/detections/`);
    // ...
    data.value = Object.freeze(parsed);
    // ... existing reset block unchanged ...
    dataReady.value = true;   // ADD: condition 1 met
    nextTick(() => seekToFrame(0));
  } catch (err) { ... }
}
```

### videoReady in currentPartVideoUrl Watcher

```javascript
// Source: direct codebase reading — currentPartVideoUrl watcher at line 1670
watch(currentPartVideoUrl, (newUrl) => {
  if (!newUrl || newUrl === videoSrc.value) return;
  const wasPlaying = isPlaying.value || _partEndedContinue;
  _partEndedContinue = false;
  if (videoRef.value) videoRef.value.pause();
  videoSrc.value = newUrl;
  nextTick(() => {
    if (!videoRef.value) return;
    const seekTs = currentPartTimestamp.value;
    const seekAndPlay = () => {
      videoRef.value.currentTime = seekTs;
      videoReady.value = true;   // ADD: condition 2 met after loadedmetadata
      if (wasPlaying) videoRef.value.play();
    };
    videoRef.value.src = newUrl;
    videoRef.value.addEventListener('loadedmetadata', seekAndPlay, { once: true });
  });
});
```

### Prediction Mode Short-Circuit

```javascript
// In loadCase(), after dataReady.value = true:
if (videoMode.value === 'prediction') {
  videoReady.value = true;   // no video element in prediction mode
}
```

### Template Gate

```html
<!-- Loading overlay -->
<div v-if="isLoading" class="loading-screen">
  <div class="loading-spinner"></div>
  <p class="loading-label">Loading...</p>
</div>

<!-- Viewer (unchanged internals) -->
<div v-else-if="data" class="app-root">
  <!-- existing content unchanged -->
</div>
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `v-if="data"` (API-only gate) | `v-if="isLoading"` (API + video gate) | Phase 2 | Prevents blank/broken viewer during video init |
| No loading indicator | Spinner overlay | Phase 2 | User feedback during case switch |

## Open Questions

1. **Prediction mode — what is the ready signal?**
   - What we know: `currentPartVideoUrl` returns `null` in prediction mode, so the `loadedmetadata`
     callback never fires. The prediction viewer shows frame images via `<img>`, not `<video>`.
   - What's unclear: Is there any async initialization in prediction mode after `data` is set, or
     is the viewer immediately ready once `dataReady = true`?
   - Recommendation: Treat prediction mode as ready immediately after `dataReady = true` — set
     `videoReady = true` in the same branch. Image `<img>` loading is per-frame-on-demand, not a
     blocking initialization step.

2. **Initial page load — first case selected from CasePicker**
   - What we know: `loadCase(props.id)` is called in `onMounted`. With `dataReady` and `videoReady`
     both initializing to `false`, `isLoading` starts `true`, so the loading screen shows
     immediately on mount. This is correct.
   - What's unclear: Nothing — this is the desired behavior per LOAD-01 ("from the moment a case is
     selected").
   - Recommendation: No additional handling needed; initialization covers this case automatically.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | vitest 3.x + @vue/test-utils |
| Config file | `web/frontend/vite.config.js` (vitest config co-located) |
| Quick run command | `cd web/frontend && npx vitest run --reporter=verbose src/__tests__/YOLOVisualizer.spec.js` |
| Full suite command | `cd web/frontend && npx vitest run --reporter=verbose` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| LOAD-01 | `isLoading` is `true` immediately when `loadCase` is called (before fetch resolves) | unit | `cd web/frontend && npx vitest run --reporter=verbose src/__tests__/YOLOVisualizer.spec.js -t "loading"` | No — Wave 0 |
| LOAD-01 | `isLoading` remains `true` after fetch resolves but before `loadedmetadata` fires | unit | same | No — Wave 0 |
| LOAD-02 | `isLoading` becomes `false` only after `videoReady` is set (simulating `loadedmetadata`) | unit | same | No — Wave 0 |
| LOAD-02 | In prediction mode, `isLoading` becomes `false` as soon as `dataReady` is `true` | unit | same | No — Wave 0 |

### Sampling Rate

- **Per task commit:** `cd web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js`
- **Per wave merge:** `cd web/frontend && npx vitest run --reporter=verbose`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `web/frontend/src/__tests__/YOLOVisualizer.spec.js` — Add a new `describe` block for loading
  screen tests (LOAD-01, LOAD-02). Existing file already has the mount helper and all mocks;
  new tests extend the same file. The existing `fetch` mock resolves immediately — tests need to
  control resolution timing (use `vi.fn()` returning a manually-resolved promise to test the
  intermediate `isLoading = true, dataReady = false` state).

## Sources

### Primary (HIGH confidence)

- Direct codebase inspection: `web/frontend/src/components/YOLOVisualizer.vue` — `loadCase()` at
  line 1060, `currentPartVideoUrl` watcher at line 1670, template root at line 3
- Direct codebase inspection: `web/frontend/src/__tests__/YOLOVisualizer.spec.js` — established
  mount pattern, mock infrastructure, setupState access
- Phase 01-01-SUMMARY.md — confirmed `loadedmetadata` callback is the correct hook; Phase 1
  established that the watcher fires correctly after video src assignment
- MDN: `HTMLMediaElement: loadedmetadata event` — fires when duration and dimensions are known,
  before the video can be played

### Secondary (MEDIUM confidence)

- REQUIREMENTS.md — "No new npm dependencies" explicitly out of scope; "no elaborate skeletons"
  for internal tool confirmed
- SUMMARY.md (project research) — identified "two flags (dataReady + videoReady)" as the
  correct pattern; confirmed pitfall of clearing loading after fetch only

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — no new stack; all primitives already in use in this component
- Architecture: HIGH — based on direct code reading of the exact lines that will change
- Pitfalls: HIGH — derived from actual code paths, not hypothetical; prediction-mode pitfall
  confirmed by reading `currentPartVideoUrl` computed at line 708
- Test plan: HIGH — existing test infrastructure is confirmed working from Phase 1

**Research date:** 2026-03-07
**Valid until:** 2026-06-07 (stable APIs; only invalidated if `currentPartVideoUrl` watcher or
`loadCase` are significantly refactored)
