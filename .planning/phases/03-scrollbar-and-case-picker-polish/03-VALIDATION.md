---
phase: 3
slug: scrollbar-and-case-picker-polish
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Vitest (inline in vite.config.js) |
| **Config file** | vite.config.js (`test.environment: jsdom`) |
| **Quick run command** | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` |
| **Full suite command** | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose`
- **After every plan wave:** Run `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 0 | PICK-01 | unit | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run src/__tests__/CasePicker.spec.js` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | SCROLL-01 | manual | — | N/A | ⬜ pending |
| 3-01-03 | 01 | 1 | SCROLL-02 | manual | — | N/A | ⬜ pending |
| 3-01-04 | 01 | 1 | PICK-01 | unit | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run src/__tests__/CasePicker.spec.js` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `web/frontend/src/__tests__/CasePicker.spec.js` — stubs for PICK-01 (asserts `.upload-root` has `overflow-y: auto` in computed style OR checks component renders a scrollable container)

*SCROLL-01 and SCROLL-02 are manual-only (CSS pseudo-element rendering cannot be verified in jsdom); no Wave 0 test file required for those.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Thin dark scrollbar visible in Chromium | SCROLL-01 | CSS pseudo-elements not rendered in jsdom | Open app in Chrome/Edge, scroll classes panel — verify thin dark scrollbar |
| Thin dark scrollbar visible in Firefox | SCROLL-01 | CSS scrollbar-color not rendered in jsdom | Open app in Firefox, scroll classes panel — verify thin dark scrollbar |
| All scrollable areas show custom scrollbar | SCROLL-02 | Visual rendering required | Scroll classes panel and any overflow areas in YOLOVisualizer |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
