---
phase: 2
slug: loading-screen
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest 3.x + @vue/test-utils |
| **Config file** | `web/frontend/vite.config.js` |
| **Quick run command** | `cd web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js` |
| **Full suite command** | `cd web/frontend && npx vitest run --reporter=verbose` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js`
- **After every plan wave:** Run `cd web/frontend && npx vitest run --reporter=verbose`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 02-01 | 0 | LOAD-01, LOAD-02 | unit | `cd web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js` | ❌ W0 | ⬜ pending |
| 2-01-02 | 02-01 | 1 | LOAD-01, LOAD-02 | unit | same | ✅ W0 | ⬜ pending |
| 2-01-03 | 02-01 | 1 | LOAD-01, LOAD-02 | unit | same | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `web/frontend/src/__tests__/YOLOVisualizer.spec.js` — Add new `describe` block for loading screen tests (LOAD-01, LOAD-02). Existing file has mount helper and fetch mock; new tests extend same file. Tests need controlled fetch resolution timing to assert intermediate loading state.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Loading indicator visible during case load | LOAD-01 | Visual rendering cannot be asserted in vitest | Select a case on slow network; confirm loading state appears immediately |
| Viewer hidden until video ready | LOAD-02 | DOM visibility during async init | Select case; confirm viewer not visible until video is fully loaded |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
