---
phase: 4
slug: transitions-matrix-responsive-resize
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 4 — Validation Strategy

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
| 4-01-01 | 01 | 1 | TRANS-01 | unit | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` | ✅ existing | ⬜ pending |
| 4-01-02 | 01 | 1 | TRANS-01, TRANS-02 | unit | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` | ✅ existing | ⬜ pending |
| 4-01-03 | 01 | 1 | TRANS-02 | manual | — | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements.

`web/frontend/src/__tests__/YOLOVisualizer.spec.js` already exists with `ResizeObserver` mocked (lines 72–76). New test cases for TRANS-01/TRANS-02 are added to this existing file — no Wave 0 stub file needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Matrix stays square while dragging panel divider | TRANS-02 | Visual rendering/layout not testable in jsdom | Load a filtered case, drag right-panel divider left and right — matrix should remain square at all widths |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
