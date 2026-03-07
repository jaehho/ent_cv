---
phase: 1
slug: frame-state-fix
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-07
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest ^2.0.0 + @vue/test-utils ^2.4.0 |
| **Config file** | `web/frontend/vite.config.js` (vitest block) |
| **Quick run command** | `cd /home/jaeho/ent_cv/web/frontend && npm test` |
| **Full suite command** | `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd /home/jaeho/ent_cv/web/frontend && npm test`
- **After every plan wave:** Run `cd /home/jaeho/ent_cv/web/frontend && npx vitest run --reporter=verbose`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~5 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | BUG-01, BUG-02 | unit | `cd web/frontend && npx vitest run src/__tests__/YOLOVisualizer.spec.js` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | BUG-01 | unit | same | ✅ W0 | ⬜ pending |
| 1-01-03 | 01 | 1 | BUG-02 | unit | same | ✅ W0 | ⬜ pending |
| 1-01-04 | 01 | 1 | BUG-01, BUG-02 | unit | same | ✅ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `web/frontend/src/__tests__/YOLOVisualizer.spec.js` — unit test stubs for BUG-01 (frame 0 on load) and BUG-02 (canvas draw at frame 0); mocks for `fetch`, `videoRef`, `nextTick`, `scheduleDraws`

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Visual canvas overlay aligns with video at frame 0 | BUG-02 | Canvas pixel content cannot be inspected in vitest | Load any case, verify overlay is blank or correct at frame 0 |
| Zoom/pan/speed reset visible in UI after case switch | BUG-01 | Reactive state tied to DOM rendering | Switch cases, verify timeline zoom = 1, pan = start, speed buttons show 1x |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
