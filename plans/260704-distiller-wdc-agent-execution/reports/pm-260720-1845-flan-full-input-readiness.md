---
title: FLAN Full-Input Diagnostic Readiness
status: in_progress
created: 2026-07-20
plan: ../plan.md
phase: 5
---

# FLAN Full-Input Diagnostic Readiness

## Project Status

| Phase | Status | Criteria | Current interpretation |
|---|---|---:|---|
| 1 — Research contract | Completed | 4/4 | Contract frozen |
| 2 — Pivot cleanup | Completed | 3/3 | Active thesis direction documented |
| 3 — Teacher pipeline | Completed | 10/11 | Optional budget-256 cache deferred |
| 4 — Target builder | Completed | 5/7 | Optional budget-256 targets deferred |
| 5 — Pilot students | In progress | 11/14 | Two completed `REVISE` runs; two screening runs pending |
| 6 — Full budget study | Pending | 0/6 | Blocked on viable student selection |
| 7 — Failure and cost analysis | Pending | 0/9 | Depends on Phases 5–6 |
| 8 — Thesis handoff | Pending | 0/5 | Depends on Phase 7 |

Conditional budget-256 criteria in completed Phases 3–4 remain intentionally
open; their prerequisite expansion has not been approved.

## Completed This Iteration

- Added a separate `flan-t5-base-full-input` diagnostic contract at 2,700
  input tokens, above the measured 2,649-token fixed-input maximum.
- Disabled truncation for that diagnostic so future overflow fails rather than
  silently changing an input pair.
- Preserved the historical 512-token FLAN result and the first ModernBERT
  collapse as negative validation evidence.
- Kept FLAN sequence-likelihood threshold calibration as a documented optional
  follow-up; it is not part of this controlled rerun.
- Kept all teacher labels, selection manifests, validation rows, and the direct
  LLM baseline fixed. Test remains untouched.

## Current Gate

Implementation readiness is complete, but experiment evidence is not. Phase 5
stays in progress until the repaired ModernBERT and/or full-input FLAN archive
is returned, reviewed, and a viable compact student is selected for Phase 6.

## Next Actions

1. Run `flan_t5_base_full_input.json` on A100 when possible under a fresh
   `STUDENT_OUTPUT_ROOT`.
2. Return and verify the compact archive and provenance contracts.
3. Compare match F1 first, then macro F1, accuracy, prediction balance, timing,
   and cost against the historical FLAN, ModernBERT, and direct-LLM evidence.
4. Continue model screening if neither prepared diagnostic is close enough for
   a meaningful active-versus-random comparison.

## Risks And Unresolved Questions

- A 2,700-token FLAN run has quadratic attention cost and may require A100 plus
  the predeclared long-input validation batch of 4.
- No result yet shows that longer FLAN inputs improve match F1.
- No final compact student or acceptance threshold has been selected.
- Validation-only screening must finish before any test evaluation.
