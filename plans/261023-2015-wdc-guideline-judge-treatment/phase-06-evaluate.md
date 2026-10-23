# Phase 6: Compare, test once, final analysis

## Overview

Priority P1. Pending. Paired comparison on validation, one test run for the
frozen arms, then train-gold analysis.

## Requirements

- New `analysis/paired_comparison.py` (+ tests): paired bootstrap of ΔF1
  (2,000 resamples, seed 0, 95% CI) and exact McNemar on per-pair correctness.
- Comparisons: C vs A, C vs B, C vs D (if run), on validation.
- Test: run once for all frozen arms after explicit authorization; report the
  same metrics and paired tests; restate the WDC test caveat.
- Train-gold analysis (last): share of flips that agree with gold, gold label
  of removed and upweighted pairs, by judge step.
- Write a journal entry and update `docs/research/decisions.md`.

## Todo List

- [ ] Paired comparison tool and tests
- [ ] Validation comparison
- [ ] Test run (authorized, once)
- [ ] Train-gold analysis
- [ ] Journal and records

## Success Criteria

- All comparisons reported with CIs; no arm or rule changed after seeing test.

## Risk Assessment

- Differences within noise (single runs, ~6% of data touched). Mitigation: report CIs honestly; seeds repeat only as a predeclared follow-up.

## Next Steps

- Repeat on DBLP-ACM as the clean confirmation (separate plan).
