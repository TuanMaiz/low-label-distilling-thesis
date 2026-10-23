# Phase 4: Compare, test once, final analysis

## Overview

Priority P1. Pending. Paired comparison on validation, one test run for the
frozen arms, then train-gold analysis.

## Requirements

- New `analysis/paired_comparison.py` (+ tests): paired bootstrap of ΔF1
  (2,000 resamples, seed 0, 95% CI) and exact McNemar on per-pair correctness.
- Comparisons: C vs A, C vs B, on validation.
- Test: run once for all frozen arms after explicit authorization; same
  metrics and paired tests; restate the WDC test caveat.
- Train-gold analysis (last): share of flips that agree with gold; gold label of
  removed and upweighted pairs, by step.
- Decide on version 2 (judge panel on step-3 pairs) from these results.
- Journal entry and `docs/research/decisions.md` update.

## Todo List

- [ ] Paired comparison tool and tests
- [ ] Validation comparison
- [ ] Test run (authorized, once)
- [ ] Train-gold analysis
- [ ] Version 2 decision, journal and records

## Success Criteria

- All comparisons reported with CIs; no arm or rule changed after seeing test.

## Risk Assessment

- Differences within noise (single runs, about 6% of data touched). Mitigation: report CIs honestly; repeated seeds only as a predeclared follow-up.

## Next Steps

- Version 2 decision; DBLP-ACM confirmation (separate plan).
