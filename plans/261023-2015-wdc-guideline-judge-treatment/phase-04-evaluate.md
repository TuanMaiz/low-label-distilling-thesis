# Phase 4: Compare, test once, final analysis

## Overview

Priority P1. In progress: paired comparison and gold audit done 2026-10-28 (seed 42). Paired comparison on validation, one test run for the
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

- [x] Paired comparison tool and tests
- [x] Validation comparison
- [x] Test run (authorized, once per arm, 2026-10-26/27)
- [x] Train-gold analysis
- [ ] Version 2 decision, journal and records

## Success Criteria

- All comparisons reported with CIs; no arm or rule changed after seeing test.

## Risk Assessment

- Differences within noise (single runs, about 6% of data touched). Mitigation: report CIs honestly; repeated seeds only as a predeclared follow-up.

## Next Steps

- Version 2 decision; DBLP-ACM confirmation (separate plan).

## Results so far (seed 42; 2026-10-28)

Paired bootstrap (2,000 resamples, seed 0, 95% percentile CI) and exact
McNemar, `outputs/uncertainty_refinement/wdc-qwen-treatment/paired-comparisons.json`.
Δ = F1(left) − F1(right), in F1 points. The CI covers test-pair sampling only,
not seed variance.

| Split | Comparison | ΔF1 | 95% CI | McNemar p |
|---|---|---:|---|---:|
| Validation | C − A | −2.36 | [−4.07, −0.58] | 0.008 |
| Validation | C − B | −0.84 | [−2.36, +0.69] | 0.483 |
| Validation | B − A | −1.52 | [−3.08, +0.15] | 0.053 |
| Validation | A − G | −0.24 | [−1.93, +1.53] | 0.744 |
| Test | C − A | −1.74 | [−3.57, +0.28] | 0.103 |
| Test | C − B | −0.32 | [−2.03, +1.33] | 1.000 |
| Test | B − A | −1.43 | [−3.34, +0.54] | 0.090 |
| Test | A − G | +0.66 | [−1.19, +2.65] | 0.650 |

Train-gold audit of arm C,
`outputs/uncertainty_refinement/wdc-qwen-treatment/category-rule-gold-audit.json`:

| Step : action | Pairs | Teacher label = gold | Gold match |
|---|---:|---:|---:|
| 1 : remove | 7 | 6 | 4 |
| 2 : flip | 10 | 10 | 10 |
| 2 : upweight | 47 | 39 | 8 |
| 3 : remove | 86 | 67 | 58 |
| 6 : upweight | 5 | 5 | 5 |
| invalid : keep | 3 | 3 | 1 |

Teacher label error rate: 28/158 (17.7%) inside the detected set versus
51/2,342 (2.2%) outside; the detected 6.3% of pairs hold 28 of the 79 teacher
errors.
