# 2026-11-05 — Companies A vs G paired test; LLM label quality vs gold on train

No GPU, no paid calls. Aggregates only (Companies and Geo-ER records are not
republished). Outputs: `outputs/uncertainty_refinement/madi-companies-qwen/paired-comparisons.json`,
`outputs/uncertainty_refinement/label-quality-vs-gold-train.json` (both git-ignored).

## Companies, A (LLM majority) vs G (gold): match F1, paired bootstrap (2,000, seed 0), exact McNemar

| Split | Pairs | A | G | ΔF1 (A−G) | 95% CI | McNemar A-only / G-only, p |
|---|---|---|---|---|---|---|
| Validation | 948 | 94.2 | 94.6 | −0.4 | [−1.7, +0.9] | 8 / 11, 0.648 |
| Test | 599 | 88.8 | 89.7 | −1.0 | [−2.7, +0.7] | 5 / 9, 0.424 |
| Test, Forbes–DBpedia | 140 | 92.0 | 92.6 | −0.7 | [−3.3, +1.5] | 1 / 2, 1.000 |
| Test, Forbes–FullContact | 459 | 87.1 | 88.2 | −1.1 | [−3.6, +1.0] | 4 / 7, 0.549 |

No significant difference. The test CI's lower bound (−2.7) is below −2, so
non-inferiority at the proposed δ = 2 is not shown on Companies (one seed).
WDC for reference: test Δ +0.7, CI [−1.2, +2.7].

## LLM majority labels vs gold, train split

| Dataset | Pairs | Gold match | LLM match | Agree | LLM precision | LLM recall | Wrong: LLM match / LLM non-match |
|---|---|---|---|---|---|---|---|
| WDC (`llm_hard`) | 2,500 | 500 | 495 | 96.8% | 92.5 | 91.6 | 37 / 42 |
| Companies | 1,968 | 698 | 584 | 92.9% | 97.8 | 81.8 | 13 / 127 |
| Geo-ER | 2,500 | 727 | 761 | 94.9% | 89.4 | 93.5 | 81 / 47 |

- Companies: the teacher is strict. Most disagreements are gold matches it calls
  non-match (127; FullContact 101, DBpedia 26). 120 of the 127 have different
  names; in a sample they are parent/subsidiary or related-company pairs that
  gold counts as one company (the known inconsistent parent/subsidiary rule, D23).
- Geo-ER: the teacher is lenient. More LLM matches gold rejects (81, median
  distance 94 m, 41 within 100 m) than the reverse (47, median 72 m).
- 3/3 agreement does not mean correct: most disagreements with gold are 3/3
  (WDC 73 of 79, Companies 133 of 140, Geo-ER 113 of 128). At 2/3 about half
  disagree with gold (6/13, 7/16, 15/30).
