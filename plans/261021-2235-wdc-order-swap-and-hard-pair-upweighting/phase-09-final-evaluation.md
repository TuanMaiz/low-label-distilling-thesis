# Phase 9: Final evaluation

## Context Links

- Frozen arms: Phase 4 (baseline, swap) and Phase 8 (upweight, control)
- Gold-arm reference: `outputs/new/wdc-qwen-gold-vs-llm-hard/gold/run/best_adapter` (never evaluated on test)
- Train gold: `data/cache/wdc_products/full_label_targets/gold.jsonl`; test: `data/cache/wdc_products/serialized/test.jsonl`
- Deferred: `REVISIT.md` → "LLM-Labeled Validation"

## Overview

Priority P1. Pending. The only phase that reads train gold or test gold. It
starts after all arms and the rule are frozen and the researcher authorizes
test access.

## Key Insights

- WDC test already shaped method ideas; report it as a non-pristine check and plan confirmation on DBLP-ACM.
- Per-epoch adapters and validation predictions allow re-selecting checkpoints without retraining.

## Requirements

**Test, once**
- Arms: baseline, swap (as ablation if not chosen), upweight, control, gold-arm reference.
- Runner `evaluate ARM --split test` with `ALLOW_TEST=1`; saved validation thresholds only.
- Metrics per arm; `compare_arms` for upweight vs baseline, upweight vs control, swap vs baseline; order-flip probe on test for the chosen recipe and upweight.

**Train-gold analysis (new `analysis/detection_quality.py`)**
- Targets: the 79 pairs where the teacher majority label differs from train gold (recount from files).
- Signals ranked over 2,500 pairs: teacher consistency (2/3 first), CL rank, teacher–student disagreement, student order flip (`abs_delta`), RHO priority, current loss.
- Precision@k and recall@k for k ∈ {25, 50, 100, 250}; average precision; pairwise overlap of top-250 sets.
- For each Phase 6 set: share of teacher labels that disagree with gold.

**Validation robustness (new `analysis/validation_robustness.py`, no retraining)**
- 10 seeded random 500-pair validation subsets. For each: choose the epoch by macro F1 (tie: match F1) from `val_predictions/epoch-NN.jsonl`; choose the threshold with `select_decision_threshold` on the subset.
- Test predictions for every epoch adapter of each frozen arm (inference only); report the test match F1 of each subset choice vs the full-validation choice (mean, min, max).

## Related Code Files

- Create: `analysis/detection_quality.py`, `analysis/validation_robustness.py`, tests for both
- Modify: `scripts/run_wdc_qwen_upweight.sh` (`evaluate --split test`, `evaluate-epochs`)
- Outputs: `outputs/uncertainty_refinement/wdc-qwen-upweight/final/`

## Implementation Steps

1. Researcher authorizes test access in writing (date in the contract).
2. Test inference for frozen arms and all epoch adapters (GPU, < 1 h); package.
3. Test comparisons and probe.
4. Join train gold; detection and set-correctness analysis.
5. Validation-subset robustness.
6. Journal; update `docs/research/` records; thesis tables.

## Todo List

- [ ] Test authorization recorded
- [ ] Test inference and comparisons
- [ ] Train-gold analysis
- [ ] Validation robustness
- [ ] Journal and records

## Success Criteria

- Each arm's test result reported once with CIs; every gold read happens after the freeze and is listed in the final report.

## Risk Assessment

- Temptation to iterate after seeing test. Mitigation: any change after this phase is a new plan, confirmed on DBLP-ACM.

## Security Considerations

- Gold joins stay in analysis outputs; selection code never imports them.

## Next Steps

- Port the frozen method to DBLP-ACM; revisit `REVISIT.md` items.
