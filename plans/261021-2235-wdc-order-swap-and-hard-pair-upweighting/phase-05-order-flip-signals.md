# Phase 5: Student order-flip signal (OOF)

## Context Links

- `experiments/wdc_qwen_oof.py`, `scripts/run_wdc_qwen_oof.sh` (`package_results` excludes adapters today)
- Frozen folds: `data/cache/wdc_products/oof/majority-3fold/` (held-out labels are majority labels)
- Existing OOF: `outputs/uncertainty_refinement/wdc-qwen-oof/oof_predictions.csv`
- Probe: `analysis/order_swap_probe.py` (`--pair-ids`)
- Deferred teacher signal: `REVISIT.md` → "Swapped-Order Teacher Pass"

## Overview

Priority P2. Pending. Re-run the three OOF folds with the chosen recipe and
logging, keep the fold adapters, and score each held-out fold in both orders.
Result: a per-training-pair student order-flip flag and size.

## Key Insights

- The flip signal was shown only on validation model errors; whether it marks training label errors is measured only in Phase 9.
- It comes from the same OOF models as student support and disagreement, so it is not independent confirmation.
- Fold weights were lost before because packaging leaves out `best_adapter/`.

## Requirements

- OOF workflow: optional `STUDENT_CONFIG` selection (baseline or swap config), `--pair-scores-dir` per fold, package members include `fold_XX/train/best_adapter/` and the pair-score files; new `OUTPUT_ROOT=outputs/uncertainty_refinement/wdc-qwen-oof-v2`; `verify-data` and existing tests still pass.
- Per fold after training: `analysis.order_swap_probe score --model fold_XX/train/best_adapter --input oof/majority-3fold/fold_XX/test-heldout.jsonl` (both orders); `summarize --threshold-json fold_XX/train/decision_threshold.json --allow-partial`.
- New `analysis/merge_order_signals.py` → `order_signals.csv`: `pair_id`, `fold_id`, `log_odds_original`, `log_odds_swapped`, `delta`, `abs_delta`, `flip` (at the fold's threshold). Exactly 2,500 rows, each pair once.
- Agreement report, old vs new OOF: Pearson r of log-odds, decision agreement, overlap of CL candidates if `supervision.build_confident_learning` is re-run on v2.
- No gold read.

## Related Code Files

- Modify: `experiments/wdc_qwen_oof.py`, `scripts/run_wdc_qwen_oof.sh`, `tests/test_wdc_qwen_oof.py`
- Create: `analysis/merge_order_signals.py`, `tests/test_merge_order_signals.py`
- Outputs: `outputs/uncertainty_refinement/wdc-qwen-oof-v2/`

## Implementation Steps

1. Workflow changes and tests (CPU); `verify-data` passes.
2. Train three folds on the GPU host; package; verify checksums locally.
3. Probe held-out folds in both orders; merge into `order_signals.csv`.
4. Agreement report; record which OOF version Phase 6 uses (contract item 5).

## Todo List

- [ ] OOF workflow changes and tests
- [ ] Three folds trained, adapters kept
- [ ] Both-order probe per fold
- [ ] `order_signals.csv`
- [ ] Agreement report

## Success Criteria

- Every training pair has one student flip value; fold adapters are in the verified package.

## Risk Assessment

- New OOF differs from old (GPU nondeterminism, recipe). Mitigation: report agreement; use one version consistently, as the contract says.
- Phase 5 finishes after Phase 6 starts. Mitigation: Phase 6 runs without the flip signal (teacher-trust then ranks by current loss only), as predeclared.

## Security Considerations

- None.

## Next Steps

- Phase 6.
