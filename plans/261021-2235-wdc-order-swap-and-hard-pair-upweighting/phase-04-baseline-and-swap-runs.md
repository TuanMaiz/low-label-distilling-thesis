# Phase 4: Baseline and swap runs

## Context Links

- Contract: `./contract.md` (recipe rule, shared recipe)
- Runner patterns: `scripts/run_wdc_qwen_variant_prompt.sh`, `scripts/run_wdc_qwen_vertical_slice.sh`
- Earlier majority run: `outputs/majority-baseline/` (validation match F1 0.8191; no weights kept)
- Colab notes: `analysis/order-swap-probe.md` (uninstall `torchao`, pin `peft==0.20.0`, `USE_TF=0`, `set -e`)

## Overview

Priority P1. Pending. Two runs on majority labels with Phase 2 logging:
(a) baseline (`none`), (b) `random_swap`. Choose the recipe by the contract rule.

## Key Insights

- Running both now avoids redoing the baseline if swap is adopted; the chosen run supplies the per-pair scores for Phase 6.
- The earlier majority run trained on `data/cache/wdc_products/majority_labels/train.jsonl`, which is not in the local folder. It must be rebuilt reproducibly first.
- No saved weights or test predictions exist for any majority-label run; (a) fills that gap.

## Requirements

**Majority training targets**
- New `supervision/build_majority_targets.py`: input `data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl` + `majority_labels/llm_unrefined.csv`; output `majority_labels/train.jsonl` in the same row schema as `refinement/quick-tier-v1/train.jsonl` (`label`, `target_text`, `label_source=llm_majority`, `split=train`), source order, plus `train.manifest.json` (input hashes, counts).
- Checks: 2,500 unique IDs; 495 match / 2,005 non-match; `input_text` identical to the union of the three `oof/majority-3fold/fold_*/test-heldout.jsonl` files; rebuild gives identical bytes.

**Runner `scripts/run_wdc_qwen_upweight.sh`**
- Arms: `baseline` (baseline config, majority targets), `swap` (swap config, majority targets), later `upweight` and `control` (Phase 7 targets).
- Actions: `config` (print resolved paths and hashes), `preflight` (CPU: files, hashes, config loads, token-length audit on train and validation), `smoke` (32 pairs, 1 epoch), `train ARM`, `evaluate ARM` (validation with saved threshold via `experiments.evaluate_student`), `probe ARM` (`analysis.order_swap_probe` score + summarize on validation), `package ARM` (tar of summary, manifest, threshold, predictions, `train_pair_scores.jsonl`, `val_predictions/`, `epoch_adapters/`, `best_adapter/`, plus SHA-256).
- Hyperparameters exactly as the contract's shared recipe; always `--pair-scores-dir`.
- Output root `outputs/uncertainty_refinement/wdc-qwen-upweight/<arm>/`; refuses to overwrite an existing run directory.
- Refuses any test input unless `ALLOW_TEST=1` (used only in Phase 9).

## Related Code Files

- Create: `supervision/build_majority_targets.py`, `tests/test_majority_targets.py`, `scripts/run_wdc_qwen_upweight.sh`
- Modify: `analysis/order-swap-probe.md` (link the runner's `probe` action)

## Implementation Steps

1. Build and verify majority targets (CPU).
2. Runner with `config`, `preflight`, `smoke`; run them locally (CPU) where possible.
3. On the GPU host: `smoke`, then `train baseline`, `train swap` (same GPU type for both).
4. `evaluate` and `probe` both; `package` both and download; verify checksums.
5. Compare with the earlier majority run (0.8191) as a sanity check only.
6. Apply the recipe rule; write the decision and numbers into `contract.md` (dated amendment).

## Todo List

- [ ] Majority targets rebuilt and verified
- [ ] Runner and preflight
- [ ] Baseline run
- [ ] Swap run
- [ ] Evaluate, probe, package
- [ ] Recipe decision recorded

## Success Criteria

- Both runs complete with pair-score logs, per-epoch adapters and verified packages.
- Recipe chosen by the predeclared rule only.

## Risk Assessment

- Free GPU sessions disconnect (a T4 run may take several times the 1.7 h of a 3090). Mitigation: Kaggle's longer sessions or a rented GPU; resumable outputs on persistent storage.
- Hardware differences between runs. Mitigation: same GPU type and precision for all Phase 4 and Phase 8 runs; recorded in summaries.

## Security Considerations

- Do not commit weights, tokens or Drive paths with personal data.

## Next Steps

- Phase 5 (uses the chosen recipe), Phase 6 (uses the chosen run's pair scores).
