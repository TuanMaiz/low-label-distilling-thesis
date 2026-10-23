# Phase 3: Train arms

## Context Links

- Recipe: `scripts/run_wdc_qwen_variant_prompt.sh`
- Datasets: `data/cache/wdc_products/refinement/category-rule-<arm>/`

## Overview

Priority P1. Pending. Blocked on GPU approval. One run per arm, same recipe,
same GPU type, validation only.

## Requirements

- New `scripts/run_wdc_qwen_treatment_arms.sh <action> <arm>` with `plan`,
  `preflight`, `train`, `evaluate`, `package` (predictions and metrics only,
  no weights unless asked). Train targets and output root chosen by arm; the
  recipe copied from the variant-prompt runner and frozen in the contract.
- Expected `train_rows` per arm read from the arm's `manifest.json`.
- Validation predictions saved per pair, with the threshold chosen by
  `select_decision_threshold`.

## Implementation Steps

1. Runner with `plan` output reviewed (CPU).
2. Train A, B, C on the rented GPU; package; verify checksums locally.

## Todo List

- [ ] Runner and preflight
- [ ] Arm runs complete
- [ ] Results packaged and verified

## Success Criteria

- Each arm has validation predictions, metrics and threshold from the same recipe.

## Risk Assessment

- GPU nondeterminism. Mitigation: same GPU type; single-run limitation stated.

## Next Steps

- Phase 4.
