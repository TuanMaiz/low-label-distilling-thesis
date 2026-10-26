# Phase 3: Train arms

## Context Links

- Runbook: `docs/runbooks/rented-v100-treatment-arms.md`

- Recipe: `scripts/run_wdc_qwen_variant_prompt.sh`
- Datasets: `data/cache/wdc_products/refinement/category-rule-<arm>/`

## Overview

Priority P1. In progress: runner done 2026-10-26; GPU runs blocked on approval. One run per arm, same recipe,
same GPU type, validation only.

## Requirements

- `scripts/run_wdc_qwen_treatment_arms.sh <action> <arm>` (written 2026-10-26; defaults Tesla V100-SXM2-32GB + fp16, contract A3; `PACKAGE_WEIGHTS=adapter` by default, merged model excluded) with `plan`,
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

- [x] Runner and offline tests (`tests/test_wdc_qwen_treatment_arms.py`, 9/9)
- [x] Arm data in git (2026-10-26): `category-rule-{A,B,C}/` committed; serialized validation/test were already tracked
- [ ] GPU preflight per arm
- [ ] Arm runs complete
- [ ] Results packaged and verified

## Success Criteria

- Each arm has validation predictions, metrics and threshold from the same recipe.

## Risk Assessment

- GPU nondeterminism. Mitigation: same GPU type; single-run limitation stated.

## Next Steps

- Phase 4.
