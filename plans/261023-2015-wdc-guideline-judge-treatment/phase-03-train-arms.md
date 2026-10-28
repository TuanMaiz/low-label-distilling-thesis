# Phase 3: Train arms

## Context Links

- Runbook: `docs/runbooks/rented-v100-treatment-arms.md`

- Recipe: `scripts/run_wdc_qwen_variant_prompt.sh`
- Datasets: `data/cache/wdc_products/refinement/category-rule-<arm>/`

## Overview

Priority P1. Completed 2026-10-27 (seed 42; extra seeds pending a later amendment). One run per arm, same recipe,
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
- [x] GPU preflight per arm
- [x] Arm runs complete
- [x] Results packaged and verified

## Success Criteria

- Each arm has validation predictions, metrics and threshold from the same recipe.

## Risk Assessment

- GPU nondeterminism. Mitigation: same GPU type; single-run limitation stated.

## Next Steps

- Phase 4.

## Result (seed 42, Tesla V100-SXM2-32GB, fp16)

Archives verified (checksum, adapter manifest, metrics recomputed from
predictions, identical `pip freeze`). Results without weights and without
`train.log` are tracked under `outputs/uncertainty_refinement/wdc-qwen-treatment/arm-{A,B,C,G}/`;
full archives stay local in `outputs/newABC/`.

| Arm | Val match F1 | Test match F1 | Test P | Test R | Epochs (best) | GPU-h |
|---|---:|---:|---:|---:|---:|---:|
| G: human labels (A4) | 0.8275 | 0.7587 | 0.6612 | 0.8900 | 8 (5) | 1.53 |
| A: LLM labels | 0.8250 | 0.7653 | 0.6657 | 0.9000 | 10 (7) | 1.92 |
| B: remove all 158 | 0.8098 | 0.7510 | 0.6434 | 0.9020 | 9 (6) | 1.63 |
| C: category rule | 0.8015 | 0.7479 | 0.6481 | 0.8840 | 9 (6) | 1.69 |

Single seed per arm; no paired test yet (Phase 4).
