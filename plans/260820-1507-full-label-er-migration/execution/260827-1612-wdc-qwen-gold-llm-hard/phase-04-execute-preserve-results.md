---
phase: 4
title: "Execute and Preserve RTX 3090 Results"
status: pending
priority: P1
effort: "GPU-runtime dependent"
dependencies: [3]
---

# Phase 4: Execute and Preserve RTX 3090 Results

## Context Links

- Contract: `../../research/wdc-qwen-training-vertical-slice-contract.md`
- Rental handoff: `./phase-03-verify-rental-handoff.md`

## Overview

Run exactly one gold cell and one `llm_hard` cell on the rented RTX 3090,
verify both validation result bundles, and preserve a hash-verifiable package
before terminating the rental.

## Requirements

- Functional: execute in order setup, preflight, gold, `llm_hard`, verify, package.
- Functional: publish complete validation artifacts for both arms.
- Non-functional: same rental workflow, pushed code commit, matching recorded
  runtime properties, precision policy, and frozen settings for both arms.
- Non-functional: test split remains untouched.

## Architecture

```text
RTX 3090 checkout at approved commit
  -> setup
  -> preflight (new runtime/input/code contract)
  -> train-gold --confirm-full-training
  -> verify gold
  -> copy verified gold package off-rental
  -> train-llm-hard --confirm-full-training
  -> verify both
  -> package-results
  -> copy package + checksum off rental
```

## Related Inputs and Outputs

- Input: `data/cache/wdc_products/full_label_targets/gold.jsonl`
- Input: `data/cache/wdc_products/full_label_targets/llm_hard.jsonl`
- Input: `data/cache/wdc_products/serialized/validation.jsonl`
- Output: `outputs/full_label/wdc-qwen-vertical-slice/wdc_products_80cc_small_100un/qwen3-reranker-0-6b/gold/run/`
- Output: `outputs/full_label/wdc-qwen-vertical-slice/wdc_products_80cc_small_100un/qwen3-reranker-0-6b/llm_hard/run/`
- Output: same root, `full-experiment-manifest.json`, results archive, and SHA-256.

## Implementation Steps

1. Pull the approved commit; confirm clean intended code state.
2. Run setup and fresh preflight with default `EXPECTED_GPU_SUBSTRING=3090`.
3. Inspect runtime identity, resolved precision, 7,500-row input audit, and hashes.
4. Run the confirmed gold action; stop and review if any gate fails.
5. Verify gold completion, package its complete checkpoint tree and result files,
   and copy that package off-rental before starting `llm_hard`.
6. Run the confirmed `llm_hard` action with no setting changes.
7. Verify both result bundles and their shared/differing contract fields.
8. Package both complete checkpoint trees plus declared result artifacts and
   checksum; copy the final package off the rental.
9. Reverify archive checksum locally before terminating the machine.
10. Record results in a journal without interpreting validation as final test.

## Execution Commands

```bash
bash scripts/run_wdc_qwen_vertical_slice.sh setup
bash scripts/run_wdc_qwen_vertical_slice.sh preflight
bash scripts/run_wdc_qwen_vertical_slice.sh train-gold --confirm-full-training
bash scripts/run_wdc_qwen_vertical_slice.sh package-arm gold
bash scripts/run_wdc_qwen_vertical_slice.sh train-llm-hard --confirm-full-training
bash scripts/run_wdc_qwen_vertical_slice.sh verify-results
bash scripts/run_wdc_qwen_vertical_slice.sh package-results
```

## Per-Arm Acceptance

- [ ] Exactly 2,500 training and 2,500 validation rows.
- [ ] Finite train/validation loss history and selected threshold.
- [ ] CUDA device name contains `3090`; no CPU fallback.
- [ ] Checkpoint manifest verifies after merged-checkpoint reload.
- [ ] Exactly 2,500 unique validation IDs in official order.
- [ ] Zero invalid predictions; finite normalized probabilities.
- [ ] Recomputed metrics equal stored metrics.
- [ ] Match precision/recall/F1, macro F1, accuracy, timing, throughput, epochs,
  optimizer steps, precision, and runtime identity are persisted.

## Failure and Recovery

- Stop on OOM, non-finite loss, overflow, stale contract, CPU fallback,
  checkpoint/reload failure, prediction/metric mismatch, LLM access, or test access.
- Preserve failed output. Do not change batch, accumulation, precision policy,
  length, warmup, epochs, or other settings to recover.
- Quarantine a partial attempt only after inspection and explicit approval.
- Restart that arm from scratch under the same frozen contract; do not count the
  failed infrastructure attempt as a completed run.
- Do not start `llm_hard` while gold has an unresolved failure.
- Do not start `llm_hard` until the verified gold package exists off-rental.
- Do not create the final two-arm package until both arms independently verify.

## Success Criteria

- [ ] Both arms complete once with matching provenance and frozen settings.
- [ ] Full validation outputs are preserved and independently hash-verifiable.
- [ ] The test split remains untouched.
- [ ] Result artifacts are ready for comparison and cost aggregation.

## Risk Assessment

The main operational risks are rental interruption, OOM, and partial outputs.
Fail closed, preserve evidence, and prohibit outcome-driven reruns or tuning.

## Security and Data Integrity

No provider key is needed. Package only declared experiment artifacts; exclude
`.env`, base caches, unrelated outputs, and ignored upstream labeler evidence.

## Next Steps

After validation review, decide whether to authorize the WDC final-test gate or
return to the broader Phase-1 dataset/model selections. Do not infer test
authorization from completion of this plan.
