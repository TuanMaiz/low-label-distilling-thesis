---
title: "WDC Qwen OOF preparation and execution plan"
date: 2026-09-30
plan: "260928-0938-uncertainty-aware-llm-label-refinement-fundamental"
---

# WDC Qwen OOF preparation and execution plan

## Approved decisions

Researcher-approved on 2026-09-30:

- Use dataset-defined `pair_id` as the OOF example identity.
- Reserve a deterministic class-aware 20% of each non-held-out pool for inner
  validation.
- Use Qwen3-Reranker-0.6B as the primary detector across datasets while all
  three compact students remain downstream evaluation models.
- Keep a second WDC-only detector as an optional robustness ablation rather
  than multiplying the main OOF matrix across all students.
- Preserve canonical given-label IDs and both raw OOF class probabilities so
  the later CL phase can construct `labels` and `pred_probs` without retraining.
- Commit the verified fold input JSONL and manifests. Their expected footprint
  is modest relative to repository limits, and this lets a rented-GPU clone
  train directly without rerunning preparation or transferring an input
  archive. Checkpoints and result artifacts remain ignored.

The existing generic student trainer and evaluator already accept dynamic
training, validation, held-out input, checkpoint, and output paths. The missing
work is therefore data preparation, leakage verification, orchestration,
recovery, and exact-once prediction merging rather than a model rewrite.

The new work is deliberately split into two phases. Phase 4 is CPU-only: it
freezes deterministic `pair_id`-level three-fold partitions, writes `train.jsonl`,
`val.jsonl`, and `test-heldout.jsonl`, proves verification/merge behavior with
fixtures, and commits the verified fold inputs. Phase 5 requires explicit
authorization on a CUDA-capable rented machine: it consumes those committed
inputs, trains a fresh Qwen model per fold, evaluates only that fold's OOF
held-out training pairs, verifies all outputs, and merges exactly 2,500
probability rows.

The official WDC validation/test splits and human-label training targets remain
outside the OOF workflow. Inner validation uses a class-aware 20% of the
remaining two outer folds and majority teacher labels because the current
trainer uses validation for checkpoint selection and early stopping. Evaluator
metrics on `test-heldout.jsonl` therefore describe agreement with the majority
teacher, not gold performance.

`pair_id` is the frozen example unit supplied by the dataset adapter; no extra
entity-level grouping is introduced. Each fold must start from the same
configured base model; the prior full-WDC checkpoint and other folds' states
cannot be reused. The merged artifact freezes `0 = non_match`, `1 = match` and
stores canonical row order, the given majority-label ID, and both OOF class
probabilities so it can later be passed directly to Confident Learning.
