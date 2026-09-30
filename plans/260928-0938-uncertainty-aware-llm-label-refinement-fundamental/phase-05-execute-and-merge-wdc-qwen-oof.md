---
title: "Phase 5: Execute and Merge WDC Qwen OOF"
status: todo
priority: P1
effort: "3 GPU training runs plus verification"
dependencies: [4]
---

# Phase 5: Execute and Merge WDC Qwen OOF

## Overview

Run the frozen three-fold WDC-Qwen OOF protocol on a CUDA-capable rented GPU,
evaluate only each fold's `test-heldout.jsonl`, and merge the three disjoint
prediction files into one complete 2,500-row probability artifact. This is the
first phase that starts OOF model training.

The merged probabilities are student evidence for later uncertainty detection.
They are not final WDC validation/test results and do not authorize Confident
Learning, ranking integration, prune, relabel, or downstream treatment runs.

## Requirements

- [ ] Phase 4 code, tests, real fold artifacts, hashes, and rendered commands
      have been reviewed and frozen in a clean Git revision.
- [ ] The rented environment has CUDA-compatible PyTorch and can load
      `Qwen/Qwen3-Reranker-0.6B` using the frozen student config.
- [ ] The researcher explicitly authorizes OOF training with
      `--confirm-oof-training`; CPU preflight remains confirmation-free.
- [ ] The runner consumes only the frozen OOF fold directories and never the
      WDC gold target, official validation, or official test paths.

## Output Contract

Use a student-specific ignored output root:

```text
outputs/uncertainty_refinement/wdc-qwen-oof/
  artifact-contract.json
  execution-state.json
  fold_01/
    train/
    test-heldout.predictions.jsonl
    test-heldout.metrics.json
    completion.json
  fold_02/...
  fold_03/...
  oof_predictions.csv
  oof_summary.json
  completion.json
```

Each fold's training directory retains the existing training summary,
checkpoint manifest, and checkpoint needed to resume/verify that fold. The
combined package includes contracts, summaries, manifests, metrics, and
predictions. Model weights are excluded by default because the later detector
uses the saved probabilities; weights may be packaged separately for archival
without being committed to Git.

## Implementation Steps

### 1. Freeze execution identity and run GPU preflight

- Check out the reviewed revision and use its uv-managed environment.
- Verify the committed Phase 4 fold-file hashes, schemas, counts, and
  partitions. Do not regenerate the folds on the rented machine.
- Confirm CUDA visibility, compatible PyTorch, available disk, model access,
  output-root ownership, and absence of conflicting partial outputs.
- Write one artifact contract that binds the Git revision, student config,
  fold-manifest hash, hyperparameters, code entry points, and output paths.
- Record the same execution identity for all three folds; a resumed run must
  match it exactly.

### 2. Train and evaluate fold 1

- Train Qwen with `fold_01/train.jsonl` and `fold_01/val.jsonl` through
  `experiments.train_student` using the frozen WDC-Qwen settings.
- Reload the persisted best checkpoint and evaluate only
  `fold_01/test-heldout.jsonl` through `experiments.evaluate_student`.
- Verify the checkpoint manifest, training summary, expected held-out IDs,
  valid probability rows, and complete atomic outputs before marking fold 1
  complete.
- On interruption, resume/recover the same fold directory; never silently
  start a replacement run with a different contract.

### 3. Repeat for folds 2 and 3

- Execute the identical lifecycle for `fold_02` and then `fold_03`.
- Keep fold outputs isolated and verify each one before proceeding.
- Do not tune hyperparameters, epochs, threshold, or stopping policy based on
  held-out predictions. A failure may be repaired operationally, but a protocol
  change requires a new named run.

### 4. Merge exactly-once OOF predictions

- Re-run Phase 4 data verification before merging.
- Verify each prediction row belongs to the held-out set of its claimed fold
  and that no pair appears in another fold's predictions.
- Require the three-way union to contain exactly 2,500 unique pair IDs and to
  match `fold_assignments.csv` exactly.
- Write `oof_predictions.csv` in frozen blinded-input order and derive
  `assigned_label_probability` from each row's majority label.
- Write `oof_summary.json` with counts, class totals, probability validity,
  fold/output hashes, training-summary hashes, checkpoint-manifest hashes,
  elapsed time, and GPU identity. Do not compare against hidden human labels.

### 5. Verify, package, and hand off

- Independently verify the merged CSV and completion record from current fold
  outputs and the frozen assignment manifest.
- Package the non-weight result evidence and generate a SHA-256 sidecar; verify the
  archive members against the current verified results.
- Keep checkpoint weights outside Git. If the rented machine will be
  terminated, download either the compact evidence package alone or a separate
  optional checkpoint archive before termination.
- Record final counts/hashes and update the plan status. Stop before computing
  uncertainty rankings or using hidden gold.

## Todo

- [ ] Approve the rented-GPU run and execute preflight.
- [ ] Train, reload, evaluate, and verify fold 1.
- [ ] Train, reload, evaluate, and verify fold 2.
- [ ] Train, reload, evaluate, and verify fold 3.
- [ ] Merge and independently verify all 2,500 OOF predictions.
- [ ] Package and download the evidence before terminating the GPU.
- [ ] Record completion evidence and hand off to detector-score planning.

## Success Criteria

- All three fold runs share the frozen execution contract and are individually
  complete and recoverably verified.
- Every WDC training pair has exactly one valid held-out Qwen prediction from a
  model that did not train on or checkpoint-select on that pair.
- `oof_predictions.csv` contains exactly 2,500 unique rows with finite valid
  probabilities and exact agreement with the fold-assignment manifest.
- No WDC human-label training target, official validation file, or official
  test file was read by preparation, training, evaluation, verification, or
  merge code.
- The committed fold inputs are consumed directly after clone/pull; no input
  archive or fold-preparation command is required on the rented machine.
- The verified non-weight result package is downloadable and sufficient for
  the next detector phase; no model weight or generated artifact is committed.
