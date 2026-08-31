---
title: "WDC-Qwen Gold vs LLM-Hard Verified Validation Results"
date: 2026-08-31
status: verified
scope: validation-only
---

# WDC-Qwen Gold vs LLM-Hard Verified Validation Results

## Summary

The first full-label vertical slice completed one predeclared RTX 3090 run for
each training-label arm. Both arms produced 2,500 valid predictions on the same
official WDC validation pairs. Independent recomputation from the prediction
JSONL files exactly reproduced every stored classification metric.

The LLM-hard arm remained close to the gold arm, but scored lower on the primary
match F1 by 0.006715 (0.67 percentage points). This is validation-only evidence
from one run per arm, not a final-test, statistical-significance, or causal
claim.

## Verified Package

| Field | Value |
|---|---|
| Local archive | `outputs/new/wdc-qwen-gold-vs-llm-hard.tar.gz` |
| Size | 5,827,366,373 bytes |
| SHA-256 | `aa7f511e3aeb1299f77e71c1356c2f4a8b8a73bbe3ccec1e834619e6cde86de1` |
| Git commit recorded by both arms | `bbbb419c074e6e6b4464f14fd44fbcf63175767e` |
| Test scope | Locked |

The downloaded archive checksum passed locally. Only its small manifests,
summaries, predictions, metrics, thresholds, and completion records were read
for this report. The archive and model weights remain under ignored `outputs/`
and are not repository inputs.

## Validation Results

| Metric | Gold | LLM-hard | LLM-hard − Gold |
|---|---:|---:|---:|
| Match precision | 0.755892 | 0.760417 | +0.004524 |
| Match recall | 0.898000 | 0.876000 | -0.022000 |
| Match F1 | 0.820841 | 0.814126 | -0.006715 |
| Macro F1 | 0.885331 | 0.881579 | -0.003752 |
| Accuracy | 0.921600 | 0.920000 | -0.001600 |
| TP / FP / TN / FN | 449 / 145 / 1855 / 51 | 438 / 138 / 1862 / 62 | — |
| Invalid predictions | 0 | 0 | 0 |
| Selected threshold | 0.013222823 | 0.000261190 | — |

Thresholds were selected independently on validation under the frozen policy;
their numerical difference is descriptive and did not change the training
inputs or unlock the test split.

## Paired Prediction Comparison

- Both files contain exactly 2,500 unique IDs in identical order with identical
  evaluation labels.
- The arms disagree on 86 predictions.
- Gold alone is correct on 45 pairs; LLM-hard alone is correct on 41 pairs.
- Both arms are wrong on 155 pairs.

## Runtime Record

| Field | Gold | LLM-hard |
|---|---:|---:|
| Completed epochs | 9 | 10 |
| Optimizer steps | 1,413 | 1,570 |
| Training wall time | 6,109.66 s | 6,816.91 s |
| Training GPU-hours | 1.6971 | 1.8936 |
| Inference time | 74.68 s | 74.06 s |
| Inference throughput | 33.48 pairs/s | 33.75 pairs/s |
| Evaluation wall time | 77.92 s | 77.33 s |

Both arms record `NVIDIA GeForce RTX 3090`, BF16, PyTorch 2.2.1, CUDA 12.1,
and Transformers 4.57.6. Gold stopped after epoch 9 under the frozen early
stopping rule; LLM-hard completed all 10 epochs.

## Verification Performed

- Archive checksum matched the downloaded checksum record.
- Each arm contained 2,500 unique, valid validation predictions.
- Pair IDs, order, and evaluation labels aligned across arms.
- Precision, recall, F1, macro metrics, accuracy, and confusion counts were
  independently recomputed from predictions and matched stored values exactly.
- Both artifact contracts recorded the same frozen code revision, dataset,
  model, schedule, warmup, and locked test scope; only the authorized arm and
  label-source-specific artifacts differed.

## Remaining Work

This completes 2 of the 18 compact-model validation cells in the global matrix.
The other two datasets, other two compact models, remaining 16 validation cells,
three direct-LLM baselines, aggregate cost analysis, and any separately
authorized final-test evaluation remain unfinished.
