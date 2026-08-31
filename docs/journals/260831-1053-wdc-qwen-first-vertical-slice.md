---
date: 2026-08-31
session: wdc-qwen-first-vertical-slice
---

# Journal: 2026-08-31 — First WDC–Qwen Vertical Slice

## Context

The first full-label experiment cell pair compares the same Qwen3-Reranker-0.6B configuration trained once on the 2,500 benchmark-gold WDC training targets and once on the 2,500 GPT-5.6 Sol-high hard-label targets. Both arms were evaluated on the same 2,500-row WDC validation split; the test split remained locked.

The downloaded result archive was verified locally at 5,827,366,373 bytes with SHA-256 `aa7f511e3aeb1299f77e71c1356c2f4a8b8a73bbe3ccec1e834619e6cde86de1`.

## What Happened

- Both arms produced 2,500 valid validation predictions with zero invalid outputs.
- Recomputed metrics matched the saved validation metrics.
- The arms disagreed on 86 validation predictions: gold was uniquely correct on 45, LLM-hard was uniquely correct on 41, and both arms were wrong on 155 pairs.

| Validation metric | Gold | LLM-hard | LLM-hard − gold |
|---|---:|---:|---:|
| Match precision | 0.7558922558922558 | 0.7604166666666666 | +0.0045244107744108 |
| Match recall | 0.898 | 0.876 | -0.022 |
| Match F1 | 0.8208409506398537 | 0.8141263940520446 | -0.0067145565878091 |
| Macro F1 | 0.8853308695851598 | 0.881578997229896 | -0.0037518723552638 |
| Accuracy | 0.9216 | 0.92 | -0.0016 |

| Confusion count | Gold | LLM-hard |
|---|---:|---:|
| True positives | 449 | 438 |
| False positives | 145 | 138 |
| True negatives | 1,855 | 1,862 |
| False negatives | 51 | 62 |

| Runtime fact | Gold | LLM-hard |
|---|---:|---:|
| Completed epochs | 9 | 10 |
| Optimizer steps | 1,413 | 1,570 |
| Training wall time | 6,109.6563 s / 1.6971 GPU-h | 6,816.9065 s / 1.8936 GPU-h |
| Validation inference time | 74.6825 s | 74.0636 s |
| Validation throughput | 33.4751 pairs/s | 33.7548 pairs/s |
| Selected decision threshold | 0.013222822919487953 | 0.00026119028916582465 |

Both runs used an NVIDIA GeForce RTX 3090, BF16 precision, PyTorch 2.2.1,
CUDA 12.1, and the same frozen training settings. The gold arm stopped after
epoch 9 under the configured early-stopping rule; LLM-hard completed all 10
epochs.

## Reflection

For this first model–dataset pair, training on LLM-hard labels approached the gold-trained model closely: the LLM-hard arm was lower by 0.67 percentage points in match F1, 0.38 points in macro F1, and 0.16 points in accuracy. It traded slightly higher match precision (+0.45 points) for lower match recall (-2.20 points). The 45-versus-41 unique-correct split also shows that neither arm simply subsumes the other.

These are validation-only results from one predeclared run per arm. They do not establish statistical significance, causality, or performance on the locked test split. The separately selected decision thresholds also mean the comparison describes each arm under its own validation-selected operating point.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Accept the WDC–Qwen gold/LLM-hard pair as the completed first vertical slice | Archive integrity, prediction alignment, and metric recomputation passed | Records 2 of the planned 18 compact-model validation cells without reopening training |
| Retain both model artifacts and compact result metadata outside Git | The archive is large and already checksum-addressed | Git receives only the scientific record, not the 5.5 GiB package or weights |
| Keep the test split locked | The broader 3×3 contract and remaining cells are incomplete | Prevents early test-set feedback from influencing experiment completion |

## Next Steps

- The narrow execution plan is complete and the master plan records 2/18
  validation cells complete.
- Freeze the two remaining datasets, two remaining compact models, and unfinished global contract fields before launching additional paid labeling or GPU cells.
- Continue the remaining matrix under the same one-run-per-cell and validation-only protocol; defer final test evaluation until the global gate is satisfied.
