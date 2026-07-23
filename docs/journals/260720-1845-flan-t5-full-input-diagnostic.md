---
date: 2026-07-20
session: flan-t5-full-input-diagnostic
---

# Journal: 2026-07-20 — FLAN-T5 Full-Input Diagnostic

## Context

The original FLAN-T5-base `train_128` run used a 512-token input cap and returned **REVISE**. Before screening moves on, this diagnostic isolates whether missing tail content limited that result. Measurement with the FLAN-T5 tokenizer found a maximum of 2,490 tokens in the fixed training targets and 2,649 tokens in fixed validation, so the requested 2,400-token cap would still truncate six rows.

## What Happened

- Added `flan-t5-base-full-input` as a separate student configuration using the same public `google/flan-t5-base` weights.
- Set `max_input_length` to 2,700 and disabled input truncation, keeping every currently fixed pair in full.
- Added shared seq2seq training and evaluation enforcement: an over-limit row now fails with its pair identifier and token count instead of being silently cut.
- Preserved the original 512-token FLAN-T5 configuration, results, student ID, and output tree; the diagnostic requires a fresh output root.
- Kept batch size 4 for the long-input seq2seq run and documented A100 as the recommended Colab runtime.
- Recorded sequence-likelihood calibration between `match` and `non-match` as a possible later ablation. It was not implemented in this diagnostic.

## Reflection

This is a controlled model-screening diagnostic, not a change to the thesis claim. It varies FLAN-T5 input coverage while keeping the budget-128 targets, teacher labels, selection manifests, validation rows, seed, generation limits, and direct baseline fixed. Separating the student ID and output root prevents the long-input artifacts from overwriting or being confused with the 512-token baseline.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Use 2,700 rather than 2,400 tokens | The measured maximum is 2,649 tokens | Every fixed training and validation pair fits without truncation |
| Fail on future overflow | Silent truncation would invalidate the full-input condition | Dataset drift becomes explicit before training or evaluation |
| Keep calibration as a note only | The immediate question is input coverage, not a combined scoring change | Any later likelihood-ratio calibration remains a separately declared ablation |
| Recommend A100 with batch 4 | Dense attention over 2,700 tokens is substantially more expensive than the 512-token run | The diagnostic prioritizes reliable execution over aggressive batching |

## Verification

- All 70 repository tests passed.
- Coverage includes config validation, training and evaluation propagation,
  explicit overflow failure, and runner contracts; result packaging remains the
  existing config-driven path.
- No GPU training, teacher call, or test-split evaluation was performed.

## Limitations

- Passing tests establishes workflow correctness, not whether longer inputs improve FLAN-T5 quality.
- The diagnostic still uses the fixed single-seed model-screening setup.

## Next

Run the diagnostic on an A100 under a fresh output root and return its compact archive for comparison with the preserved 512-token FLAN-T5 baseline:

```bash
STUDENT_CONFIG=configs/students/flan_t5_base_full_input.json \
  STUDENT_OUTPUT_ROOT=outputs/students-flan-full-input \
  bash scripts/run_phase05_colab.sh all
```
