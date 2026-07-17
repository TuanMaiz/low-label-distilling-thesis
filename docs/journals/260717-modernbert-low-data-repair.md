---
date: 2026-07-17
session: modernbert-low-data-repair
---

# Journal: 2026-07-17 — ModernBERT Low-Data Repair

## Context

The first ModernBERT-base `train_128` validation run returned **REVISE** after collapsing toward single-class predictions. Of 2,500 validation pairs, `gold_random` predicted 2,476 matches, while `llm_random` and `llm_active_bucketed_v1` predicted only 160 and 8. Their match F1 scores were 0.3347, 0.1545, and 0.0118 respectively. The original archive remains negative diagnostic evidence.

The repair keeps the model, seed, budget, targets, teacher labels, selection manifests, and validation rows unchanged. No new teacher calls are allowed, and the fixed test split remains untouched.

## What Happened

- Changed classifier tokenization to pass records A and B as one complete pair, raised the fixed-input cap to 2,400 after measuring a 2,334-token maximum, and disabled truncation so future overflow fails explicitly.
- Restricted automatic BF16 to native Ampere-or-newer support, making Tesla T4 resolve to FP16.
- Raised the classifier batch to 16 and added 10% warmup.
- Added staged tuning: train the classification head alone for two epochs, then unfreeze the final four encoder blocks using separate head and encoder learning rates of `1e-3` and `1e-5`.
- Changed classifier checkpoint selection from validation loss to validation macro F1, with match F1 as the tie-breaker.
- Selected a decision threshold on validation macro F1, persisted it with the best checkpoint and run artifacts, and required evaluation and packaging to reuse it.
- Preserved seq2seq behavior and recorded the new classifier settings in artifact contracts and summaries.

## Reflection

The first run mixed an architecture diagnostic with avoidable low-data training confounds: one-sided truncation, BF16 on T4, immediate full-model tuning with batch 4, loss-only checkpointing, and a fixed 0.5 threshold. This repair isolates those mechanics without changing the experimental data or inspecting test results. It is a declared repair run, not evidence that ModernBERT now succeeds.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Preserve the failed archive and use a new output root | Avoid overwriting negative evidence or reusing incompatible contracts | The repaired run remains auditable against the first run |
| Tune the head before the final encoder blocks | Reduce destructive updates under 128-row supervision | The classifier receives a stable warm-up before limited representation adaptation |
| Select checkpoint and threshold on validation macro F1 | Loss and a fixed 0.5 cutoff did not protect against single-class collapse | Model selection directly reflects balanced classification quality without touching test |

## Verification

- Focused repair checks and the full repository test suite passed.
- The Colab runner passed Bash syntax validation.
- Contract smoke checks covered the new training settings, T4 FP16 resolution, and threshold artifacts.

## Limitations

- No repaired ModernBERT training or validation was performed locally.
- The repair still uses one seed and cannot establish robustness until the predeclared validation run is reviewed.

## Next Steps

Run the repair on Colab under a fresh output root:

```bash
bash scripts/run_phase05_colab.sh setup
STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh all
```

Return `outputs/students-modernbert-repair/modernbert-base/artifacts/phase05_modernbert-base_train_128_results.tar.gz` for comparison with the failed first run, FLAN-T5 pilot, and fixed direct LLM baseline.
