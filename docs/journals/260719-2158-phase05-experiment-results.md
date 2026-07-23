---
date: 2026-07-19
session: phase05-experiment-results
---

# Journal: 2026-07-19 — Phase 5 Experiment Results

## Context

This entry consolidates the completed validation-only evidence before the ModernBERT repair run. The fixed test split has not been evaluated. The historical structured-rationale experiment, FLAN-T5-base Phase 5 pilot, and first ModernBERT-base diagnostic are complete; repaired ModernBERT results do not exist yet.

The earlier structured-rationale direction is negative evidence. At roughly the same low-label budget, label-only FLAN-T5 reached match F1 `0.4686`, macro F1 `0.6449`, and accuracy `0.7324`, while structured rationales reached `0.3639`, `0.4931`, and `0.5260`. Rationales raised match recall from `0.5900` to `0.6780` but reduced match precision from `0.3887` to `0.2487`. Source: `AGENTS.md`.

## What Happened

### FLAN-T5-base validation

Source: `outputs/phase05_train_128_results/phase05_results/summary/phase05_train_128.pilot.csv`.

| Arm | Match F1 | Macro F1 | Accuracy | Training seconds | Inference seconds / 2,500 pairs | Teacher/direct cost (USD) |
|---|---:|---:|---:|---:|---:|---:|
| `gold_random_student` | 0.5251 | 0.6779 | 0.7504 | 1,128.81 | 183.89 | 0.000000 |
| `llm_random_student` | 0.5441 | 0.6897 | 0.7580 | 1,139.51 | 183.44 | 0.044698 teacher labels |
| `llm_active_bucketed_v1_student` | 0.5345 | 0.7047 | 0.8028 | 1,121.56 | 183.66 | 0.035554 teacher labels |
| Direct GPT-5.4-mini | 0.8734 | 0.9208 | 0.9492 | — | — | 0.796499 direct validation inference |

Active selection improved macro F1 by `0.0150` and accuracy by `0.0448` over random LLM-label distillation, but reduced the primary match F1 by `0.0096`. It also beat gold-random context by `0.0094` match F1, `0.0268` macro F1, and `0.0524` accuracy. This is a useful secondary-metric signal, but it does not support the primary claim reliably. Direct GPT-5.4-mini remained far stronger than every student.

### First ModernBERT-base validation

Source: `outputs/phase05_modernbert-base_train_128_results/phase05_results/summary/phase05_train_128.pilot.csv`.

| Arm | Match F1 | Macro F1 | Accuracy | Confusion matrix (TP/FP/TN/FN) | Training seconds | Inference seconds / 2,500 pairs |
|---|---:|---:|---:|---:|---:|---:|
| `gold_random_student` | 0.3347 | 0.1782 | 0.2080 | 498 / 1,978 / 22 / 2 | 968.96 | 175.08 |
| `llm_random_student` | 0.1545 | 0.5130 | 0.7768 | 51 / 109 / 1,891 / 449 | 970.30 | 175.16 |
| `llm_active_bucketed_v1_student` | 0.0118 | 0.4500 | 0.7992 | 3 / 5 / 1,995 / 497 | 982.17 | 175.00 |

ModernBERT collapsed toward single-class predictions: gold-random predicted match for 2,476 of 2,500 rows, whereas random and active LLM-label students predicted match for only 160 and 8. Active exceeded random accuracy by `0.0224`, but lost `0.1427` match F1 and `0.0630` macro F1. These values are preserved as negative diagnostic evidence, not treated as a fair verdict on the architecture.

### Timing and cost interpretation

The training and inference seconds above are synchronized measurements on a Tesla T4. FLAN-T5 training took `0.3115–0.3165` GPU-hours and inference took about `0.0734–0.0736` seconds per pair. ModernBERT training took `0.2692–0.2728` GPU-hours and inference took about `0.0700–0.0701` seconds per pair.

Estimated teacher labeling cost was `0.04469775` USD for random selection and `0.03555375` USD for active selection. The fixed direct GPT-5.4-mini validation run had an estimated cost of `0.7964985` USD for 2,500 pairs, or `0.0003185994` USD per pair. These are token-accounted teacher/direct API estimates; they are distinct from student GPU-time estimates.

The `low`, `base`, and `high` cost tables apply analytical rates of `0.25`, `1.00`, and `4.00` USD per GPU-hour to measured seconds. They are sensitivity assumptions, not observed Colab charges or current provider quotes. At the base `1.00` USD/hour assumption:

| Student family | Arm | Upfront cost (USD) | Break-even queries | Total at 2,500 pairs (USD) | Savings vs direct (USD) |
|---|---|---:|---:|---:|---:|
| FLAN-T5 | gold random | 0.3136 | 1,052 | 0.3646 | 0.4319 |
| FLAN-T5 | LLM random | 0.3612 | 1,212 | 0.4122 | 0.3843 |
| FLAN-T5 | LLM active | 0.3471 | 1,165 | 0.3981 | 0.3984 |
| ModernBERT | gold random | 0.2692 | 900 | 0.3178 | 0.4787 |
| ModernBERT | LLM random | 0.3142 | 1,051 | 0.3629 | 0.4336 |
| ModernBERT | LLM active | 0.3084 | 1,031 | 0.3570 | 0.4395 |

Sources: `outputs/phase05_train_128_results/phase05_results/summary/phase05_train_128.cost_scenarios.csv` and `outputs/phase05_modernbert-base_train_128_results/phase05_results/summary/phase05_train_128.cost_scenarios.csv`. Under the low assumption all students saved money at 2,500 comparisons; under the high assumption none did. Break-even and savings therefore depend materially on the declared GPU-hour scenario.

## Reflection

The completed evidence does not yet show that active LLM labeling improves primary match F1 at equal budget. FLAN-T5 offers a promising macro-F1 and accuracy signal, but direct GPT-5.4-mini remains much more accurate and the student result misses the primary metric. The first ModernBERT diagnostic cannot resolve whether this is FLAN-specific because its prediction collapse coincided with confounded low-data training mechanics: incomplete pair preservation, BF16 selection on T4, immediate full-model tuning, small classifier batches, loss-based checkpointing, and a fixed decision threshold.

Cost evidence is nevertheless useful: one-time labeling plus student inference can become cheaper than repeated direct matching, but only after enough queries and only under an explicit GPU-rate assumption. Quality and break-even must therefore be reported together rather than presenting cost savings independently.

## Decisions

| Decision | Rationale |
|---|---|
| Mark both completed Phase 5 student runs **REVISE** | FLAN active misses the primary match-F1 comparison; first ModernBERT collapses toward single-class predictions |
| Preserve structured-rationale and first ModernBERT artifacts as negative evidence | Both failures clarify which supervision and training mechanics are unreliable |
| Keep the fixed test split untouched | Validation has not justified final test evaluation |
| Do not commit to one next student yet | The current model trials are exploratory diagnostics for finding a compact student whose validation quality is close enough to test the thesis question fairly |

## Next Steps

- Try additional compact student models under the same fixed budget-128 targets,
  teacher labels, selection manifests, and validation set.
- Use these trials to identify a student whose quality is close enough to make
  the active-versus-random labeling comparison meaningful; no specific model
  has been selected as the final next step yet.
- Compare candidate students on match F1 first, with macro F1, accuracy,
  training/inference time, and projected cost as required supporting evidence.
- Include the separate `flan-t5-base-full-input` diagnostic with a 2,700-token
  complete-input contract. If its generated decisions remain biased, retain
  validation sequence-likelihood threshold calibration as an optional later
  job rather than part of this controlled rerun.
- Do not call the teacher again or evaluate the fixed test split during model
  screening.
