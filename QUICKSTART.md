# Quick Start

## Active Direction

This repository now follows the active LLM-labeling WDC thesis pivot:

> selected WDC training pairs -> LLM-generated answer-only labels -> compact ER
> student -> active-vs-random cost/quality comparison against gold-label student
> training and direct LLM matching.

Main execution plan:
`plans/260704-distiller-wdc-agent-execution/plan.md`

Experiment contract:
`plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`

Thesis writing plan:
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The old structured-rationale direction is preserved as negative evidence in
journals and plans. Its code/artifacts have been removed from the active tree.
The plain random LLM-label distillation plan remains the control, but the main
extension is active selection under the same low-label budgets.

The FLAN-T5-base validation pilot and first ModernBERT-base diagnostic are
complete with **REVISE** decisions. The first ModernBERT run collapsed toward a
single class under the 128-row budget; its returned archive is preserved as
negative diagnostic evidence. A repair run keeps every target and validation
row fixed while changing only the classifier training mechanics: pair-aware
longest-first truncation, native-BF16 detection, staged encoder unfreezing,
batch 16, macro-F1 checkpoint selection, and a persisted validation threshold.
The test split remains untouched.

## Environment

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
source .venv/bin/activate
```

Install/update dependencies when needed:

```bash
uv pip install -r requirements.txt
```

Run local checks:

```bash
.venv/bin/python -m unittest discover -s tests
```

Student configurations live in `configs/students/`. Future runs write their
artifacts to `outputs/students/{student_id}/train_{budget}/`; the fixed direct
LLM baseline stays under `outputs/distiller_wdc/direct_llm/`.

For the ModernBERT Colab diagnostic:

```bash
bash scripts/run_phase05_colab.sh setup
STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh all
```

The Colab runtime needs Transformers 4.57 or newer. ModernBERT-base and FLAN-T5
are public and ungated, so neither Hugging Face login nor `HF_TOKEN` is needed.

## Current Work Order

1. Phase 1: completed research contract for cost-aware active LLM labeling.
2. Phase 2: update guidance so old rationale work is clearly historical.
3. Phase 3: implement direct LLM matching, answer-only teacher labeling, and
   fixed selection manifests.
4. Phase 4: build `gold_random`, `llm_random`,
   `llm_active_bucketed_v1`, and optional `mixed_gold_llm_active` targets for
   compact student training.
5. Phase 5: completed FLAN-T5 128-budget validation pilot; decision **REVISE**.
6. Revision diagnostic: rerun ModernBERT-base with the repaired low-data
   classifier training contract across the same three arms.

## Revised Validation Gate

The first new pilot should produce this table shape:

| Arm | Budget / Eval Set | Variant | Match F1 | Macro F1 | Accuracy | LLM cost |
|---|---|---|---:|---:|---:|---:|
| A | train 128, validation | `gold_random` | | | | |
| B | fixed validation eval | `direct_llm_matcher` | | | | |
| C | train 128, validation | `llm_random` | | | | |
| D | train 128, validation | `llm_active_bucketed_v1` | | | | |
| D optional | train 128, validation | `mixed_gold_llm_active` | | | | |

The FLAN-T5 gate produced a **REVISE** decision. Apply the same table to the
ModernBERT classifier diagnostic, then decide whether active selection beats or
usefully diagnoses random LLM labeling at the same budget while preserving a
quality/cost story against repeated direct LLM inference. Do not evaluate test
data during this diagnostic.

## Historical Rationale Result

At budget 128, structured rationales increased recall but hurt precision and
overall F1:

| Variant | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| `label_only` | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Do not revive structured rationales as the main path unless explicitly asked.
