# CLAUDE.md

This file provides guidance for Codex/Claude when working in this repository.
Workspace-level agent guidance also lives in `../AGENTS.md`; read both files
when starting a fresh conversation or changing workflow conventions.

## Active Project Overview

**Master's thesis:** Cost-aware LLM-label distillation for Entity Resolution /
Entity Matching, centered on WDC Products.

**Current research question:**
> Can compact Entity Matching students distilled from LLM-generated teacher
> labels approach gold-label supervised students while being cheaper at
> inference time than using the LLM directly as the matcher?

**Active execution plan:**
`plans/260704-distiller-wdc-agent-execution/plan.md`

**Experiment contract:**
`plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`

**Companion writing plan:**
`plans/260704-distiller-wdc-thesis-writing/plan.md`

## Current Status

The project has pivoted away from the structured-rationale thesis. The old
Phase 03 result is preserved as negative evidence:

| Variant | Train rows | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `label_only` | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Interpretation: structured rationales increased recall but badly hurt
precision, match F1, macro F1, and accuracy. Treat rationale distillation as a
historical/optional negative-history ablation, not the active thesis claim.

The current thesis direction is label-level LLM-to-student distillation:

- `gold_label_student`: compact student trained on dataset labels; quality
  standard, not the cost baseline.
- `direct_llm_matcher`: LLM classifies fixed evaluation pairs directly; repeated
  inference-cost baseline.
- `llm_label_distilled_student`: LLM labels training pairs once, then a compact
  student performs cheap inference; main proposed method.
- `mixed_gold_llm`: optional fallback if pure LLM labels are noisy.

## Build & Run Commands

Always use the project uv-managed virtual environment for Python commands.
Do not run bare system `python`, `pip`, or `pytest` from this repository.
Prefer `.venv/bin/python -m ...` and `uv pip ...` so commands use the pinned
project environment.

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
```

Use the existing 128-row label-only FLAN-T5-base result as historical baseline
context. New target generation should use active `gold_label` naming from
`supervision/build_targets.py`.

## Active Architecture

```text
WDC Products raw/cache data
  -> serialized pair JSONL
  -> low-label budget sampler
  -> direct LLM matcher on fixed evaluation pairs
  -> answer-only teacher LLM labeler
  -> teacher-label validator and cache
  -> target builder: gold_label / llm_label / mixed_gold_llm
  -> compact student training
  -> validation and test evaluation
  -> aggregation, cost table, error analysis
  -> thesis tables and figures
```

Teacher LLM calls are used only for direct-baseline measurement or training-data
creation. Final distilled-student inference must use the compact student without
calling the teacher.

## Immediate Next Steps

Follow the active plan phases:

1. Finish Phase 2 cleanup by keeping guidance aligned with the DistillER/WDC
   contract.
2. Phase 3: implement answer-only direct LLM matching on a fixed validation set
   or predeclared validation sample, with token and cost logging.
3. Phase 3: implement answer-only teacher-label generation for WDC training
   budgets, starting with `train_128`.
4. Phase 4: build `llm_label` and optional `mixed_gold_llm` student targets.
5. Phase 5: train/evaluate the 128-budget pilot against the existing gold-label
   compact-student reference and the direct LLM baseline.

The anti-cherry-pick rule is important: fixed evaluation split/sample, prompt
version, model slug, budgets, and cost fields must be declared before results
are inspected.

## Tech Stack

- Language: Python 3.13
- ML: PyTorch, HuggingFace Transformers
- Data: pandas, datasets
- Validation: Pydantic v2
- First student: `google/flan-t5-base`
- Teacher/direct matcher: OpenRouter-backed LLM calls, answer-only labels first
- Metrics: match precision, match recall, match F1, macro F1, accuracy,
  invalid-output rate, confusion matrix counts, token/cost fields

## Reusable Existing Code

Keep and adapt carefully:

- `data/schema.py`, `data/er_dataset_loader.py`, `data/low_label_sampler.py`,
  and `data/serialize_pairs.py`: WDC pair loading, budgets, and serialization.
- `experiments/train_mt5.py` and `experiments/evaluate_student.py`: reusable
  compact seq2seq student training/evaluation entry points.
- `supervision/build_targets.py`: active gold-label target builder; extend here
  for `llm_label` and `mixed_gold_llm`.
- `models/seq2seq_student.py`: reusable compact seq2seq dataset/model helpers.
- `utils/metrics.py`: binary Entity Matching metrics.

## Removed Historical Code

The branch intentionally removes old Wikidata, multilingual-name, mBART, FEBRL,
and structured-rationale code from the active tree. Keep the research record in
the journals/plans, but do not recreate those packages unless a new experiment
requires them.

## Conventions

- File naming: `snake_case.py`
- Schemas: Pydantic `BaseModel` with explicit fields
- Reproducibility: explicit `seed` parameters, default 42
- Evaluation: train/validation/test split, no test leakage into teacher-label
  generation
- Cost logging: store prompt version, model slug, input tokens, output tokens,
  estimated cost, parsed label, gold label, and validity for LLM calls
- Keep files focused; split large modules when they grow beyond a single
  responsibility

When workflow, status, commands, or conventions change after a meaningful
implementation iteration, update both `../AGENTS.md` and this file if the
change affects future agents.
