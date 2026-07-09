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

## Current Work Order

1. Phase 1: completed research contract for cost-aware active LLM labeling.
2. Phase 2: update guidance so old rationale work is clearly historical.
3. Phase 3: implement direct LLM matching, answer-only teacher labeling, and
   fixed selection manifests.
4. Phase 4: build `gold_random`, `llm_random`, `llm_active_hybrid`, and optional
   `mixed_gold_llm_active` targets for compact student training.
5. Phase 5: run the 128-budget pilot and compare active selection against
   random LLM labels and direct LLM cost.

## First Decision Gate

The first new pilot should produce this table shape:

| Arm | Budget / Eval Set | Variant | Match F1 | Macro F1 | Accuracy | LLM cost |
|---|---|---|---:|---:|---:|---:|
| A | train 128, validation | `gold_random` | | | | |
| B | fixed validation eval | `direct_llm_matcher` | | | | |
| C | train 128, validation | `llm_random` | | | | |
| D | train 128, validation | `llm_active_hybrid` | | | | |
| D optional | train 128, validation | `mixed_gold_llm_active` | | | | |

Continue only if the active LLM-label student beats or usefully diagnoses the
random LLM-label student at the same budget, while preserving a quality/cost
story against repeated direct LLM inference.

## Historical Rationale Result

At budget 128, structured rationales increased recall but hurt precision and
overall F1:

| Variant | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| `label_only` | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Do not revive structured rationales as the main path unless explicitly asked.
