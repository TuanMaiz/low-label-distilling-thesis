# Quick Start

## Active Direction

This repository now follows the DistillER/WDC thesis pivot:

> LLM-generated answer-only labels -> compact ER student -> cost and quality
> comparison against gold-label student training and direct LLM matching.

Main execution plan:
`plans/260704-distiller-wdc-agent-execution/plan.md`

Experiment contract:
`plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`

Thesis writing plan:
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The old structured-rationale direction is preserved as negative evidence in
journals and plans. Its code/artifacts have been removed from the active tree.

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

1. Phase 1: completed research contract for cost-aware LLM-label distillation.
2. Phase 2: update guidance so old rationale work is clearly historical.
3. Phase 3: implement direct LLM matching and answer-only teacher labeling.
4. Phase 4: build `gold_label`, `llm_label`, and optional `mixed_gold_llm`
   targets for compact student training.
5. Phase 5: run the 128-budget pilot and compare against direct LLM cost.

## First Decision Gate

The first new pilot should produce this table shape:

| Arm | Budget / Eval Set | Variant | Match F1 | Macro F1 | Accuracy | LLM cost |
|---|---|---|---:|---:|---:|---:|
| A | train 128, validation | `gold_label` | | | | |
| B | fixed validation eval | `direct_llm_matcher` | | | | |
| C | train 128, validation | `llm_label` | | | | |
| C optional | train 128, validation | `mixed_gold_llm` | | | | |

Continue only if the LLM-label student has a useful quality/cost story compared
with the gold-label reference and repeated direct LLM inference.

## Historical Rationale Result

At budget 128, structured rationales increased recall but hurt precision and
overall F1:

| Variant | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| `label_only` | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Do not revive structured rationales as the main path unless explicitly asked.
