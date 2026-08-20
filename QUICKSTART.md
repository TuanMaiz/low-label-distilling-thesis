# Quick Start

## Active Direction

The repository is migrating to a full-label Entity Resolution experiment:

`3 datasets × 3 compact cross-encoder models × {gold, LLM hard labels}`

The direct LLM matcher remains a per-dataset accuracy and cost baseline. The
retired low-label, active-selection, and Phase-05 workflows are available only
through Git history and must not be run.

Active plan:
`plans/260820-1507-full-label-er-migration/plan.md`

Legacy writing plan (revise before thesis drafting):
`plans/260704-distiller-wdc-thesis-writing/plan.md`

## Environment

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
```

## Current Work Order

Start with Phase 1 of the active plan. Freeze the exact datasets, models, LLM
labeler, prompt, evaluation scope, artifact schema, and cost ceiling before
implementing new loaders or making paid LLM-labeling calls.

There is no active production experiment runner during this migration. Phase 5
will introduce `scripts/run_full_label_experiments.sh` after the contract,
datasets, targets, and compact models are verified.
