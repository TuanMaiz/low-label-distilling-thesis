# Quick Start

## Active Direction

This repository now follows the reasoning-rationale distillation thesis pivot:

> validated structured rationales from a reasoning LLM -> compact seq2seq ER student -> low-label label-efficiency evaluation.

Main plan:
`../plans/260619-reasoning-rationale-distillation/plan.md`

Cleanup inventory:
`../docs/codebase-pivot-cleanup-inventory.md`

## Environment

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
source .venv/bin/activate
```

Install/update dependencies when needed:

```bash
uv pip install -r requirements.txt
```

## Current Work Order

1. Phase 00: completed non-destructive pivot cleanup.
2. Phase 01: implement dataset loader, serializer, and low-label sampler.
3. Phase 02: implement structured rationale schema, teacher prompt, and validator.
4. Phase 03: run minimal mT5-small pilot.

Phase 01 dataset prep now lives under `data/` and `experiments/`. Upcoming
Phase 02-03 work should add rationale and student-training entrypoints under
`rationales/`, `models/`, and `experiments/`.

## First Decision Gate

By the end of Phase 03, produce this table:

| Budget | Label-only mT5 F1 | Structured-rationale mT5 F1 | Difference |
|---:|---:|---:|---:|
| 16 | | | |
| 32 | | | |
| 64 | | | |
| 128 | | | |

Continue only if structured rationales show useful signal, especially at 16/32/64 labels.

## Legacy Warning

These old commands were moved to `legacy/multilingual_name/main.py` and are
kept only as historical reference:

```bash
python legacy/multilingual_name/main.py --mode test
python legacy/multilingual_name/main.py --mode train --model mbart
```

They belong to the older Wikidata/mBART direction. Do not use them as the
current workflow.
