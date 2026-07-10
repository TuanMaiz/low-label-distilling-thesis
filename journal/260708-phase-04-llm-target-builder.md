# Phase 4 LLM Target Builder

Date: 2026-07-08
Plan: `plans/260704-distiller-wdc-agent-execution/plan.md`
Tags: `wdc-products`, `target-builder`, `llm-labeling`, `phase-04`

## Context

Phase 3 produced live OpenRouter teacher-label caches for the fixed random and
bucketed active 128-row manifests. The next step was to turn those labels into
compact-student training targets.

## What Changed

- Extended `supervision/build_targets.py` with:
  - `gold_random`.
  - `llm_random`.
  - `llm_active_bucketed_v1`.
- LLM variants join a selected-pair manifest with a valid teacher-label cache
  by `pair_id`.
- LLM target rows use teacher `label` as the student `target_text`.
- Gold labels are preserved as `gold_label` only for audit and later
  teacher-noise analysis.
- Active target rows preserve bucket metadata and cost metadata.
- Added `scripts/run_phase04_targets.sh` for reproducible target generation.

## Generated Targets

- `data/cache/wdc_products/targets/train_128.gold_random.targets.jsonl`
- `data/cache/wdc_products/targets/train_128.llm_random.targets.jsonl`
- `data/cache/wdc_products/targets/train_128.llm_active_bucketed_v1.targets.jsonl`

## Verification

- `scripts/run_phase04_targets.sh local`: passed.
- `.venv/bin/python -m unittest discover -s tests`: 20 tests passed.
- Target row counts: 128 rows for each generated target file.

## Notes

The earlier sandbox DNS-failure reject rows were deleted after the successful
OpenRouter run. The `teacher_labels/` directory now keeps only the successful
`.labels.jsonl` caches. The successful label caches validate cleanly and were
used for target generation.

## Superseded Artifact Note

Updated 2026-07-10: the model-less LLM target files listed above were
superseded after switching the Phase 3 teacher/direct model to
`openai/gpt-5.4-mini`. Phase 5 should use:

- `data/cache/wdc_products/targets/train_128.llm_random.openai-gpt-5-4-mini.targets.jsonl`
- `data/cache/wdc_products/targets/train_128.llm_active_bucketed_v1.openai-gpt-5-4-mini.targets.jsonl`

The old model-less LLM target files were removed to avoid accidental training
from stale GPT-4o-mini teacher labels.
