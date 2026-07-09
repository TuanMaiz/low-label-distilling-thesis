---
phase: 4
title: "Training Target Builder"
status: completed
priority: P1
effort: "2-3 days"
dependencies: [3]
---

# Phase 4: Training Target Builder

## Overview

Extend target generation so WDC pairs can train students under gold labels,
random LLM labels, actively selected LLM labels, and mixed supervision.

## Requirements

- Functional: build compact student targets from validated teacher labels.
- Functional: preserve selection strategy metadata for random and active LLM-label variants.
- Functional: keep validation and test labels gold-only.
- Non-functional: target files must preserve pair IDs for downstream error analysis.
- Reproducibility: output metadata must identify supervision source, selection strategy, and teacher prompt version.

## Architecture

```text
low-label train pairs
  + selection manifest
  + teacher-label cache
  -> target builder
  -> gold_random targets
  -> llm_random targets
  -> llm_active_* targets
  -> mixed_gold_llm_active targets
```

## Related Code Files

- Modify: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/build_targets.py`
- Modify tests: `/mnt/d/Study/Cao-hoc/luan-van/code/tests/`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/low_label/`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/selection/`
- Write outputs under: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/targets/`

## Implementation Steps

1. Add target variant `gold_random` if the existing `label_only` naming is too tied to old experiments.
2. Add target variant `llm_random`:
   - join random selected pair rows with teacher-label cache by `pair_id`.
   - drop invalid teacher labels.
   - log dropped rows.
3. Add target variant `llm_active_bucketed_v1`:
   - read the fixed active-selection manifest.
   - join active selected pair rows with teacher-label cache by `pair_id`.
   - preserve selection bucket, bucket rank/quota, rank, and score fields.
4. Add optional variants `llm_active_uncertainty`, `llm_active_diversity`, and `llm_active_hard_negative` only if Phase 5 needs them.
5. Add target variant `mixed_gold_llm_active`:
   - use small gold seed for a subset.
   - use teacher labels for the remainder.
   - record the source per row.
6. Ensure target JSONL rows include:
   - pair ID.
   - serialized input.
   - target label.
   - supervision source.
   - selection strategy.
   - selection rank and score if applicable.
   - teacher metadata if applicable.
7. Add target validation:
   - no duplicate pair IDs.
   - no missing target.
   - expected label vocabulary only.
   - train/validation/test split separation.
8. Generate pilot target files for `128.random`, `128.llm_random`, and `128.llm_active_bucketed_v1`.
9. Generate `256` targets only after the fixed `256` manifests exist.

## Implementation Status

Updated 2026-07-08:

- Extended `supervision/build_targets.py` with `gold_random`, `llm_random`,
  and `llm_active_bucketed_v1` variants.
- LLM-label targets join selected-pair manifests with validated teacher-label
  caches by `pair_id`.
- LLM target rows use teacher `label` for `target_text` and preserve
  `gold_label` only for audit.
- Active target rows preserve `selection_bucket`, bucket rank/quota, selection
  rank, score, seed, teacher model, prompt version, token counts, and cost.
- Added `scripts/run_phase04_targets.sh` for reproducible target generation.
- Generated the 128-budget pilot targets:
  - `data/cache/wdc_products/targets/train_128.gold_random.targets.jsonl`
  - `data/cache/wdc_products/targets/train_128.llm_random.targets.jsonl`
  - `data/cache/wdc_products/targets/train_128.llm_active_bucketed_v1.targets.jsonl`
- Optional `mixed_gold_llm_active` and 256-budget targets are deferred until the
  128 pilot result justifies them.

## Success Criteria

- [x] `train_128.gold_random.targets.jsonl` exists or equivalent old label-only target is mapped.
- [x] `train_128.llm_random.targets.jsonl` exists.
- [x] `train_128.llm_active_bucketed_v1.targets.jsonl` exists if active pilot is approved.
- [ ] `train_256.llm_random.targets.jsonl` exists if the `256` manifest is created.
- [ ] `train_256.llm_active_bucketed_v1.targets.jsonl` exists if active pilot expands.
- [x] Validation/test targets use gold labels only.
- [x] Target builder tests pass.

## Risk Assessment

- Risk: mixed variant creates hidden leakage.
  Mitigation: only mix within training split; never use validation/test labels for teacher training.
- Risk: teacher-label cache has fewer valid rows than expected.
  Mitigation: write actual row counts into target metadata and thesis tables.
- Risk: selection metadata is lost before failure analysis.
  Mitigation: require selection strategy, bucket, rank, and score fields in each selected target row.
