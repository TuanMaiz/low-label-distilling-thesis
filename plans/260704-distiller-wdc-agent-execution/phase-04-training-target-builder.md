---
phase: 4
title: "Training Target Builder"
status: pending
priority: P1
effort: "2-3 days"
dependencies: [3]
---

# Phase 4: Training Target Builder

## Overview

Extend target generation so the same WDC pairs can train students under gold labels, LLM labels, and mixed supervision.

## Requirements

- Functional: build compact student targets from validated teacher labels.
- Functional: keep validation and test labels gold-only.
- Non-functional: target files must preserve pair IDs for downstream error analysis.
- Reproducibility: output metadata must identify supervision source and teacher prompt version.

## Architecture

```text
low-label train pairs
  + teacher-label cache
  -> target builder
  -> gold_label targets
  -> llm_label targets
  -> mixed_gold_llm targets
```

## Related Code Files

- Modify: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/build_targets.py`
- Modify tests: `/mnt/d/Study/Cao-hoc/luan-van/code/tests/`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/low_label/`
- Write outputs under: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/targets/`

## Implementation Steps

1. Add target variant `gold_label` if the existing `label_only` naming is too tied to old experiments.
2. Add target variant `llm_label`:
   - join pair rows with teacher-label cache by `pair_id`.
   - drop invalid teacher labels.
   - log dropped rows.
3. Add target variant `mixed_gold_llm`:
   - use small gold seed for a subset.
   - use teacher labels for the remainder.
   - record the source per row.
4. Ensure target JSONL rows include:
   - pair ID.
   - serialized input.
   - target label.
   - supervision source.
   - teacher metadata if applicable.
5. Add target validation:
   - no duplicate pair IDs.
   - no missing target.
   - expected label vocabulary only.
   - train/validation/test split separation.
6. Generate pilot target files for `128` and `256`.

## Success Criteria

- [ ] `train_128.gold_label.targets.jsonl` exists or equivalent old label-only target is mapped.
- [ ] `train_128.llm_label.targets.jsonl` exists.
- [ ] `train_256.llm_label.targets.jsonl` exists.
- [ ] Validation/test targets use gold labels only.
- [ ] Target builder tests pass.

## Risk Assessment

- Risk: mixed variant creates hidden leakage.
  Mitigation: only mix within training split; never use validation/test labels for teacher training.
- Risk: teacher-label cache has fewer valid rows than expected.
  Mitigation: write actual row counts into target metadata and thesis tables.
