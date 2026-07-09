---
phase: 6
title: "Full Budget Study"
status: pending
priority: P1
effort: "2-3 weeks"
dependencies: [5]
---

# Phase 6: Full Budget Study

## Overview

Scale the pilot into thesis-grade evidence across low-label budgets, comparing
random and active LLM-label selection for label efficiency and cost efficiency.

## Requirements

- Functional: run accepted random and active variants across `16`, `32`, `64`, `128`, and `256`.
- Functional: aggregate results into tables and plots.
- Functional: carry the direct LLM matcher as a fixed horizontal reference for quality and inference cost.
- Non-functional: use consistent validation/test splits and documented seeds.
- Statistical caution: repeat seeds where feasible, but do not block the thesis on expensive repetitions.

## Architecture

```text
budget x selection strategy x supervision variant
  -> training jobs
  -> validation/test metrics
direct LLM matcher
  -> fixed quality/cost reference
all results
  -> aggregated CSV
  -> label-efficiency curve
  -> cost-efficiency comparison
```

## Related Code Files

- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/run_low_label_study.py`
- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/aggregate_results.py`
- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/analysis/plot_label_efficiency.py`
- Outputs: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/summary/`
- Figures: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/figures/`

## Implementation Steps

1. Use Phase 5 decision to choose variants:
   - always include `gold_random` as the trusted low-budget quality context.
   - always include `llm_random` as the random distillation control.
   - include the best active strategy, expected first candidate `llm_active_bucketed_v1`.
   - include `mixed_gold_llm_active` if pilot suggests it stabilizes results.
2. Generate or verify targets for all budgets.
3. Run training for each budget and variant.
4. Save one metrics JSON per run.
5. Aggregate metrics into CSV:
   - budget.
   - variant.
   - selection strategy.
   - experiment arm.
   - seed.
   - precision.
   - recall.
   - F1.
   - accuracy.
   - invalid-output rate.
   - cost.
6. Add direct LLM matcher as a fixed reference row or horizontal line.
7. Plot label-efficiency curves for random and active LLM-label selection.
8. Plot or table active-vs-random gain at the same budget.
9. Produce a thesis-ready table showing performance and cost side by side.

## Success Criteria

- [ ] Full budget table exists.
- [ ] Label-efficiency plot exists.
- [ ] Active-vs-random same-budget comparison exists.
- [ ] Cost-efficiency table exists.
- [ ] Direct LLM matcher appears as the inference-cost baseline.
- [ ] Results are stable enough to write Chapter 5.

## Risk Assessment

- Risk: training all combinations is too slow.
  Mitigation: prioritize `128`, `256`, then smaller budgets; skip weak variants.
- Risk: low-budget curves are jagged.
  Mitigation: use multiple seeds for the most important budgets if possible.
- Risk: one active strategy is overfit to WDC quirks.
  Mitigation: present it as WDC-focused evidence and use optional external-validity only after core runs finish.
