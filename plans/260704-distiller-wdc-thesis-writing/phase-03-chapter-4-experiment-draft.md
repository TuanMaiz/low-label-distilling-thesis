---
phase: 3
title: "Chapter 4 Experiment Draft"
status: pending
priority: P1
effort: "3-5 days"
dependencies: [1]
---

# Phase 3: Chapter 4 Experiment Draft

## Overview

Write the experiment setup before polishing results. This chapter should make the study reproducible.

## Requirements

- Functional: document dataset, splits, label budgets, direct LLM matcher, teacher model, student model, baselines, hyperparameters, hardware/software, and metrics.
- Non-functional: separate setup from results; do not interpret too much here.
- Dependency: needs the experiment contract and pilot run configuration.

## Architecture

Suggested Chapter 4 sections:

```text
dataset
  -> experiment arms
  -> student supervision variants
  -> label budgets
  -> direct LLM matcher setup
  -> teacher setup
  -> student setup
  -> baselines
  -> metrics
  -> environment
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/stats.json`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/low_label/`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/requirements.txt`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/configs/`

## Implementation Steps

1. Write dataset description.
2. Add dataset statistics table.
3. Define train/validation/test split.
4. Define label budgets.
5. Define experiment arms:
   - gold-label compact student.
   - direct LLM matcher.
   - LLM-label distilled compact student.
6. Define student supervision variants:
   - gold labels.
   - LLM-generated labels.
   - mixed labels if used.
7. Define direct LLM matcher setting:
   - model.
   - prompt version.
   - fixed evaluation split or predeclared sample.
   - token/cost logging.
8. Define teacher-labeling setting:
   - model.
   - prompt version.
   - answer-only output.
   - validation rule.
9. Define student setting:
   - model name.
   - max input length.
   - max target length.
   - training epochs.
   - early stopping.
10. Define metrics.
11. Define hardware/software environment.

## Success Criteria

- [ ] Dataset statistics table exists.
- [ ] Experiment matrix table exists.
- [ ] Direct LLM matcher setup is fixed before results.
- [ ] Hyperparameter table exists.
- [ ] Metric definitions are clear.
- [ ] Chapter 4 can reproduce the run setup.

## Risk Assessment

- Risk: setup changes after pilot.
  Mitigation: mark early draft as provisional until full runs finish.
- Risk: cost assumptions are vague.
  Mitigation: define measured API cost separately from estimated human-labeling cost.
