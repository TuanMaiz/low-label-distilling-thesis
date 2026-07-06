---
phase: 2
title: "Chapter 3 System Draft"
status: pending
priority: P1
effort: "3-5 days"
dependencies: [1]
---

# Phase 2: Chapter 3 System Draft

## Overview

Draft the system/pipeline chapter while implementation details are fresh.

## Requirements

- Functional: describe the implemented data, teacher-label, target-building, training, evaluation, and cost-logging pipeline.
- Non-functional: keep this chapter descriptive and reproducible, not argumentative.
- Dependency: needs agent execution Phases 3-4 for final file paths and commands.

## Architecture

Suggested Chapter 3 flow:

```text
system overview
  -> WDC data preparation
  -> low-label sampler
  -> teacher label generation
  -> label validation/cache
  -> target builder
  -> student training
  -> evaluation and cost logging
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/er_dataset_loader.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/low_label_sampler.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/serialize_pairs.py`
- Read after implementation: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/generate_teacher_labels.py`
- Read after implementation: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/build_targets.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/train_mt5.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/evaluate_student.py`

## Implementation Steps

1. Draft Chapter 3 outline.
2. Create Figure 3.1: overall pipeline.
3. Create Figure 3.2: teacher-label cache and target builder flow.
4. Write dataset preparation section.
5. Write teacher-labeling module section.
6. Write student-training module section.
7. Write evaluation and cost-logging module section.
8. Add command snippets only if the school format allows; otherwise put commands in appendix.

## Success Criteria

- [ ] Chapter 3 skeleton exists.
- [ ] At least two system figures are planned.
- [ ] Each pipeline module maps to code paths.
- [ ] Chapter 3 can be read without knowing the codebase.

## Risk Assessment

- Risk: chapter becomes code documentation.
  Mitigation: explain workflow and data artifacts, not every function.
- Risk: figures are missing.
  Mitigation: reserve figure numbers early.
