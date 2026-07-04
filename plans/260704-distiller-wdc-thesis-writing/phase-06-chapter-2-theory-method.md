---
phase: 6
title: "Chapter 2 Theory Method"
status: pending
priority: P1
effort: "1 week"
dependencies: [1, 3]
---

# Phase 6: Chapter 2 Theory Method

## Overview

Write the theoretical and methodological foundation that supports the experiment.

## Requirements

- Functional: define ER formulation, pair serialization, compact student training, teacher-label supervision, distillation framing, metrics, and cost metrics.
- Non-functional: keep math light but precise enough for a master's thesis.

## Architecture

Suggested chapter flow:

```text
problem formulation
  -> data representation
  -> supervised student learning
  -> LLM-generated supervision
  -> distillation framing
  -> evaluation metrics
  -> cost metrics
  -> experimental method
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/schema.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/utils/metrics.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/teacher_label_schema.py` after implementation
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/analysis/cost_summary.py` after implementation

## Implementation Steps

1. Define Entity Matching as binary classification over record pairs.
2. Define label notation:
   - `y_gold`.
   - `y_teacher`.
   - `y_student`.
3. Define compact student objective.
4. Define teacher-label supervision:
   - teacher labels are generated offline.
   - student inference does not call teacher.
5. Define metrics:
   - precision.
   - recall.
   - F1.
   - accuracy.
   - invalid-output rate if seq2seq student is used.
6. Define cost metrics:
   - cost per teacher-labeled pair.
   - total labeling cost.
   - cost per trained budget.
7. Define research method:
   - controlled comparison across budgets.
   - same dataset split.
   - same student model.
   - varying supervision source.

## Success Criteria

- [ ] Chapter 2 theory outline exists.
- [ ] Metric formulas are correct.
- [ ] Cost definitions are explicit.
- [ ] Methodology matches Chapter 4 setup.

## Risk Assessment

- Risk: Chapter 2 duplicates Chapter 4.
  Mitigation: Chapter 2 defines concepts; Chapter 4 gives concrete settings.
- Risk: distillation term is challenged.
  Mitigation: explain that this is label-level knowledge distillation / teacher supervision, not necessarily rationale distillation.
