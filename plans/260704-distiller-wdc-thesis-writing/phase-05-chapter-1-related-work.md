---
phase: 5
title: "Chapter 1 Related Work"
status: pending
priority: P1
effort: "1 week"
dependencies: [1]
---

# Phase 5: Chapter 1 Related Work

## Overview

Write `Chuong 1: Tong quan` as the related-work and research-gap chapter, not as a generic introduction.

## Requirements

- Functional: cover ER/EM, product matching, WDC, transformer ER, active learning/data selection for ER, LLMs for ER, DistillER, and LLM-generated labels.
- Non-functional: clearly admit related work overlap and position the thesis as controlled evaluation/adaptation.

## Architecture

Suggested chapter flow:

```text
ER/EM background
  -> product matching and WDC
  -> neural/transformer ER
  -> active learning and data selection for ER
  -> LLMs for ER
  -> distillation and LLM supervision
  -> DistillER and close work
  -> remaining gap
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/peeters-2023-entity-matching-using-large-language-models.pdf`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/steiner-2024-fine-tuning-large-language-models-for-entity-matching.pdf`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/wadhwa-2024-learning-from-natural-language-explanations-for-generalizable-entity-matching.pdf`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/zeakis-2026-distiller-knowledge-distillation-in-entity-resolution-with-large-language-models.pdf`

## Implementation Steps

1. Build citation table with columns:
   - paper.
   - task.
   - dataset.
   - teacher/student model.
   - relation to this thesis.
2. Write ER/EM overview.
3. Write product matching and WDC overview.
4. Write neural ER overview.
5. Write active learning and data selection for ER overview.
6. Write LLM for ER overview.
7. Write DistillER and LLM-labeled training data subsection.
8. Write gap:
   - need a WDC-focused, cost-aware compact-student study.
   - need explicit active-vs-random LLM-label selection under fixed budgets.
   - need explicit label-budget, teacher-noise, and failure behavior analysis.
9. Keep any stronger novelty language out unless advisor approves it.

## Success Criteria

- [ ] Chapter 1 outline exists.
- [ ] Related-work comparison table exists.
- [ ] DistillER overlap is acknowledged.
- [ ] Gap statement is narrow and defensible.

## Risk Assessment

- Risk: related work makes thesis look too close.
  Mitigation: frame as reproduction/adaptation/evaluation with WDC and cost, suitable for master's thesis.
- Risk: chapter becomes a literature dump.
  Mitigation: end every subsection with why it matters for this thesis.
