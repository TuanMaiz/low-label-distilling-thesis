---
phase: 1
title: "Thesis Contract"
status: pending
priority: P1
effort: "1 day"
dependencies: []
---

# Phase 1: Thesis Contract

## Overview

Freeze the thesis title, safe claim, chapter structure, contribution wording, and evidence requirements before writing long prose.

## Requirements

- Functional: define working Vietnamese and English titles.
- Functional: map the topic into the official thesis structure.
- Non-functional: avoid overclaiming novelty against DistillER and related LLM-labeling work.

## Architecture

This phase creates the writing contract:

```text
research evidence needed
  -> official chapter structure
  -> contribution wording
  -> figure/table plan
  -> writing order
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/plan.md`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/zeakis-2026-distiller-knowledge-distillation-in-entity-resolution-with-large-language-models.pdf`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/steiner-2024-fine-tuning-large-language-models-for-entity-matching.pdf`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/research/thesis-contract.md`

## Implementation Steps

1. Choose working title:
   - English: "Cost-Aware Distillation of LLM-Generated Labels for Entity Matching".
   - Vietnamese can be finalized with advisor.
2. Define one safe research question.
3. Define 3-4 thesis contributions:
   - WDC-focused evaluation of LLM-to-student label distillation for compact ER students.
   - comparison against direct LLM matching as the inference-cost baseline.
   - label-budget, cost-efficiency, and break-even analysis.
   - failure analysis of direct LLM, teacher-label, and student errors.
4. Define what evidence each contribution needs.
5. Define chapter structure using official requirement:
   - front matter.
   - `Mo dau`.
   - `Chuong 1: Tong quan`.
   - `Chuong 2: Co so ly thuyet va phuong phap nghien cuu`.
   - `Chuong 3: Xay dung he thong`.
   - `Chuong 4: Thuc nghiem`.
   - `Chuong 5: Ket qua va ban luan`.
   - `Chuong 6: Ket luan va khuyen nghi`.
6. Define figure/table list needed before writing.

## Success Criteria

- [ ] Working title exists.
- [ ] Thesis claim is safe.
- [ ] Chapter outline follows official structure.
- [ ] Evidence checklist exists for each contribution.

## Risk Assessment

- Risk: writing starts before claim is known.
  Mitigation: write only skeleton prose until pilot results exist.
- Risk: advisor expects more novelty.
  Mitigation: keep optional expansion ready: second student or extra dataset.
