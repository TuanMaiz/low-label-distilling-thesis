---
title: "DistillER WDC Thesis Writing"
description: "Detailed writing plan for the official master's thesis structure, written around the DistillER-style WDC experiment after results are known."
status: pending
priority: P1
effort: "8-10 weeks alongside experiments"
branch: "phase-03/minimal-distillation-pilot"
tags: [thesis, writing, entity-resolution, llm-labeling, wdc]
blockedBy: []
blocks: []
created: "2026-07-04T08:00:47.237Z"
createdBy: "ck:plan"
source: skill
---

# DistillER WDC Thesis Writing

## Overview

This plan turns the official school requirement into a practical writing schedule for the DistillER-style WDC thesis pivot.

Write like a paper, but format like the official thesis:

1. Run and understand experiments first.
2. Draft system and experiment chapters while implementation details are fresh.
3. Write results before finalizing the introduction.
4. Write `Mo dau` and conclusion last, so the claims match the evidence.

## Official Structure

The official requirement uses:

- Front matter.
- `Mo dau`.
- `Chuong 1: Tong quan`.
- Main research chapters.
- Final chapter: conclusion and recommendations.
- Publications, references, appendices.

So the thesis should not treat Chapter 1 as a generic introduction. Introduction-like content belongs in `Mo dau`; Chapter 1 should be the related-work and gap chapter.

## Thesis Working Claim

Conservative claim:

> This thesis evaluates whether compact Entity Matching students distilled from LLM-generated teacher labels can approach gold-label supervised students on WDC Products while reducing repeated inference cost compared with direct LLM matching.

Do not claim:

- that LLM-label distillation for ER is new.
- that the method beats DistillER generally.
- that results transfer to all ER datasets unless experiments show it.

## Formatting Rules To Preserve

| Item | Requirement |
|---|---|
| Body font | Times New Roman 13 |
| Chapter/section title font | Times New Roman 14 or 15 |
| Paper | A4, one-sided |
| Line spacing | 1.5 |
| Margins | top 2.5 cm, bottom 2.5 cm, left 3.5 cm, right 2 cm |
| Page number | centered bottom |
| Table title | above table, e.g. `Bang 3.2. ...` |
| Figure title | below figure, e.g. `Hinh 4.5. ...` |
| Equation number | by chapter, e.g. `(3.5)` |
| Citations | square brackets, e.g. `[5]`, `[9-12]` |

## Phases

| Phase | Name | Status | Priority | Effort | Dependencies |
|-------|------|--------|----------|--------|--------------|
| 1 | [Thesis Contract](./phase-01-thesis-contract.md) | Pending | P1 | 1 day | None |
| 2 | [Chapter 3 System Draft](./phase-02-chapter-3-system-draft.md) | Pending | P1 | 3-5 days | Phase 1, agent Phases 3-4 |
| 3 | [Chapter 4 Experiment Draft](./phase-03-chapter-4-experiment-draft.md) | Pending | P1 | 3-5 days | Phase 1, agent Phase 5 |
| 4 | [Chapter 5 Results Draft](./phase-04-chapter-5-results-draft.md) | Pending | P1 | 1-2 weeks | agent Phases 5-7 |
| 5 | [Chapter 1 Related Work](./phase-05-chapter-1-related-work.md) | Pending | P1 | 1 week | Phase 1 |
| 6 | [Chapter 2 Theory Method](./phase-06-chapter-2-theory-method.md) | Pending | P1 | 1 week | Phase 1, Phase 3 |
| 7 | [Mo Dau And Conclusion](./phase-07-mo-dau-and-conclusion.md) | Pending | P1 | 3-5 days | Phase 4 |
| 8 | [Formatting Submission Package](./phase-08-formatting-submission-package.md) | Pending | P1 | 1 week | Phases 2-7 |

## Writing Order

Recommended order:

1. Phase 1: freeze thesis contract.
2. Phase 2 and Phase 3: draft system and experiment setup early.
3. Phase 4: write results as soon as experiment tables exist.
4. Phase 5 and Phase 6: write related work and theory around the final claim.
5. Phase 7: write `Mo dau` and conclusion last.
6. Phase 8: format, references, lists, appendices.

## Relationship To Agent Execution Plan

The companion execution plan is:

`/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/plan.md`

Writing should follow the evidence generated there. If the experiment changes, update this writing plan before drafting final chapters.

## Final Deliverables

- Official thesis outline.
- Chapter draft checklist.
- Figures and tables list.
- Citation list grouped by language.
- Appendix plan.
- Advisor checkpoint package.
- Final formatting checklist.
