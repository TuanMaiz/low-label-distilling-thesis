---
phase: 1
title: Research Contract
status: completed
priority: P1
effort: 1 day
dependencies: []
---

# Phase 1: Research Contract

## Overview

Freeze the research question, experiment matrix, datasets, metrics, and
stop/continue gates before generating more labels or training models. Updated
on 2026-07-07 to add the active LLM-label selection extension under the same
WDC low-budget thesis scope.

## Requirements

- Functional: define exact WDC split, budgets, selection strategies, supervision variants, teacher model, student model, metrics, output paths, and cost fields.
- Non-functional: keep the claim conservative and defensible against DistillER and LLM-labeled ER related work.
- Reproducibility: every run must have fixed seed, input file, output directory, and prompt version.

## Architecture

This phase does not implement code. It produces the contract that all later modules follow:

```text
research question
  -> experiment matrix
  -> selection strategy contract
  -> artifact path convention
  -> metric and cost definitions
  -> stop/continue gates
```

## Related Code Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/CLAUDE.md`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/AGENTS.md`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/stats.json`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/documents/zeakis-2026-distiller-knowledge-distillation-in-entity-resolution-with-large-language-models.pdf`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/research/active-selection-extension.md`

## Implementation Steps

1. State the final working title: "Cost-Aware Active LLM Labeling for Low-Budget Entity Matching".
2. State the primary question: whether active selection of LLM-labeled WDC training pairs improves compact ER students over random LLM-label distillation at the same cost.
3. Define supervision and selection variants: `gold_random`, `llm_random`, `llm_active_bucketed_v1`, optional `llm_active_hybrid`, optional `llm_active_uncertainty`, optional `llm_active_diversity`, and optional `mixed_gold_llm_active`.
4. Define pilot budgets: `128` and `256`.
5. Define full budgets: `16`, `32`, `64`, `128`, `256`, plus larger reference only if cheap.
6. Define primary metric as F1, with precision and recall always reported separately.
7. Define cost metrics: teacher prompt tokens, completion tokens, cost per labeled pair, total teacher-label cost, student inference cost.
8. Define stop/continue thresholds:
   - Continue if `llm_active_bucketed_v1` improves over or usefully diagnoses `llm_random` at the same budget.
   - Revise if active labels help only one metric, select noisy teacher examples, or need mixed gold+LLM supervision.
   - Stop if random and active LLM-label students are both weak and no useful failure story exists.
9. Write the one-page contract in the plan `research/` folder.

## Success Criteria

- [x] Research question is frozen.
- [x] Experiment matrix is written.
- [x] Stop/continue gates are explicit.
- [x] Later agents can run without re-debating the thesis direction.

## Output

- Created: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`
- Created: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/research/active-selection-extension.md`

## Risk Assessment

- Risk: contract becomes too ambitious.
  Mitigation: one dataset, one teacher, one student until pilot succeeds.
- Risk: novelty is overstated.
  Mitigation: explicitly position as controlled WDC adaptation, active-selection comparison, and cost study.
