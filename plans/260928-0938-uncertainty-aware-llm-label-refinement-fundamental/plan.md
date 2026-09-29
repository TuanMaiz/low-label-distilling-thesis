---
title: "Uncertainty-Aware LLM Label Refinement for Compact Entity Matching"
description: "Foundational plan for gold-free detection and selective refinement of LLM-generated Entity Matching labels, followed by hidden-gold evaluation and cost analysis."
status: pending
priority: P1
effort: "TBD after contract freeze"
branch: "refactor/full-label-er-migration"
tags: [entity-resolution, llm-labeling, label-refinement, noisy-labels, experimental]
blockedBy: []
blocks: [260704-distiller-wdc-thesis-writing]
created: 2026-09-28
---

# Uncertainty-Aware LLM Label Refinement for Compact Entity Matching

## Overview

Extend the current full-label experiment from a comparison of human labels and
one-pass LLM labels into a controlled study of gold-free label-error detection,
selective prune/relabel treatments, compact-model training, hidden-gold
evaluation, and end-to-end cost. Refinement is not assumed to help every
dataset; identifying when it should be skipped is a valid result.

Main research question:

> When and how should LLM-generated labels be refined to train compact Entity
> Matching models effectively?

This is a living foundation. It records the complete intended pipeline and its
scope guards, but does not freeze unresolved scientific choices or authorize
paid calls, OOF training, additional full-model runs, or final-test access.

## Relationship to the Existing Plan

This plan extends
[`Full-Label Cross-Encoder ER Migration`](../260820-1507-full-label-er-migration/plan.md).
It preserves three datasets, three compact cross-encoder students, full human
and LLM training labels, Match F1, direct-LLM cost comparison, and locked test
sets. It adds three-pass teacher labeling, OOF evidence, label-error detection,
prune/relabel treatments, analytical controls, and post-hoc error analysis.

The completed WDC-Qwen one-pass vertical slice remains a pilot and engineering
proof. If the main experiment adopts three-pass majority labels, that result is
not automatically a final main-experiment cell.

## Research Questions

1. How often and in what ways do LLM-generated labels disagree with hidden
   human labels across datasets?
2. How accurately can teacher consistency, Confident Learning, and
   teacher-student disagreement detect those errors without human labels?
3. Do prune and selective relabel improve downstream compact models relative
   to unrefined labels and equally sized random interventions?
4. Do the selected rules transfer across datasets and student architectures?
5. When is refinement worth its extra API/GPU cost, and when should it be
   skipped?

## Scientific Boundary

```text
Operational pipeline (no gold)
training pairs -> three teacher passes -> majority labels -> 3-fold OOF
-> detection/ranking -> prune or evidence-check relabel -> student training

Evaluation pipeline (hidden gold allowed only after outputs freeze)
frozen rankings/targets -> detection metrics -> downstream metrics
-> random/oracle comparison -> error analysis -> cost/decision analysis
```

Human training labels, validation labels, and test labels must not determine
detector scores, thresholds, selection counts, selected pair IDs, or treatment
outputs. Test remains locked until every method and validation choice is frozen.

## Preserved Scope

- Three benchmark datasets and three compact cross-encoder students.
- One teacher model: GPT-5.6 Sol-high.
- Three teacher judgments per training pair for self-consistency.
- Prune and selective relabel only; no reweight treatment.
- Match F1 primary; detection, supporting accuracy, timing, throughput, and
  cost metrics secondary.
- Gold-free operational pipeline with evaluation-only gold/oracle controls.
- Tiered pilot, detection study, and downstream generalization instead of a
  full combinatorial ablation matrix.

## Explicit Non-Goals

- Active learning, low-label regimes, rationale distillation, or cascades.
- Full student-specific detection matrices.
- A large collection of outlier-detection algorithms.
- Prompt-variant expansion unless exact repeats yield no measurable signal.
- Repeated final-test tuning or gold-derived detector thresholds.
- Additional datasets or students beyond the frozen three.

## Current Phase

| # | Phase | Status |
|---|-------|--------|
| 1 | [Freeze the Foundational Scientific and Execution Contract](./phase-01-start.md) | Pending |
| 2 | [Run Three Separate Teacher Passes](./phase-02-run-three-separate-teacher-passes.md) | Complete |
| 3 | [Build WDC Teacher Majority and Consistency Artifacts](./phase-03-build-wdc-teacher-majority-and-consistency-artifacts.md) | Complete |

Phase 2 deliberately reuses the current labeler and executes separate named
runs rather than introducing a multipass orchestration framework. Later phases
must be added through the plan CLI after the relevant Phase 1 decisions freeze.
Expected extensions are OOF and detection, treatments and controls, tiered
execution, evaluation/error analysis, cost aggregation, and final verification.

Phase 3 is the small offline bridge from the three WDC teacher passes to the
future OOF phase. It produces the majority label and `3/3` or `2/3`
consistency for every training pair without reading gold or calling a model.
It completed with 2,500 majority labels: 2,487 pairs have `3/3` consistency
and 13 have `2/3` consistency.

## Success Criteria

- [ ] The research questions, gold boundary, datasets, students, teacher
      protocol, OOF protocol, detector definitions, treatments, controls,
      metrics, workload, budgets, and test-release gate are reviewed and frozen.
- [ ] Every expensive action has an offline/dry-run acceptance path.
- [ ] The proposed main matrix fits the thesis schedule without silently
      expanding every ablation across all students.
- [ ] The existing WDC and DBLP work is reused rather than rebuilt.
- [ ] Detailed executable phase files can extend this foundation without
      changing its central scientific separation.

<!-- slug: uncertainty-aware-llm-label-refinement-fundamental -->
