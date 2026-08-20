---
title: "Full-Label Cross-Encoder ER Migration"
description: "Migrate the ER study to a frozen 3-dataset × 3-cross-encoder × 2-label-source experiment with direct-LLM and cost baselines."
status: pending
priority: P1
effort: "4-6 weeks"
branch: "refactor/full-label-er-migration"
tags: [entity-resolution, cross-encoder, llm-labeling, experiment-migration, tdd]
blockedBy: []
blocks: [260704-distiller-wdc-thesis-writing]
created: "2026-08-20T08:07:45.001Z"
createdBy: "ck:plan"
source: skill
---

# Full-Label Cross-Encoder ER Migration

## Overview

Test whether compact cross-encoder ER students trained on complete LLM-generated
hard-label training sets are a practical alternative to the same students
trained on benchmark gold labels. The fixed matrix is 3 benchmark datasets × 3
eligible cross-encoder students × 2 training-label sources, with one fixed
seed/run per cell. A direct LLM matcher is one per-dataset accuracy/cost
baseline, not a training arm.

Gold validation/test labels are evaluation-only. Match-class F1 is primary;
precision, recall, macro F1, accuracy, labeling/training/inference cost,
throughput, and break-even query count are supporting evidence.

The deleted execution plan and Phase-05 runner are available only in Git
history; they are not implementation dependencies. This plan blocks
`plans/260704-distiller-wdc-thesis-writing/plan.md` until the contract and
result schema are stable.

## Scope

In scope: immutable dataset/model/teacher declarations, dataset-neutral pair
loading, complete gold and LLM targets, 18 student cells, 3 direct baselines,
provenance, metrics, cost analysis, and handoff documentation.

Out of scope: low-label budgets, active selection, rationale distillation,
adaptive bi-/cross-encoder cascades, a multi-seed experiment dimension, and
extra datasets/models unless the supervisor explicitly requires them.

## Architecture

`committed scientific plan -> included dataset/student configs -> namespaced serialized splits ->
complete gold/LLM train targets -> cross-encoder train/evaluate -> immutable
cell artifacts -> aggregate accuracy/cost/break-even`

Identities are type-specific: normalized splits use dataset/version/split/hash;
teacher caches add teacher/prompt; student cells add label source/model/revision/
seed; direct outputs use
`outputs/full_label/{experiment_id}/{dataset_id}/direct_llm/{teacher_id}/{scope_id}/`.
Validation/test labels never enter prompts or train targets. Missing, duplicate,
invalid, or extra LLM labels fail closed.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Freeze Experiment Contract](./phase-01-freeze-experiment-contract.md) | Pending |
| 2 | [Generalize Dataset Pipeline](./phase-02-generalize-dataset-pipeline.md) | Pending |
| 3 | [Build Full Label Targets](./phase-03-build-full-label-targets.md) | Pending |
| 4 | [Finalize Cross-Encoder Students](./phase-04-finalize-cross-encoder-students.md) | Pending |
| 5 | [Refactor Experiment Runner](./phase-05-refactor-experiment-runner.md) | Pending |
| 6 | [Aggregate Metrics and Cost](./phase-06-aggregate-metrics-and-cost.md) | Pending |
| 7 | [Verify and Handoff](./phase-07-verify-and-handoff.md) | Pending |

## Dependencies

- Phase 1 records scientific decisions in a human-readable checklist and a Git
  commit freezes changes. Phase 2 implements exactly three included dataset
  configs; Phases 3 and 4 may then proceed in parallel.
- Phase 5 needs Phases 3-4; Phase 6 needs Phase 5; Phase 7 needs all prior phases.

## Verification Command

`.venv/bin/python -m unittest discover -s tests`
