---
title: DistillER WDC Agent Execution
description: >-
  Detailed execution plan for a DistillER-style WDC Products thesis pivot using
  gold-label compact students, direct LLM matching, LLM-generated teacher
  labels, compact student distillation, label-budget curves, and cost analysis.
status: in_progress
priority: P1
effort: 6-8 weeks for thesis-core evidence
branch: phase-03/minimal-distillation-pilot
tags:
  - research
  - entity-resolution
  - llm-labeling
  - distillation
  - wdc
  - thesis
blockedBy: []
blocks: []
created: '2026-07-04T08:00:47.234Z'
createdBy: 'ck:plan'
source: skill
---

# DistillER WDC Agent Execution

## Overview

This plan replaces the failed rationale-first thesis direction with a safer DistillER-style experiment:

> Can compact Entity Matching students distilled from LLM-generated teacher labels approach gold-label supervised students on WDC Products while being cheaper at inference time than direct LLM matching?

The thesis must not claim that LLM-label distillation for ER is new. DistillER and related work already exist. The defendable contribution is a controlled WDC Products study with explicit label budgets, direct LLM matching cost, compact student distillation, cost accounting, and failure analysis.

## Scope Challenge

- Existing code: WDC loader, serialized WDC pairs, low-label samplers, target builder, FLAN-T5 training, student evaluation, metric utilities, and OpenRouter-backed teacher infrastructure already exist.
- Minimum change set: add direct answer-only LLM matching, answer-only teacher-label generation, validated label cache, target building from LLM labels, pilot runs, aggregation, cost summary, and failure analysis.
- Complexity check: the core should touch about 7-8 modules. Keep one dataset, one teacher, and one student until the pilot proves signal.
- Selected scope: hold scope. Build a strong thesis-core experiment first; treat more datasets and more students as optional expansion.

## Research Contract

Primary research question:

> Can compact ER students distilled from LLM-generated teacher labels approach the performance of students trained on gold labels, while avoiding the repeated inference cost of direct LLM matching?

Secondary questions:

- Does mixing a small gold seed with LLM-generated labels improve stability?
- How much one-time teacher labeling cost is needed per useful student checkpoint?
- How expensive is direct LLM matching on the same evaluation pairs?
- What errors are introduced by teacher labels, and which product attributes cause them?

## Core Experiment Arms

| Arm | Variant | Training / Inference Pattern | Purpose |
|---|---|---|---|
| A | `gold_label` | compact student trained on dataset labels | trusted supervised performance standard |
| B | `direct_llm_matcher` | LLM predicts each evaluation pair directly | inference-cost baseline |
| C | `llm_label` | LLM labels training pairs, compact student learns from them | main distillation method |
| C optional | `mixed_gold_llm` | small gold seed plus LLM labels | safer practical distillation variant |
| historical | `old_structured_rationale` | existing rationale target outputs | optional negative-history ablation only |

## Architecture

```text
WDC Products raw/cache data
  -> serialized pair JSONL
  -> low-label budget sampler
  -> direct LLM matcher on fixed evaluation pairs
  -> answer-only teacher LLM labeler
  -> teacher-label validator and cache
  -> target builder: gold / llm / mixed
  -> compact student training
  -> validation and test evaluation
  -> aggregation, cost table, error analysis
  -> thesis tables and figures
```

Teacher LLM calls are used only to prepare training labels. Final inference must use the compact student without teacher calls.

## Planned Artifacts

| Artifact | Suggested Path |
|---|---|
| Experiment contract | `plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md` |
| Direct LLM predictions | `outputs/distiller_wdc/direct_llm/*.predictions.jsonl` |
| Teacher-label cache | `data/cache/wdc_products/teacher_labels/*.jsonl` |
| Label targets | `data/cache/wdc_products/targets/*.llm_label.targets.jsonl` |
| Mixed targets | `data/cache/wdc_products/targets/*.mixed_gold_llm.targets.jsonl` |
| Student outputs | `outputs/distiller_wdc/...` |
| Aggregated metrics | `outputs/distiller_wdc/summary/*.csv` |
| Failure analysis | `outputs/distiller_wdc/analysis/*.csv` |
| Thesis figures | `outputs/distiller_wdc/figures/*.png` |

## Phases

| Phase | Name | Status | Priority | Effort | Dependencies |
|-------|------|--------|----------|--------|--------------|
| 1 | [Research Contract](./phase-01-research-contract.md) | Completed | P1 | 1 day | None |
| 2 | [Pivot Cleanup](./phase-02-pivot-cleanup.md) | Pending | P1 | 0.5-1 day | Phase 1 |
| 3 | [Teacher Labeling And Direct LLM Baseline](./phase-03-teacher-labeling-pipeline.md) | Pending | P1 | 3-5 days | Phase 1 |
| 4 | [Training Target Builder](./phase-04-training-target-builder.md) | Pending | P1 | 2-3 days | Phase 3 |
| 5 | [Pilot Student Runs And Direct LLM Baseline](./phase-05-pilot-student-runs.md) | Pending | P1 | 1-2 weeks | Phase 4 |
| 6 | [Full Budget Study](./phase-06-full-budget-study.md) | Pending | P1 | 2-3 weeks | Phase 5 |
| 7 | [Failure And Cost Analysis](./phase-07-failure-and-cost-analysis.md) | Pending | P1 | 1 week | Phase 5, 6 |
| 8 | [Thesis Artifact Handoff](./phase-08-thesis-artifact-handoff.md) | Pending | P1 | 2-3 days | Phase 7 |

## Stop Or Continue Gates

| Gate | Continue If | Revise If | Stop If |
|---|---|---|---|
| After Phase 3 | Direct LLM costs logged and valid labels generated, invalid rate under 5-10 percent | Prompt parsing is brittle | Teacher cannot label or directly match WDC reliably |
| After Phase 5 | `llm_label` F1 is near gold baseline and projected cheaper than repeated direct LLM matching, or mixed variant helps | Precision/recall tradeoff is unstable | LLM labels are worse than simple baseline with no useful explanation |
| After Phase 6 | Label-efficiency or cost-efficiency story is clear | Need second student/model | No thesis-grade pattern appears |

## Validation Commands

Use the project virtual environment for local checks:

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
.venv/bin/python -m unittest discover -s tests
```

Representative training and evaluation commands should keep the existing pattern:

```bash
.venv/bin/python -m experiments.train_mt5 \
  --train-targets data/cache/wdc_products/targets/train_128.llm_label.targets.jsonl \
  --validation-targets data/cache/wdc_products/targets/validation.gold_label.targets.jsonl \
  --output-dir outputs/distiller_wdc/flan-t5-base/train_128/llm_label \
  --model-name google/flan-t5-base
```

## Main Risks

| Risk | Level | Mitigation |
|---|---|---|
| Idea is close to DistillER and Steiner/Bizer | High | Frame as WDC-focused controlled adaptation, not invention |
| Direct LLM baseline is expensive | Medium | Use a fixed validation set or predeclared sample and project cost transparently |
| LLM labels are too noisy | High | Add validation, mixed labels, and failure analysis |
| WDC is too hard or too imbalanced | Medium | Report precision/recall separately; use validation thresholding if classifier added |
| FLAN-T5 is not the best ER student | Medium | Add MiniLM/RoBERTa only after pilot |
| Compute budget grows | Medium | Pilot with 128 and 256 before full budgets |
| Thesis writing falls behind experiments | Medium | Hand off tables, commands, and figures into the thesis writing plan after each phase |

## Relationship To Thesis Writing Plan

The companion writing plan is:

`/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`

Agent Phase 1 feeds thesis Chapter 2 and Chapter 4. Agent Phases 3-4 feed Chapter 3. Agent Phases 5-7 feed Chapter 5. Agent Phase 8 packages the outputs for final writing.
