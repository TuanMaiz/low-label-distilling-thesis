---
title: Cost-Aware Active LLM Labeling For WDC Entity Matching
description: >-
  Detailed execution plan for a WDC Products thesis pivot using low-budget
  active selection of LLM-labeled training pairs, compact student distillation,
  direct LLM matching, label-budget curves, and cost analysis.
status: in_progress
priority: P1
effort: 6-8 weeks for thesis-core evidence
branch: codex/distiller-wdc-implementation
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

# Cost-Aware Active LLM Labeling For WDC Entity Matching

## Overview

This plan replaces the failed rationale-first thesis direction with a safer
active-selection extension of the DistillER-style experiment:

> Under low-label budgets and hard product-matching conditions, can actively
> selected LLM-labeled training pairs produce better compact ER students than
> random LLM-label distillation, and when are they cheaper than repeated direct
> LLM matching?

The thesis must not claim that LLM-label distillation or active learning for ER
is new. DistillER, Steiner/Bizer-style LLM-labeling work, active ER, and direct
LLM matching papers already cover the broad method families. The defendable
contribution is a controlled WDC Products study through a cost, low-budget, data
selection, and failure-slice analysis lens: explicit label-budget curves,
selection-strategy comparisons, direct LLM matching cost, compact student
distillation, break-even cost accounting, teacher-noise analysis, and WDC
difficulty-slice failure analysis.

## Thesis Lens

The method family is intentionally not presented as novel. The thesis studies
an existing LLM-teacher-to-student pattern through a narrower empirical lens:

- **Cost lens:** compare repeated direct LLM inference with one-time teacher
  labeling plus compact-student training and inference.
- **Low-label budget lens:** report behavior across `16 / 32 / 64 / 128` and,
  if feasible, `256 / full`, rather than only one training size.
- **Data-selection lens:** compare random LLM-label distillation with active
  pair selection under the same labeling budgets, so the thesis asks which pairs
  are worth spending LLM calls on.
- **WDC difficulty lens:** explain performance on hard product-pair slices such
  as hard negatives, missing attributes, brand conflicts, long descriptions,
  model-number/title overlap, and price/currency mismatch.
- **Teacher-noise lens:** compare LLM-generated labels with gold labels before
  training, then check whether the compact student inherits, amplifies, or
  smooths teacher mistakes.
- **External-validity lens:** keep WDC Products as the thesis-core stress test;
  add Abt-Buy, Walmart-Amazon, or DBLP-style datasets later only as replication
  checks if time permits.

## Scope Challenge

- Existing code: WDC loader, serialized WDC pairs, low-label samplers, target
  builder, FLAN-T5 training, student evaluation, metric utilities, and
  OpenRouter-backed teacher infrastructure already exist.
- Minimum change set: add direct answer-only LLM matching, answer-only
  teacher-label generation, validated label cache, active pair-selection
  manifests, target building from selected LLM labels, pilot runs, aggregation,
  cost summary, and failure analysis.
- Complexity check: keep one dataset, one teacher, one student, random
  selection, and at most two active selection strategies until the pilot proves
  signal.
- Selected scope: hold scope. Build a strong WDC thesis-core experiment first;
  treat more datasets, more students, and complex iterative active learning as
  optional expansion.

## Research Contract

Primary research question:

> Under low-label budgets on WDC Products, can active selection of
> LLM-labeled training pairs produce compact ER students that outperform random
> LLM-label distillation at the same labeling cost, while becoming cheaper than
> repeated direct LLM matching at useful inference volumes?

Secondary questions:

- Which selection strategy works best under a fixed LLM-labeling budget:
  random, uncertainty, diversity, hard-negative, or hybrid selection?
- Do active strategies select genuinely useful pairs, or mostly pairs where the
  LLM teacher is noisy?
- Does mixing a small gold seed with LLM-generated labels improve stability?
- How much one-time teacher labeling cost is needed per useful student checkpoint?
- How expensive is direct LLM matching on the same evaluation pairs?
- What errors are introduced by teacher labels, and which WDC product-pair
  slices cause them?
- Do students inherit teacher errors or smooth over some noisy labels?
- Do the WDC findings replicate on one additional dataset if time permits?

## Core Experiment Arms

| Arm | Variant | Training / Inference Pattern | Purpose |
|---|---|---|---|
| A | `gold_random` | compact student trained on randomly sampled gold labels | trusted low-budget supervised performance standard |
| B | `direct_llm_matcher` | LLM predicts each evaluation pair directly | repeated inference-cost baseline |
| C | `llm_random` | random training pairs are LLM-labeled, then a compact student learns from them | random distillation control |
| D | `llm_active_uncertainty` | uncertain candidate pairs are LLM-labeled, then distilled | active-selection test |
| D optional | `llm_active_diversity` | diverse representative candidate pairs are LLM-labeled, then distilled | coverage-oriented active-selection test |
| D | `llm_active_bucketed_v1` | equal default coverage of easy-match, hard-match, easy-non-match, and hard-negative candidate buckets before LLM labeling | practical active-selection candidate |
| D optional | `llm_active_hybrid` | older blended-score selector retained for comparison only | superseded active-selection pilot |
| D optional | `mixed_gold_llm_active` | small gold seed plus actively selected LLM labels | safer practical active distillation variant |
| historical | `old_structured_rationale` | recorded Phase 03 result only | negative-history context, not active code |

## Architecture

```text
WDC Products raw/cache data
  -> serialized pair JSONL
  -> low-label budget sampler / active pair selector
  -> direct LLM matcher on fixed evaluation pairs
  -> answer-only teacher LLM labeler
  -> teacher-label validator and cache
  -> target builder: gold / random LLM / active LLM / mixed
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
| Active-selection note | `plans/260704-distiller-wdc-agent-execution/research/active-selection-extension.md` |
| Direct LLM predictions | `outputs/distiller_wdc/direct_llm/*.predictions.jsonl` |
| Selection manifests | `data/cache/wdc_products/selection/*.jsonl` |
| Teacher-label cache | `data/cache/wdc_products/teacher_labels/*.jsonl` |
| Random LLM-label targets | `data/cache/wdc_products/targets/*.llm_random.targets.jsonl` |
| Active-label targets | `data/cache/wdc_products/targets/*.llm_active_*.targets.jsonl` |
| Mixed targets | `data/cache/wdc_products/targets/*.mixed_gold_llm_active.targets.jsonl` |
| Student outputs | `outputs/distiller_wdc/...` |
| Aggregated metrics | `outputs/distiller_wdc/summary/*.csv` |
| Failure analysis | `outputs/distiller_wdc/analysis/*.csv` |
| Thesis figures | `outputs/distiller_wdc/figures/*.png` |

Optional external-validity artifacts for later datasets should live under the
same `outputs/distiller_wdc/` summary conventions only after the WDC pilot has a
clear signal. They are not required for the WDC thesis-core claim.

## Phases

| Phase | Name | Status | Priority | Effort | Dependencies |
|-------|------|--------|----------|--------|--------------|
| 1 | [Research Contract](./phase-01-research-contract.md) | Completed | P1 | 1 day | None |
| 2 | [Pivot Cleanup](./phase-02-pivot-cleanup.md) | Completed | P1 | 0.5-1 day | Phase 1 |
| 3 | [Teacher Labeling, Active Selection, And Direct LLM Baseline](./phase-03-teacher-labeling-pipeline.md) | In Progress | P1 | 3-5 days | Phase 1 |
| 4 | [Training Target Builder](./phase-04-training-target-builder.md) | Completed | P1 | 2-3 days | Phase 3 |
| 5 | [Pilot Student Runs And Direct LLM Baseline](./phase-05-pilot-student-runs.md) | Pending | P1 | 1-2 weeks | Phase 4 |
| 6 | [Full Budget Study](./phase-06-full-budget-study.md) | Pending | P1 | 2-3 weeks | Phase 5 |
| 7 | [Failure And Cost Analysis](./phase-07-failure-and-cost-analysis.md) | Pending | P1 | 1 week | Phase 5, 6 |
| 8 | [Thesis Artifact Handoff](./phase-08-thesis-artifact-handoff.md) | Pending | P1 | 2-3 days | Phase 7 |

## Stop Or Continue Gates

| Gate | Continue If | Revise If | Stop If |
|---|---|---|---|
| After Phase 3 | Direct LLM costs logged and valid labels generated, invalid rate under 5-10 percent | Prompt parsing is brittle | Teacher cannot label or directly match WDC reliably |
| After Phase 5 | active LLM-label student beats or clearly diagnoses random LLM-label student at the same budget, and projected cost is lower than repeated direct LLM matching | active strategy helps only one metric or mostly selects noisy pairs | random and active LLM-label students are both worse than simple baseline with no useful diagnostic story |
| After Phase 6 | label-selection, label-efficiency, or cost-efficiency story is clear | need simpler active strategy, second student/model, or more seeds | no thesis-grade pattern appears |

## Validation Commands

Use the project virtual environment for local checks:

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
.venv/bin/python -m unittest discover -s tests
```

Representative training and evaluation commands should keep the existing pattern:

```bash
.venv/bin/python -m experiments.train_mt5 \
  --train-targets data/cache/wdc_products/targets/train_128.llm_active_bucketed_v1.targets.jsonl \
  --validation-targets data/cache/wdc_products/targets/validation.gold_label.targets.jsonl \
  --output-dir outputs/distiller_wdc/flan-t5-base/train_128/llm_active_bucketed_v1 \
  --model-name google/flan-t5-base
```

## Main Risks

| Risk | Level | Mitigation |
|---|---|---|
| Idea is close to DistillER, active ER, and Steiner/Bizer | High | Frame as WDC-focused cost-aware active LLM labeling under low budgets, not invention |
| Direct LLM baseline is expensive | Medium | Use a fixed validation set or predeclared sample and project cost transparently |
| LLM labels are too noisy | High | Add validation, mixed labels, and failure analysis |
| WDC is too hard or too imbalanced | Medium | Report precision/recall separately; use validation thresholding if classifier added |
| FLAN-T5 is not the best ER student | Medium | Add MiniLM/RoBERTa only after pilot |
| Compute budget grows | Medium | Pilot with 128 and 256 before full budgets |
| Active strategies become too complex | Medium | Start with random and one equal-ratio bucketed selector; avoid iterative retraining until needed |
| Extra datasets distract from thesis core | Medium | Treat non-WDC datasets as optional external-validity checks after WDC signal |
| Thesis writing falls behind experiments | Medium | Hand off tables, commands, and figures into the thesis writing plan after each phase |

## Relationship To Thesis Writing Plan

The companion writing plan is:

`/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`

Agent Phase 1 feeds thesis Chapter 2 and Chapter 4. Agent Phases 3-4 feed
Chapter 3, including selection manifests and target construction. Agent Phases
5-7 feed Chapter 5, including active-vs-random, cost, and failure-slice
results. Agent Phase 8 packages the outputs for final writing.
