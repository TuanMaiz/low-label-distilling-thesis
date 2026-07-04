---
title: "DistillER-Style WDC Pivot Agent Execution Plan"
description: "Execution plan for testing whether LLM-generated labels can train compact ER students on WDC Products at lower human-label cost."
status: pending
priority: P1
effort: "6-8 weeks for thesis-core evidence"
created: 2026-07-04
tags: [entity-resolution, llm-labeling, distillation, wdc, thesis]
blockedBy: []
blocks: []
---

# DistillER-Style WDC Pivot Agent Execution Plan

> Superseded by the detailed ck plan:
> `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/plan.md`.

## Purpose

Build the minimum experimental evidence for a master's thesis around this claim:

> LLM-generated labels can reduce human labeling cost for Entity Matching by training compact student models whose performance approaches gold-label supervision on WDC Products.

This is not framed as inventing DistillER. It is a controlled adaptation and stress test for WDC Products, compact students, label budget, and cost.

## Scope Challenge

- Existing code: WDC serialization, low-label sampling, target building, FLAN-T5 training, evaluation metrics, and previous rationale-generation code already exist.
- Minimum change set: stop rationale-first work; reuse WDC splits and student evaluation; add answer-only LLM label generation, LLM-label targets, cost logging, and result aggregation.
- Complexity: keep one main dataset, one teacher, one student at first. Add a second student only after the pilot shows signal.

## Not In Scope First

- Multi-teacher comparison.
- Full DistillER reproduction.
- Rationale or explanation distillation as the main claim.
- Many datasets before WDC pilot succeeds.
- Complex active-learning or routing methods.

## Phase 0: Freeze The Old Rationale Direction

### Goal

Make the pivot explicit so future agents do not keep optimizing the failed structured-rationale experiment.

### Tasks

1. Keep old rationale results as negative evidence.
2. Record that structured rationales increased recall but hurt precision and F1.
3. Do not delete old code; mark it as legacy or optional ablation.
4. Update project guidance only after the new pilot contract is stable.

### Success Criteria

- The thesis direction is now LLM-label supervision, not rationale supervision.
- Old rationale work can still be cited as motivation for the pivot.

## Phase 1: Experiment Contract

### Goal

Define the exact pilot before spending compute.

### Main Question

Can a compact ER student trained on LLM-generated labels approach a compact student trained on gold labels under the same WDC low-label budgets?

### Pilot Matrix

| Axis | First Choice |
|---|---|
| Dataset | WDC Products |
| Teacher | One low-cost LLM |
| Teacher output | Answer-only match / non-match |
| Student | Existing FLAN-T5-base pipeline first, encoder classifier second if time |
| Budgets | 128 and 256 for pilot |
| Comparisons | gold labels vs LLM labels vs optional mixed labels |
| Metrics | precision, recall, F1, accuracy, invalid-output rate, teacher cost |

### Success Criteria

- A one-page experiment contract exists.
- Each run has a fixed input path, output path, seed, and budget.
- Test labels are never sent to the teacher for training-data generation.

## Phase 2: LLM Label Generation

### Goal

Generate cheap answer-only labels for WDC training pairs.

### Tasks

1. Reuse serialized WDC pair format.
2. Create an answer-only prompt: given two product records, output `match` or `non_match`.
3. Cache one JSONL row per labeled pair.
4. Store teacher model, prompt version, timestamp, input token count, output token count, estimated cost, and raw answer.
5. Validate outputs into canonical labels.

### Suggested Files

- Modify: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/prompts.py` or create a separate label prompt module.
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/generate_teacher_labels.py`.
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/validate_teacher_labels.py`.
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/analysis/cost_summary.py` if missing or incomplete.

### Success Criteria

- LLM-label caches exist for 128 and 256 WDC training pairs.
- Invalid label rate is reported.
- Cost per labeled pair is reported.

## Phase 3: Build Training Targets

### Goal

Train the same student architecture with different supervision sources.

### Variants

| Variant | Target Source |
|---|---|
| `gold_label` | Original dataset label |
| `llm_label` | Validated teacher label |
| `mixed_gold_llm` | Small gold seed plus LLM-labeled expansion |

### Tasks

1. Extend target builder to accept teacher-label cache.
2. Produce target JSONL files for each supervision variant.
3. Ensure validation and test targets always use gold labels.
4. Keep output labels compact: `match` / `non_match`.

### Success Criteria

- Target files are generated for each pilot variant.
- The same validation/test split is used across all variants.

## Phase 4: Pilot Student Runs

### Goal

Find out quickly whether the thesis has a positive signal.

### Runs

| Budget | Variant |
|---:|---|
| 128 | gold_label |
| 128 | llm_label |
| 128 | mixed_gold_llm, optional |
| 256 | gold_label |
| 256 | llm_label |
| 256 | mixed_gold_llm, optional |

### Tasks

1. Train the compact student for each target file.
2. Evaluate on validation.
3. Evaluate promising runs on test.
4. Save metrics and predictions.
5. Compare against the current label-only baseline.

### Success Criteria

- A pilot table exists with F1, precision, recall, accuracy, invalid-output rate, and cost.
- Decision can be made: expand, revise, or stop.

## Phase 5: Full Label-Budget Study

### Goal

Turn the pilot into thesis-grade evidence.

### Budgets

Use `16 / 32 / 64 / 128 / 256`, plus a larger reference budget if compute allows.

### Tasks

1. Run all successful variants across all budgets.
2. Repeat with multiple seeds if time allows.
3. Aggregate mean and standard deviation.
4. Plot label-efficiency curves.
5. Add cost-efficiency table: human label count, LLM label cost, student inference cost.

### Success Criteria

- Results show where LLM labels help, match, or fail compared with gold labels.
- Plots and tables are ready for thesis Chapter 5.

## Phase 6: Failure Analysis

### Goal

Explain the behavior, not just report numbers.

### Tasks

1. Sample false positives and false negatives.
2. Compare teacher errors against student errors.
3. Group failures by product attribute: title, brand, model number, price, description.
4. Identify whether LLM labels create systematic bias.

### Success Criteria

- At least one qualitative error table exists.
- Thesis can explain why the method works or fails.

## Phase 7: Optional Expansion

### Goal

Add variety only after WDC results are stable.

### Options

- Add one non-WDC product dataset not used by DistillER.
- Add one classic ER dataset from Magellan or DeepMatcher.
- Add MiniLM/RoBERTa classifier as a more standard compact ER student.

### Rule

Do not expand until Phase 4 answers the pilot question.

## Final Deliverables

- Experiment contract.
- Teacher-label generation cache.
- Training targets for gold, LLM, and optional mixed supervision.
- Student checkpoints and predictions.
- Metrics tables.
- Label-efficiency plots.
- Cost summary.
- Failure-analysis table.
- Thesis-ready method and experiment notes.
