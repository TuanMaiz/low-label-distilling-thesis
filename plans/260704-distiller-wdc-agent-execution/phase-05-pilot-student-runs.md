---
phase: 5
title: "Pilot Student Runs And Direct LLM Baseline"
status: pending
priority: P1
effort: "1-2 weeks"
dependencies: [4]
---

# Phase 5: Pilot Student Runs And Direct LLM Baseline

## Overview

Run the smallest experiment that can decide whether the thesis direction has
signal: gold-label compact student, direct LLM matcher, random LLM-label
distilled student, and one actively selected LLM-label distilled student.

## Requirements

- Functional: train and evaluate `gold_random`, `llm_random`, and `llm_active_bucketed_v1` at budget `128`.
- Functional: add `mixed_gold_llm_active` only if pure active LLM labels are noisy but promising.
- Functional: expand to budget `256` only after fixed `256` selection manifests exist.
- Functional: include the fixed direct LLM matcher result and cost from Phase 3 in the same pilot table.
- Non-functional: save checkpoints, predictions, metrics, and run metadata in stable output folders.
- Decision: end this phase with a clear continue/revise/stop recommendation.

## Architecture

```text
targets for 128 random and active selected pairs
  -> train compact student
  -> validation metrics
direct LLM predictions
  -> validation metrics and inference cost
student + direct LLM metrics
  -> random-vs-active same-budget comparison
  -> test metrics for promising variants
  -> pilot decision table
```

## Related Code Files

- Reuse: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/train_mt5.py`
- Reuse: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/evaluate_student.py`
- Create optional: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/run_llm_label_pilot.py`
- Create optional: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/aggregate_results.py`
- Outputs: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/`

## Implementation Steps

1. Confirm or map the existing `gold_label`/`label_only` baseline as `gold_random` at budget `128`.
2. Load or run the fixed `direct_llm_matcher` baseline on the validation set or predeclared validation sample.
3. Train `llm_random` student at budget `128`.
4. Train `llm_active_bucketed_v1` student at budget `128` after its manifest and teacher cache are fixed.
5. If cost and time allow, train `mixed_gold_llm_active` at budget `128`.
6. Repeat the same student set at budget `256` only if the sampler/selector has been extended.
7. Evaluate all student models on the same validation split used for the direct LLM comparison.
8. Evaluate the best or most informative variants on the test split only after the validation decision.
9. Aggregate metrics into one pilot table:
   - F1.
   - precision.
   - recall.
   - accuracy.
   - invalid-output rate.
   - selection strategy.
   - direct LLM inference cost.
   - teacher-labeling cost.
   - estimated student inference cost.
   - gain or loss versus `llm_random`.
10. Write a pilot decision note.

## Success Criteria

- [ ] Pilot metrics table exists.
- [ ] Predictions are saved for each run.
- [ ] Direct LLM baseline quality and cost are included.
- [ ] The `llm_random` and `llm_active_bucketed_v1` gaps from `gold_random` are quantified.
- [ ] The active-vs-random gain or loss at the same budget is quantified.
- [ ] Cost gap between direct LLM matching and distilled student inference is quantified.
- [ ] Continue/revise/stop decision is written.

## Risk Assessment

- Risk: FLAN-T5 output parsing introduces invalid outputs.
  Mitigation: keep labels compact and track invalid-output rate.
- Risk: validation result is noisy at low budgets.
  Mitigation: treat pilot as directional; only expand after consistent signal.
- Risk: active selection improves recall but damages precision.
  Mitigation: report precision/recall separately and inspect hard-negative false positives.
- Risk: no positive signal.
  Mitigation: use failure analysis to decide whether mixed labels, simpler selection, or encoder classifier is needed.
- Risk: direct LLM comparison is accused of cherry-picking.
  Mitigation: use the fixed evaluation set or a predeclared sample before inspecting direct LLM results.
