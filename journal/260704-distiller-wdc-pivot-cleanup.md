---
title: "DistillER/WDC Pivot Cleanup"
date: 2026-07-04
phase: 2
status: completed
plan: ../plans/260704-distiller-wdc-agent-execution/plan.md
tags: [entity-resolution, wdc-products, distillation, llm-labeling, pivot-cleanup]
---

# DistillER/WDC Pivot Cleanup

## Context

The repository guidance still pointed at the older structured-rationale thesis
even though the research decision has moved to cost-aware LLM-label
distillation. This was risky because future agents could keep optimizing the
failed rationale experiment instead of implementing the new direct LLM and
teacher-label baselines.

## Why The Rationale Direction Is No Longer Active

The Phase 03 WDC Products pilot at budget 128 did not support the central
rationale-distillation claim:

| Variant | Train rows | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `label_only` | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Structured rationales increased recall, but the student overpredicted matches
and lost too much precision, macro F1, and accuracy. This makes the old story
weak as a main thesis claim. The result should remain visible as negative
evidence and as motivation for trying simpler answer-only teacher labels.

## New Safe Direction

The active thesis direction is:

> Can compact Entity Matching students distilled from LLM-generated teacher
> labels approach gold-label supervised students while being cheaper at
> inference time than using the LLM directly as the matcher?

The three core arms are:

| Arm | Variant | Role |
|---|---|---|
| A | `gold_label_student` | quality standard using trusted dataset labels |
| B | `direct_llm_matcher` | repeated inference-cost baseline |
| C | `llm_label_distilled_student` | main method: one-time teacher labels plus cheap student inference |

The optional practical fallback is `mixed_gold_llm`, where a small gold seed is
combined with LLM-generated labels if pure teacher labels are noisy.

## Cleanup Done

- Updated repo-level guidance in `AGENTS.md` and `CLAUDE.md`.
- Updated workspace-level guidance in `../AGENTS.md`.
- Updated `QUICKSTART.md` so the first decision gate is the DistillER/WDC pilot.
- Initially marked `rationales/README.md` as historical/reusable
  infrastructure. A follow-up cleanup on 2026-07-06 removed the old rationale
  package from the active tree and kept the result only as research history.
- Updated dataset and document READMEs to reflect the current pivot.

## Next Step

Move into Phase 3:

1. Implement answer-only direct LLM matching on a fixed validation split or
   predeclared validation sample.
2. Log prompt version, model slug, input tokens, output tokens, estimated cost,
   parsed label, gold label, and validity.
3. Implement answer-only teacher-label generation for `train_128`.
4. Keep all prompts, budgets, and evaluation samples fixed before inspecting
   results.
