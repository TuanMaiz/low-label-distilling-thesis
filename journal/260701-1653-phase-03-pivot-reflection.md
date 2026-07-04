---
title: "Phase 03 Pivot Reflection: From Rationale Distillation to Cost-Aware LLM Supervision"
date: 2026-07-01
phase: 3
status: pivot_review
plan: ../../plans/260619-reasoning-rationale-distillation/plan.md
tags: [entity-resolution, llm, rationale-distillation, low-label, pivot, cost-aware-supervision]
---

# Phase 03 Pivot Reflection

## Context

Phase 03 tested the central rationale-distillation question:

> Are structured, validated, attribute-grounded rationales actually better than free-text explanations or labels only when labels are scarce?

The current structured-rationale result does not look promising. On WDC Products with 128 low-label examples, the structured-rationale FLAN-T5-base model improved recall but badly hurt precision and macro performance. The model strongly overpredicted `match`, even though the rationale-covered training subset was roughly balanced. This suggests the problem is not simply class imbalance in the 122 examples, but a train/evaluation prior mismatch plus weak decision calibration.

Current validation summary:

| Variant | Train examples | Precision | Recall | Same-F1 | Macro-F1 | Accuracy | Predicted match rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| label_only | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 | 30.4% |
| structured_rationale | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 | 54.5% |

The important interpretation is blunt: the current rationale target is not giving a clean label-efficiency gain. It may be adding output complexity and recall bias rather than useful supervision.

## Literature Pressure

The broad idea of using LLM explanations or distillation for entity matching is already crowded.

- Direct LLM matching: strong zero/few-shot performance, but costly and slow at scale.
- Explanation/rationale distillation: Wadhwa et al. already distill natural-language explanations into smaller conditional-generation EM models.
- LLM-as-labeler workflows: recent work studies using LLMs to label ER training data, then training smaller matchers such as Ditto/RoBERTa/MiniLM.
- Knowledge distillation: DistillER frames LLM teacher to smaller ER student distillation directly.
- Benchmark realism: newer ER work criticizes standard benchmarks for balanced labels, closed entity sets, and unrealistic candidate-pair construction.

So the thesis should not be framed as simply “LLM rationales improve ER.” That sounds too close to existing work and Phase 03 does not currently support it.

## Pivot Options Discussed

### Pivot 1: Change Dataset

Changing dataset is only safe if it changes the scientific question, not if it looks like searching for a dataset where rationales happen to work.

Weak version:

> Try another dataset where structured rationales perform better.

Stronger version:

> Study when LLM-assisted ER methods fail under realistic dataset conditions such as class imbalance, unseen entities, hard negatives, and low annotation budgets.

Recommended dataset path:

- Keep WDC Products first, but use harder/unseen/corner-case slices more deliberately.
- Add standard benchmarks only for comparison: Abt-Buy, Walmart-Amazon, Amazon-Google, DBLP-ACM, DBLP-Scholar.
- Consider realistic/open-environment benchmarks if the thesis becomes about benchmark realism and failure modes.
- Avoid building a custom Vietnamese/e-commerce dataset unless there is enough time, because dataset construction can become the thesis itself.

### Pivot 2: Make ER Cheaper at Similar Effectiveness

This looks more promising than continuing rationale distillation.

Weak version:

> Use LLM labels to train a small ER model cheaply.

This overlaps with DistillER and recent LLM-labeling-for-EM work.

Stronger version:

> Under a fixed LLM labeling budget, which pair-selection strategy gives the best compact ER student?

Possible strategies:

| Strategy | Meaning |
|---|---|
| Random pair labeling | Basic baseline |
| Balanced labeling | Force match/non-match balance |
| Prior-matched labeling | Match validation/deployment prior, e.g. 20% match / 80% non-match |
| Hard-negative selection | Ask LLM about similar-looking non-matches |
| Uncertainty selection | Train a cheap model first, send uncertain pairs to LLM |
| Diversity selection | Avoid redundant candidate pairs |
| Hybrid selection | Combine uncertainty, diversity, hard negatives, and prior control |

Metrics should include both quality and cost:

- precision, recall, same-F1, macro-F1, accuracy
- number of LLM calls
- estimated labeling cost
- cost per F1 point
- inference speed compared with direct LLM matching
- student performance compared with human-label and full-LLM-label baselines

## Recommended New Direction

The most defensible pivot is:

> Budget-aware LLM supervision for efficient entity matching.

Possible thesis titles:

- Budget-Aware LLM Labeling for Efficient Entity Matching
- Reducing Annotation Cost in Entity Matching with Selective LLM Supervision
- Cost-Aware LLM Supervision for Low-Label Entity Resolution

Revised research question:

> Can selective LLM labeling train compact ER models that match human-label or full-LLM-label baselines at lower labeling cost?

This is safer than the rationale story because Phase 03 failure becomes motivation:

> Rationale targets did not reliably improve compact students; therefore the thesis shifts from explanation distillation to cost-effective supervision.

## Immediate Next Steps

1. Freeze Phase 03 rationale results as a negative/diagnostic result instead of spending more effort polishing rationale targets.
2. Run a matched-subset label-only baseline on the same 122 rationale-covered examples only if needed to close the Phase 03 analysis cleanly.
3. Design the new pilot around LLM labeling budget, pair-selection strategy, and compact student performance.
4. Prefer WDC Products harder/unseen/corner-case settings before switching dataset.
5. Build a first comparison table with random vs prior-matched vs hard-negative/uncertainty-selected LLM-labeled examples.

## Decision

Current rationale-distillation framing is probably too weak and too crowded. The more promising thesis path is not “better rationales,” but “cheaper supervision”: use LLMs selectively to create the most useful training labels for compact ER models under strict cost budgets.
