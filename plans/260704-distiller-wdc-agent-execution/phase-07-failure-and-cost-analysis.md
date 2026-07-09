---
phase: 7
title: "Failure And Cost Analysis"
status: pending
priority: P1
effort: "1 week"
dependencies: [5, 6]
---

# Phase 7: Failure And Cost Analysis

## Overview

Explain when actively selected LLM-generated labels help, fail, or shift model
behavior, and whether selected-label distillation is cheaper than direct LLM
matching at useful inference volumes. This phase carries the thesis lens: cost
break-even, low-label budget behavior, data selection, WDC difficulty slices,
and teacher-noise inheritance.

## Requirements

- Functional: analyze teacher errors, student errors, false positives, and false negatives.
- Functional: compare random and active selection by selected-pair composition, teacher noise, and student error.
- Functional: compute cost summaries for each labeling budget and for direct LLM matching.
- Functional: report WDC difficulty slices, not only aggregate metrics.
- Functional: separate teacher-noise behavior from student-learning behavior.
- Non-functional: produce small human-readable samples suitable for thesis tables.

## Architecture

```text
selection manifests + direct LLM predictions + teacher-label cache + student predictions + gold labels
  -> error join by pair_id
  -> selection strategy comparison
  -> error category analysis
  -> direct-vs-distilled cost aggregation
  -> thesis tables and examples
```

## Related Code Files

- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/analysis/analyze_failures.py`
- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/analysis/cost_summary.py`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/**/predictions.jsonl`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/direct_llm/*.predictions.jsonl`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/teacher_labels/*.jsonl`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/selection/*.jsonl`
- Outputs: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/analysis/`

## Implementation Steps

1. Join selection manifests, direct LLM predictions, student predictions, gold labels, and teacher labels by pair ID where applicable.
2. Compute confusion matrices per variant and budget.
3. Compare selected-pair composition by strategy:
   - match/non-match balance where gold labels are available for analysis.
   - average title/token overlap.
   - missing-attribute rates.
   - hard-negative or near-duplicate indicators where available.
4. Compare teacher-vs-gold disagreement by selection strategy.
5. Sample false positives and false negatives.
6. Categorize errors by visible product attributes:
   - title overlap.
   - brand conflict.
   - model number conflict.
   - missing key attribute.
   - near-duplicate accessories or bundles.
   - hard negative flag where available.
   - long description or noisy description.
   - price/currency mismatch.
7. Compare teacher label errors with student errors:
   - teacher wrong, student follows.
   - teacher wrong, student corrects.
   - teacher correct, student fails.
8. Compute cost summary:
   - direct LLM inference cost by evaluated split/sample.
   - teacher labeling cost by budget and selection strategy.
   - cost per valid teacher label.
   - active-selection cost if measurable, otherwise mark as local preprocessing cost.
   - student inference cost assumption.
   - break-even query count where distillation becomes cheaper than direct LLM matching.
   - approximate cost saved versus human labeling assumption, if a human-cost assumption is used.
9. Export two thesis-ready tables:
   - representative failure cases.
   - cost by budget and variant.
10. Export one selection-level summary table:
   - budget.
   - selection strategy.
   - teacher disagreement rate.
   - student match F1.
   - gain or loss versus `llm_random`.
   - total teacher-labeling cost.
11. Export one slice-level summary table:
   - slice name.
   - support count.
   - teacher disagreement rate.
   - student error rate.
   - direct LLM error rate where available.
   - dominant false-positive or false-negative pattern.

## Success Criteria

- [ ] Failure-analysis CSV exists.
- [ ] Cost-summary CSV exists.
- [ ] Selection-strategy summary exists.
- [ ] Direct LLM matching cost is separated from teacher-labeling cost.
- [ ] Break-even estimate exists or is explicitly marked unavailable.
- [ ] WDC slice-level summary exists or missing-slice fields are explicitly documented.
- [ ] Teacher-noise inheritance summary exists.
- [ ] At least one compact qualitative table is ready for Chapter 5.
- [ ] The thesis can explain precision/recall changes.

## Risk Assessment

- Risk: failure categories are subjective.
  Mitigation: keep categories simple and tie each to visible attributes.
- Risk: active selection looks better only because it changes class balance.
  Mitigation: report selected-pair composition and compare precision/recall, not only F1.
- Risk: cost assumptions are challenged.
  Mitigation: separate measured API cost from hypothetical human-labeling cost.
