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

Explain when LLM-generated labels help, fail, or shift model behavior, and whether distillation is cheaper than direct LLM matching at useful inference volumes.

## Requirements

- Functional: analyze teacher errors, student errors, false positives, and false negatives.
- Functional: compute cost summaries for each labeling budget and for direct LLM matching.
- Non-functional: produce small human-readable samples suitable for thesis tables.

## Architecture

```text
direct LLM predictions + teacher-label cache + student predictions + gold labels
  -> error join by pair_id
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
- Outputs: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/analysis/`

## Implementation Steps

1. Join direct LLM predictions, student predictions, gold labels, and teacher labels by pair ID where applicable.
2. Compute confusion matrices per variant and budget.
3. Sample false positives and false negatives.
4. Categorize errors by visible product attributes:
   - title overlap.
   - brand conflict.
   - model number conflict.
   - missing key attribute.
   - near-duplicate accessories or bundles.
5. Compare teacher label errors with student errors:
   - teacher wrong, student follows.
   - teacher wrong, student corrects.
   - teacher correct, student fails.
6. Compute cost summary:
   - direct LLM inference cost by evaluated split/sample.
   - teacher labeling cost by budget.
   - cost per valid teacher label.
   - student inference cost assumption.
   - break-even query count where distillation becomes cheaper than direct LLM matching.
   - approximate cost saved versus human labeling assumption, if a human-cost assumption is used.
7. Export two thesis-ready tables:
   - representative failure cases.
   - cost by budget and variant.

## Success Criteria

- [ ] Failure-analysis CSV exists.
- [ ] Cost-summary CSV exists.
- [ ] Direct LLM matching cost is separated from teacher-labeling cost.
- [ ] Break-even estimate exists or is explicitly marked unavailable.
- [ ] At least one compact qualitative table is ready for Chapter 5.
- [ ] The thesis can explain precision/recall changes.

## Risk Assessment

- Risk: failure categories are subjective.
  Mitigation: keep categories simple and tie each to visible attributes.
- Risk: cost assumptions are challenged.
  Mitigation: separate measured API cost from hypothetical human-labeling cost.
