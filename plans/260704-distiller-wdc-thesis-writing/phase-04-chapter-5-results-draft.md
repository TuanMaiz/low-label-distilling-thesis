---
phase: 4
title: "Chapter 5 Results Draft"
status: pending
priority: P1
effort: "1-2 weeks"
dependencies: [3]
---

# Phase 4: Chapter 5 Results Draft

## Overview

Write the core evidence chapter after pilot and full-budget metrics exist.

## Requirements

- Functional: report active-vs-random label selection, label-efficiency, direct LLM inference cost, distillation cost-efficiency, and failure-analysis results.
- Non-functional: discuss both positive and negative findings.
- Dependency: needs agent execution Phases 5-7.

## Architecture

Suggested Chapter 5 logic:

```text
pilot result
  -> active-vs-random selection result
  -> direct LLM matcher result
  -> full budget result
  -> cost result
  -> failure analysis
  -> comparison with related work
  -> limitations
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/summary/*.csv`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/figures/*.png`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/analysis/*.csv`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/thesis_artifact_index.md`

## Implementation Steps

1. Write the pilot result subsection first.
2. Add main label-budget table.
3. Add label-efficiency figure.
4. Compare `llm_random` vs `llm_active_hybrid` at the same budget.
5. Compare `gold_random` student vs LLM-label students for quality.
6. Compare `direct_llm_matcher` vs selected-label distilled student for repeated inference cost.
7. Add mixed-label results if used.
8. Add cost table and break-even estimate.
9. Add selection-strategy and failure-analysis tables.
10. Discuss what result supports the thesis.
11. Discuss what result weakens the thesis.
12. Compare cautiously with DistillER, active ER, and LLM-labeling related work.
13. Write limitations:
    - one main dataset.
    - one or few teacher models.
    - cost depends on provider.
    - teacher labels may encode hidden bias.

## Success Criteria

- [ ] Main result table exists.
- [ ] Main plot exists.
- [ ] Active-vs-random same-budget comparison exists.
- [ ] Cost table exists.
- [ ] Direct LLM matching appears as cost baseline.
- [ ] Failure examples exist.
- [ ] Chapter 5 has a clear answer to the research question.

## Risk Assessment

- Risk: results are not positive.
  Mitigation: write a valid negative/diagnostic thesis around when LLM labels fail and why.
- Risk: result chapter overclaims.
  Mitigation: every claim must point to a table, figure, or failure sample.
