---
date: 2026-07-24
session: qwen-screening-phase6-decision
---

# Journal: 2026-07-24 — Qwen Screening and Phase 6 Decision

## Context

Phase 5 screened Qwen3-Reranker-0.6B with LoRA on the fixed budget-128
training variants and validation set. The goal was to find a compact student
that performs well enough to support the thesis comparison before proceeding
to the full-budget study.

## What Happened

| Variant | Match F1 | Macro F1 | Accuracy |
|---------|---------:|---------:|---------:|
| `gold_random` | 0.7104 | 0.8159 | 0.8764 |
| `llm_random` | 0.7267 | 0.8283 | 0.8884 |
| `llm_active_bucketed_v1` | 0.7472 | 0.8386 | 0.8904 |
| Direct LLM matcher | 0.8734 | 0.9208 | 0.9492 |

- Active selection beat LLM-random by 0.0205 match F1, 0.0103 macro F1,
  and 0.0020 accuracy.
- Qwen predicted both classes and avoided the single-class collapse observed
  in the earlier ModernBERT diagnostic.
- Under the analytical base assumption of $1 per GPU-hour, active selection
  breaks even with direct matching at about 2,281 queries. At 2,500
  comparisons, estimated cost is $0.732 for active versus $0.796 for direct
  matching. These are sensitivity assumptions, not observed provider charges.

## Reflection

Qwen passes model screening: all three student variants exceed 0.81 macro F1,
and active selection produces the best observed student result. This is enough
to pause architecture search and continue the planned study. It is not yet
enough to claim that active selection is reliably superior. The result uses
only training seed 42, and the paired-bootstrap 95% interval for the active
minus LLM-random macro-F1 difference is approximately `[-0.0045, 0.0247]`,
which includes zero.

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Pass Qwen3-Reranker-0.6B screening and pause model search | Strong validation quality with no prediction collapse | Use Qwen as the student for the next planned phase |
| Defer multi-seed replication | The user wants to proceed to Phase 6 first | Do not present the current active advantage as stable or conclusive |
| Preserve the test split | Phase 6 protocol and gate are not set yet | No test evaluation before the next-phase design is fixed |

## Next Steps

- Define and predeclare the Phase 6 Full Budget Study protocol and decision
  gate before touching the test split.
- Later, repeat `llm_random` and `llm_active_bucketed_v1` on the fixed target
  sets with multiple training seeds; this requires no new teacher calls.
- Optionally replicate selection seeds afterward. This stronger check requires
  new selected pairs and new teacher labels.
