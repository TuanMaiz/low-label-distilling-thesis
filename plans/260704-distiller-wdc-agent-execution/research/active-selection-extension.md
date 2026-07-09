---
title: "Active LLM Selection Extension"
status: active
created: 2026-07-07
plan: "/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/plan.md"
---

# Active LLM Selection Extension

## Purpose

This note records the thesis pivot from plain random LLM-label distillation to
cost-aware active LLM labeling under low-label budgets.

The key question is not only whether LLM labels can train a compact Entity
Matching student. The sharper question is:

> If LLM labels are scarce, which WDC product pairs are worth spending them on?

## Research Question

Primary question:

> Under low-label budgets for WDC Products Entity Matching, can active selection
> of LLM-labeled training pairs produce compact students that are more
> cost-effective than random LLM-label distillation and repeated direct LLM
> matching?

Secondary questions:

- Which selection strategy works best under the same LLM-label budget: random,
  uncertainty, diversity, hard-negative, or hybrid selection?
- Do actively selected hard pairs improve student learning, or do they mainly
  expose cases where the LLM teacher is noisy?
- How much quality is gained per LLM-labeling dollar compared with random
  selection?
- Which WDC difficulty slices benefit most from active selection?

## Experiment Shape

Keep the first experiment small:

| Variant | Selection | Label Source | Purpose |
|---|---|---|---|
| `gold_random` | random balanced sample | dataset gold labels | trusted low-budget quality context |
| `llm_random` | random balanced sample | LLM teacher labels | random distillation control |
| `llm_active_bucketed_v1` | 25 percent each from easy-match, hard-match, easy-non-match, and hard-negative candidate buckets | LLM teacher labels | first active-selection test |
| `direct_llm_matcher` | fixed validation/test pairs | LLM predictions | repeated inference-cost baseline |

If the first active pilot works, expand to:

- `llm_active_uncertainty`
- `llm_active_diversity`
- `llm_active_hard_negative`
- `llm_active_hybrid`
- `mixed_gold_llm_active`

## Anti-Cherry-Pick Rules

- Selection manifests must be written before teacher labels or student results
  are inspected.
- Validation/test gold labels must not influence selection.
- Every selected row must preserve `pair_id`, `selection_strategy`, `budget`,
  `seed`, `rank`, and any selection score.
- Compare strategies at the same budget, same teacher model, same prompt
  version, same student model, and same validation split.

## Thesis Contribution Wording

Safe contribution:

> This thesis studies cost-aware active LLM labeling for low-budget product
> Entity Matching, comparing random and actively selected teacher-label budgets
> through compact student performance, teacher-noise behavior, failure slices,
> and break-even cost against direct LLM matching.

Avoid claiming:

- active learning for ER is new.
- LLM-label distillation for ER is new.
- the proposed strategy is generally superior across all ER datasets.
- WDC results transfer without additional evidence.

## First Decision Gate

Continue if `llm_active_bucketed_v1` at budget `128` is better than
`llm_random` on match F1 or gives a clear diagnostic story about teacher noise
and WDC failure slices.

Revise if the active strategy improves recall but damages precision, or selects
many teacher-wrong examples.

Stop this extension if active selection is no better than random selection and
does not produce a useful failure-analysis story.
