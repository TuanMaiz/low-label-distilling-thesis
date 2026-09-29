---
title: Uncertainty-aware label-refinement foundational plan
date: 2026-09-28
summary: Created a living foundation that extends the full-label ER plan with gold-free detection, prune/relabel, tiered execution, and hidden-gold evaluation.
---

## What happened

Created
`plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/`
as the living foundation for the revised thesis pipeline. The plan preserves
the existing three-dataset, three-student full-label comparison while adding
three teacher passes, majority labels, three-fold OOF evidence, LLM
consistency, Confident Learning, teacher-student disagreement, prune/relabel
treatments, random/oracle controls, post-hoc error analysis, and cost analysis.

## Decision

The new plan is not yet the frozen execution authority. Phase 1 must resolve
thirteen open scientific and workload decisions before any new paid labeling,
OOF GPU training, expanded main-model runs, or final-test access.

The recommended starting design is one designated detector student, Confident
Learning count plus disagreement ranking plus consistency stratification, and
designated-student-only controls and ablations. These recommendations remain
explicitly unfrozen until review.

## Validation

`ak plan validate` passed and the plan store was reindexed successfully. The
current WDC-Qwen one-pass result remains a pilot if three-pass majority labels
become the official LLM training source.

## Next steps

Review Phase 1 decisions individually, freeze the experiment contract, estimate
the final workload and budget, then extend the foundation with executable phases
through the plan CLI.

## Status

This entry records planning work only. It does not authorize paid labeling,
OOF training, expanded model runs, or final-test access. AgentWiki publish was
not requested and was skipped.

## Teacher-call phase extension — 2026-09-28

Added Phase 2, `Run Three Separate Teacher Passes`, to the foundational plan.
The immediate scope is WDC: retain the completed Sol-high artifact as
`pass_01`, then run two complete fresh calls as `pass_02` and `pass_03` in
separate output directories.

The current WDC runner always reuses the 300 screening responses, which would
artificially inflate consistency if left unchanged. The phase therefore permits
one minimal code change: add an explicit `--no-reuse-existing` option that omits
the reuse paths already supported as optional by `run_setting(...)`. No new
multipass orchestration framework is planned.

Each paid pass requires its own dry run, current-pricing review, explicit
approval, spend ceiling, completion review, and 2,500-pair alignment check.
Pass 3 remains blocked until pass 2 verifies. Majority voting and consistency
calculation are deliberately deferred to a later offline phase.
