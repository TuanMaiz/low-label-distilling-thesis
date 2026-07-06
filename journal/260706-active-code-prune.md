---
title: "Active Code Prune For DistillER/WDC Branch"
date: 2026-07-06
phase: cleanup
status: completed
plan: ../plans/260704-distiller-wdc-agent-execution/plan.md
tags: [entity-resolution, wdc-products, cleanup, distillation]
---

# Active Code Prune For DistillER/WDC Branch

## Context

After committing the custom skills and planning files, the active branch still
contained old implementation paths from prior pivots: Wikidata/name matching,
mBART generation, FEBRL baselines, and structured-rationale teacher code. Those
files made the repository harder to read and increased the chance that future
work would follow stale instructions.

## Decision

Remove old pivot code from the active tree and keep only what supports the
current WDC Products cost-aware LLM-label distillation path.

Kept:

- WDC Products loading, low-label sampling, and serialization.
- Cached WDC serialized/low-label/gold-label target artifacts.
- Seq2seq student training and evaluation.
- Binary ER metrics.
- Current DistillER/WDC execution and thesis plans.
- Research papers, journals, and custom agent skills.

Removed:

- `legacy/` Wikidata, multilingual-name, mBART, and old notebook code.
- FEBRL loaders and FEBRL baseline runners.
- Structured-rationale package, configs, scripts, cached rationales, and
  structured-rationale targets.
- Superseded flat pivot plan drafts.
- Old model/base abstractions from the multilingual-name path.

## Follow-up

The active supervision namespace is now `supervision/`. Phase 3 should add
answer-only LLM provider, prompt, direct matcher, teacher-label schema,
generation, and validation modules there.
