---
title: "Handoff Context And Next Step"
date: 2026-07-06
status: handoff
branch: codex/distiller-wdc-implementation
tags: [entity-resolution, wdc-products, llm-labeling, distillation, handoff]
---

# Handoff Context And Next Step

## Context

The thesis has pivoted away from structured-rationale distillation. The old
Phase 03 WDC result showed that structured rationales increased recall but hurt
precision and overall quality:

| Variant | Train rows | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `label_only` | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Current thesis question:

> Can compact Entity Matching students distilled from LLM-generated teacher
> labels approach gold-label supervised students while being cheaper at
> inference time than using the LLM directly as the matcher?

The safe framing is not "we invented LLM-label distillation for ER." DistillER
and related work already exist. The safe contribution is a controlled WDC
Products study with explicit cost accounting, gold-label student comparison,
direct LLM inference baseline, LLM-label distilled student, and failure
analysis.

## Current Branch State

Branch: `codex/distiller-wdc-implementation`.

Latest committed recovery point before cleanup:

`9e4a78a docs(thesis): add distiller WDC plans and agent skills`

After that commit, the branch has uncommitted cleanup changes:

- Updated guidance to the DistillER/WDC direction.
- Removed old implementation paths: `legacy/`, FEBRL loaders/baselines,
  structured-rationale code/config/scripts/caches/targets, old multilingual
  model abstractions, and superseded flat plan drafts.
- Kept active WDC code, current plans, papers, custom skills, journals, cached
  WDC serialized/low-label/gold-label targets, seq2seq student training/eval,
  and binary ER metrics.
- Added `supervision/` as the active namespace for supervision artifacts.
- Added `models/seq2seq_student.py` as the compact student helper replacing the
  old `models/mt5_student.py` name.

Important new/active files:

- `supervision/build_targets.py`
- `models/seq2seq_student.py`
- `experiments/train_mt5.py`
- `experiments/evaluate_student.py`
- `data/er_dataset_loader.py`
- `data/low_label_sampler.py`
- `data/serialize_pairs.py`
- `plans/260704-distiller-wdc-agent-execution/plan.md`
- `plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`

## Verification

The cleanup was verified with:

```bash
.venv/bin/python -m unittest discover -s tests
```

Result:

```text
Ran 9 tests in 37.606s
OK
```

Other checks passed:

- `ck plan validate plans/260704-distiller-wdc-agent-execution/plan.md`
  reported `0 errors, 0 warnings`.
- `git diff --check` was clean.
- Stale import scan found no imports pointing to deleted packages.

## Next Step

First, commit the cleanup so the branch has a stable recovery point.

Suggested commit:

```bash
git add -A
git commit -m "chore: prune repo to active WDC distillation path"
```

Then start Phase 3:

1. Create `supervision/llm_providers.py` for OpenRouter chat completions.
2. Create `supervision/prompts.py` for answer-only EM prompts.
3. Create `supervision/teacher_label_schema.py` for validated labels and cost
   metadata.
4. Create `supervision/direct_llm_matcher.py` to classify a fixed validation
   split or fixed validation sample and log token/cost fields.
5. Create `supervision/generate_teacher_labels.py` to label `train_128` first.
6. Create `supervision/validate_teacher_labels.py` for cache validation.

Anti-cherry-pick rule: decide the evaluation sample, prompt version, model
slug, temperature, and output paths before inspecting direct LLM results.
