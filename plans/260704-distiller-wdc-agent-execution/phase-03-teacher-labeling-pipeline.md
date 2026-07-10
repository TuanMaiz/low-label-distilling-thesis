---
phase: 3
title: "Teacher Labeling, Active Selection, And Direct LLM Baseline"
status: completed
priority: P1
effort: "3-5 days"
dependencies: [1]
---

# Phase 3: Teacher Labeling, Active Selection, And Direct LLM Baseline

## Overview

Build answer-only LLM labeling for WDC training pairs, fixed selection
manifests for random and active low-label budgets, and a direct LLM matcher
baseline on fixed evaluation pairs, with validation, caching, prompt
versioning, and cost logging.

## Requirements

- Functional: generate `match` or `non_match` labels for sampled training pairs.
- Functional: write selected-pair manifests before teacher labels or results are inspected.
- Functional: support at least `random` and one active selection strategy for the pilot.
- Functional: generate direct LLM predictions for a fixed validation/test set or predeclared sample.
- Functional: cache every teacher response with metadata and validation status.
- Non-functional: deterministic resume behavior; repeated runs must not re-label cached valid rows unless forced.
- Cost: store enough token and price metadata to compute both teacher-labeling cost and direct LLM inference cost.

## Architecture

```text
serialized train pairs
  -> random / active selection manifest
  -> prompt builder
  -> teacher provider
  -> raw response cache
  -> label parser
  -> validated teacher-label JSONL
  -> cost summary

serialized validation/test pairs
  -> same answer-only prompt family
  -> teacher provider
  -> raw response cache
  -> label parser
  -> direct LLM prediction JSONL
  -> direct inference cost summary
```

## Cache Schema

Each JSONL row should contain at least:

```json
{
  "pair_id": "string",
  "dataset": "wdc_products",
  "budget": 128,
  "selection_strategy": "random",
  "selection_rank": 1,
  "selection_score": null,
  "teacher_model": "provider/model",
  "prompt_version": "answer_only_v1",
  "raw_answer": "match",
  "label": "match",
  "valid": true,
  "input_tokens": 0,
  "output_tokens": 0,
  "estimated_cost_usd": 0.0,
  "mode": "teacher_label",
  "created_at": "ISO-8601"
}
```

For direct LLM matching, use the same schema with `mode: "direct_prediction"` and include `gold_label` so direct LLM quality can be evaluated.

## Related Code Files

- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/llm_providers.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/config.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/prompts.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/teacher_label_schema.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/generate_teacher_labels.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/direct_llm_matcher.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/supervision/validate_teacher_labels.py`
- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/data/low_label_sampler.py`
- Optional create: `/mnt/d/Study/Cao-hoc/luan-van/code/data/select_active_pairs.py`
- Create or modify: `/mnt/d/Study/Cao-hoc/luan-van/code/analysis/cost_summary.py`
- Add tests: `/mnt/d/Study/Cao-hoc/luan-van/code/tests/test_teacher_labels.py`

## Implementation Steps

1. Define `TeacherLabel`, `DirectLLMPrediction`, and shared cache-row schemas.
2. Add an answer-only prompt:
   - include both records.
   - ask for exactly one label.
   - forbid explanations in the first version.
3. Implement parser accepting only canonical outputs:
   - `match`
   - `non_match`
4. Add selected-pair manifests:
   - preserve the existing random balanced low-label sample.
   - add the first active manifest, `llm_active_bucketed_v1`.
   - default to 25 percent each for `easy_match_candidate`,
     `hard_match_candidate`, `easy_non_match_candidate`, and
     `hard_negative_candidate`.
   - record `selection_strategy`, `selection_bucket`, `budget`, `seed`, `rank`,
     bucket rank/quota, and score fields.
5. Add teacher-label generator CLI:
   - `--pairs`
   - `--output`
   - `--model`
   - `--prompt-version`
   - `--selection-strategy`
   - `--limit`
   - `--resume`
   - `--seed`
6. Add direct LLM matcher CLI:
   - `--input`
   - `--output`
   - `--model`
   - `--prompt-version`
   - `--limit`
   - `--sample-seed`
   - `--resume`
7. Add validator CLI that reports:
   - valid count.
   - invalid count.
   - duplicate pair IDs.
   - label distribution.
   - estimated total cost.
   - selection-strategy distribution where available.
8. Add unit tests for prompt construction, parser behavior, duplicate handling, selection metadata, and cost aggregation.
9. Generate pilot teacher-label caches for `train_128.random` and `train_128.llm_active_bucketed_v1`.
10. Generate `train_256` caches only after the sampler/selector creates a fixed `256` manifest.
11. Generate direct LLM predictions and cost logs for the fixed validation set or predeclared validation sample.

## Implementation Status

Updated 2026-07-07:

- Code scaffolding for steps 1-8 is implemented.
- Added strict answer-only schemas, prompt/parser helpers, OpenRouter provider,
  teacher-label generation CLI, direct LLM matcher CLI, cache validator, and
  reusable cost summaries.
- Added fixed selection-manifest generation in `data/select_active_pairs.py`,
  including the current `llm_active_bucketed_v1` selector and the older
  `llm_active_hybrid` pilot selector.
- Added cache/schema fields for `selection_strategy`, `selection_rank`,
  `selection_score`, `selection_seed`, `selection_uses_gold_label`,
  `selection_bucket`, `selection_bucket_rank`, and `selection_bucket_quota`.
- Unit tests cover prompt construction, strict parsing, resume behavior,
  invalid-output routing, duplicate detection, fixed direct-evaluation sampling,
  direct cost-summary writing, selection metadata, manifest creation, and schema
  validation.
- Local manifests now exist:
  - `data/cache/wdc_products/selection_manifests/train_128.random.jsonl`
  - `data/cache/wdc_products/selection_manifests/train_128.llm_active_bucketed_v1.jsonl`
- Live OpenRouter generation has been run for the 128-budget pilot with
  `openai/gpt-5.4-mini`: random and active teacher-label caches exist, and the
  full validation direct prediction artifact exists.
- The `llm_active_bucketed_v1` manifest is label-free for scoring. Gold labels
  are retained only for audit/evaluation. Its default budget allocation is
  32/32/32/32 for the four candidate buckets at budget 128.

Predeclared first-run defaults before inspecting results:

| Field | Value |
|---|---|
| Prompt version | `answer_only_v1` |
| Provider | `openrouter` |
| Default model slug | `openai/gpt-5.4-mini` (`openrouter:openai/gpt-5.4-mini` in cache rows) |
| Temperature | `0.0` |
| Teacher-label first budget | `train_128` |
| First selection strategy | `random` |
| First active strategy | `llm_active_bucketed_v1`, after fixed manifest creation |
| Active bucket ratios | `0.25` each for easy-match, hard-match, easy-non-match, hard-negative candidates |
| Teacher-label output | `data/cache/wdc_products/teacher_labels/train_128.random.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl` |
| Teacher-label rejects | `data/cache/wdc_products/teacher_labels/train_128.random.openrouter.openai-gpt-5-4-mini.answer_only_v1.rejects.jsonl` |
| Active teacher-label output | `data/cache/wdc_products/teacher_labels/train_128.llm_active_bucketed_v1.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl` |
| Direct-eval default | full `validation` split unless cost forces a predeclared `--limit N --sample-seed 42` sample |
| Direct-eval output | `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.predictions.jsonl` |
| Direct-eval cost summary | `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json` |

## Success Criteria

- [x] `train_128.random` selection manifest exists.
- [x] `train_128.llm_active_bucketed_v1` selection manifest exists if active pilot is approved.
- [x] `train_128.random` teacher-label cache exists.
- [x] `train_128.llm_active_bucketed_v1` teacher-label cache exists if active pilot is approved.
- [ ] `train_256` teacher-label cache exists if the sampler/selector is extended.
- [x] Direct LLM prediction cache exists for the fixed evaluation set or sample.
- [x] Invalid teacher output rate is reported on live cache.
- [x] Label distribution is reported on live cache.
- [x] Estimated teacher-labeling cost per budget is reported on live cache.
- [x] Estimated direct LLM inference cost per evaluated pair is reported on live cache.
- [x] Unit tests pass.

## Risk Assessment

- Risk: teacher outputs explanations or extra text.
  Mitigation: strict parser plus retry or invalid-row tracking.
- Risk: teacher labels are biased toward `match`.
  Mitigation: report label distribution and inspect false positives in Phase 7.
- Risk: active selection picks examples the teacher cannot label reliably.
  Mitigation: compare teacher-vs-gold disagreement by selection strategy before
  judging the student.
- Risk: active bucket scoring still contains hand-designed feature scores.
  Mitigation: make the main ratio defensible with predeclared equal 25 percent
  bucket quotas, keep ratio parameters explicit, frame bucket scores as
  transparent heuristics, and revisit selector ablations before final thesis
  claims. See `REVISIT.md`.
- Risk: costs are undercounted.
  Mitigation: store raw token metadata when provider returns it; otherwise store documented estimate.
