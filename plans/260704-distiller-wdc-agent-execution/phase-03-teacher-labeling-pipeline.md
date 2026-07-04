---
phase: 3
title: "Teacher Labeling And Direct LLM Baseline"
status: pending
priority: P1
effort: "3-5 days"
dependencies: [1]
---

# Phase 3: Teacher Labeling And Direct LLM Baseline

## Overview

Build answer-only LLM labeling for WDC training pairs and a direct LLM matcher baseline on fixed evaluation pairs, with validation, caching, prompt versioning, and cost logging.

## Requirements

- Functional: generate `match` or `non_match` labels for sampled training pairs.
- Functional: generate direct LLM predictions for a fixed validation/test set or predeclared sample.
- Functional: cache every teacher response with metadata and validation status.
- Non-functional: deterministic resume behavior; repeated runs must not re-label cached valid rows unless forced.
- Cost: store enough token and price metadata to compute both teacher-labeling cost and direct LLM inference cost.

## Architecture

```text
serialized train pairs
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

- Reuse: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/model_providers.py`
- Reuse/modify: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/config.py`
- Modify or split: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/prompts.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/teacher_label_schema.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/generate_teacher_labels.py`
- Create or extend: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/run_direct_llm_matcher.py`
- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/rationales/validate_teacher_labels.py`
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
4. Add teacher-label generator CLI:
   - `--pairs`
   - `--output`
   - `--model`
   - `--prompt-version`
   - `--limit`
   - `--resume`
   - `--seed`
5. Add direct LLM matcher CLI:
   - `--input`
   - `--output`
   - `--model`
   - `--prompt-version`
   - `--limit`
   - `--sample-seed`
   - `--resume`
6. Add validator CLI that reports:
   - valid count.
   - invalid count.
   - duplicate pair IDs.
   - label distribution.
   - estimated total cost.
7. Add unit tests for prompt construction, parser behavior, duplicate handling, and cost aggregation.
8. Generate pilot teacher-label caches for `train_128` and `train_256`.
9. Generate direct LLM predictions and cost logs for the fixed validation set or predeclared validation sample.

## Success Criteria

- [ ] `train_128` teacher-label cache exists.
- [ ] `train_256` teacher-label cache exists.
- [ ] Direct LLM prediction cache exists for the fixed evaluation set or sample.
- [ ] Invalid teacher output rate is reported.
- [ ] Label distribution is reported.
- [ ] Estimated teacher-labeling cost per budget is reported.
- [ ] Estimated direct LLM inference cost per evaluated pair is reported.
- [ ] Unit tests pass.

## Risk Assessment

- Risk: teacher outputs explanations or extra text.
  Mitigation: strict parser plus retry or invalid-row tracking.
- Risk: teacher labels are biased toward `match`.
  Mitigation: report label distribution and inspect false positives in Phase 7.
- Risk: costs are undercounted.
  Mitigation: store raw token metadata when provider returns it; otherwise store documented estimate.
