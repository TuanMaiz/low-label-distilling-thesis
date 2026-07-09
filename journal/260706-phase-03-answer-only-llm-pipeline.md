# 2026-07-06 - Phase 3 Answer-Only LLM Pipeline

## What Changed

- Added the Phase 3 answer-only LLM supervision scaffold under `supervision/`.
- Implemented strict `match` / `non_match` parsing, Pydantic cache schemas,
  OpenRouter chat-completions provider, teacher-label generation, direct LLM
  validation/test prediction, and cache validation.
- Added reusable cost and duplicate summaries in `analysis/cost_summary.py`.
- Added unit tests for prompt construction, strict parser behavior,
  deterministic resume, invalid-output rejects, fixed direct-evaluation
  sampling, cost summary writing, and cache validation.

## Declared Defaults Before Live Results

- Prompt version: `answer_only_v1`.
- Provider/model: `openrouter:openai/gpt-4o-mini`.
- Temperature: `0.0`.
- First teacher budget: `train_128`.
- Direct baseline: full validation split by default. If cost is too high,
  declare a fixed `--limit N --sample-seed 42` sample before inspecting results.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests
```

Result: 16 tests passed.

`git diff --check` is clean.

## Next Step

Run the live OpenRouter calls for `train_128` teacher labels and the fixed
validation direct-baseline artifact, then validate both caches before Phase 4
target building.
