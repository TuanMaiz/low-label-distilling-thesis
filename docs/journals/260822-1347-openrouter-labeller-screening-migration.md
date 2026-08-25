---
date: 2026-08-22
session: openrouter-labeller-screening-migration
---

# Journal: 2026-08-22 — OpenRouter Labeller Screening Migration

## Context

Phase 1 needs to screen three GPT-5.6 Sol configurations on the same frozen,
blinded 300-pair WDC training sample. The screening prototype had a separate
OpenAI transport even though the repository already had an OpenRouter provider
client and cost-accounting conventions.

## What Happened

- Extended `OpenRouterAnswerOnlyClient` with a raw Chat Completions path while
  preserving the normalized answer-only interface used by existing teacher and
  direct-matching workflows.
- Migrated screening to `OPENROUTER_API_KEY` and removed its dedicated OpenAI
  client. Requests now carry strict JSON-schema output, explicit reasoning
  effort, and OpenAI-only provider routing with fallbacks disabled.
- Froze `sol_high` and `sol_max` to `openai/gpt-5.6-sol`, and `sol_pro_max` to
  `openai/gpt-5.6-sol-pro`.
- Preserved provider-reported charged cost when available, conservative cost
  reserves when usage is missing or ambiguous, and a hard per-setting spend
  ceiling. Later reservations also honor the largest observed attempt cost.
- Added typed retryable HTTP and ambiguous transport failures, numeric
  `Retry-After` support, resumable attempt journals, and hashes for both the
  runner and shared provider client.
- Kept the model-facing sample fully blinded; gold remains comparison-only and
  final predictions remain `pair_id,result`.

## Reflection

Reusing one provider client reduces transport drift and lets screening inherit
the repository's established response provenance and cost patterns. Provider
pinning is important for a fair screening comparison: an OpenRouter model slug
alone does not guarantee that every request uses the same upstream provider.
The explicit paid-run flag and spend ceiling remain necessary even with routing
price limits because retries and incomplete usage metadata can otherwise make
local accounting optimistic.

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Use the shared OpenRouter client | Avoid duplicate transport and accounting logic | Screening and production supervision share one provider boundary |
| Pin OpenAI upstream and disable fallbacks | Keep model/provider conditions comparable | Requests fail closed instead of silently changing provider |
| Require structured answer-only output | Prevent rationale leakage and simplify validation | Only `match` or `non_match` is accepted |
| Prefer charged cost, reserve conservatively otherwise | Protect the approved budget under incomplete or anomalous usage | A run stops before its next request could exceed the ceiling |

## Verification

- `tests.test_llm_providers`: 3 tests passed.
- `labeller-screening/tests`: 7 tests passed.
- No paid OpenRouter calls were made; all verification used local fakes and
  mocked transport.

## Next Steps

- Manually review the frozen sample, prompt, current OpenRouter pricing, and a
  positive per-setting spend ceiling.
- Run the three paid settings separately only after supplying
  `--confirm-paid-screening`.
- Compare complete predictions against gold by `pair_id`, then record the human
  labeler selection in the Phase-1 experiment contract.
