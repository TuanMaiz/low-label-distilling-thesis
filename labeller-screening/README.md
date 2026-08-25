# Labeller screening

This folder contains the completed WDC labeler screening and its production
runner. Screening froze one deterministic random sample of 300 WDC training
pairs, ran the same blinded sample through three GPT-5.6 Sol settings via
OpenRouter, and compared answer-only predictions with private gold labels by
`pair_id`.

## Status

- All three paid screening settings are complete.
- `sol_high` was selected for its accuracy/cost balance.
- Full WDC training labeling is complete: 2,500/2,500 valid labels.
- The production artifact reused 300 verified screening labels and made 2,200
  new calls, with zero invalid results, zero retries, and USD 2.693225 total
  cost under the frozen USD 5 ceiling.
- Final labels:
  `../data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv`.

Read `plan.md` before running anything. Preparing the sample and dry runs are
offline. Actual API calls require `OPENROUTER_API_KEY` and the explicit
`--confirm-paid-screening` flag plus a reviewed positive
`--spend-ceiling-usd`. Run each setting separately so its 300-pair run cost and
attempt journal are independently visible.

OpenRouter is pinned to the OpenAI upstream provider with provider and model
fallbacks disabled: OpenRouter is the API gateway, while OpenAI is the provider
that serves the requested model. `sol_high` and `sol_max` use
`openai/gpt-5.6-sol`; `sol_pro_max` uses
`openai/gpt-5.6-sol-pro`. The runner records OpenRouter's returned usage and
actual charged cost when available.

The model-facing file contains only `pair_id` and `input_text`. Do not merge the
gold CSV into it. Final result files contain exactly `pair_id,result`; audit and
usage provenance are stored separately.

`run_full_wdc.py` is resumable, but the authorized WDC labeling run is already
complete. Do not start a new paid WDC run or call validation/test data. The
broader 3×3 experiment contract remains unfinished.
