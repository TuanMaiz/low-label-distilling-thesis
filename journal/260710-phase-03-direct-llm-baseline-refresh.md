# Phase 3 Direct LLM Baseline Refresh

Date: 2026-07-10
Plan: `plans/260704-distiller-wdc-agent-execution/plan.md`
Tags: `wdc-products`, `direct-llm`, `cost-accounting`, `phase-03`

## Context

The first direct LLM validation baseline used `openai/gpt-4o-mini`. Its result
was plausible but weaker than strong LLM-as-matcher numbers reported for WDC
Products, mainly because match recall was low. We decided to use a stronger but
still cost-aware OpenRouter model, `openai/gpt-5.4-mini`, for the direct
matcher baseline.

## What Changed

- Updated the Phase 3 default model to `openai/gpt-5.4-mini`.
- Added model-specific artifact naming so direct prediction caches do not mix
  models.
- Exposed pass-through arguments on `scripts/run_phase03_reproducible.sh direct`
  so model, limit, input, and output can be supplied without editing config.
- Raised the default max output tokens from 8 to 16 because `gpt-5.4-mini`
  rejects values below 16.
- Added thesis-writing notes to
  `plans/260704-distiller-wdc-agent-execution/reports/phase-03-flow-io-report.md`
  explaining metric provenance and contextual comparison to prior work.

## Direct Baseline Result

Clean validation artifact:

- `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.predictions.jsonl`
- `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json`

Final result on the full WDC validation split:

| Model | Eval rows | Valid | Match precision | Match recall | Match F1 | Macro F1 | Accuracy | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `openai/gpt-4o-mini` | 2500 | 2500 | 0.8579 | 0.6880 | 0.7636 | 0.8558 | 0.9148 | `$0.1493` |
| `openai/gpt-5.4-mini` | 2500 | 2500 | 0.8708 | 0.8760 | 0.8734 | 0.9208 | 0.9492 | `$0.7965` |

Confusion matrix for the clean `gpt-5.4-mini` result:

```text
tp = 438
fp = 65
tn = 1935
fn = 62
```

The large improvement over `gpt-4o-mini` mainly comes from match recall:
`0.6880 -> 0.8760`.

## Retry Note

The first full `gpt-5.4-mini` run only produced 386 valid rows and 2114 invalid
rows, mostly because OpenRouter returned temporary HTTP 403 region errors. A
small retry on 10 previously failed rows succeeded, so the run was resumed from
a clean cache seeded with the 386 valid rows. The resumed run reused those 386
rows and generated the remaining 2114 rows, producing a final clean 2500-row
artifact with zero invalid predictions.

This is not a repeated-result cherry-pick: the final artifact contains one
latest valid prediction per fixed validation pair. The retry was operational
recovery from provider availability errors.

## Metric Provenance

The direct metric block in the `.cost.json` file is computed from the prediction
JSONL rows:

```text
label      = parsed direct LLM prediction
gold_label = WDC gold label copied from serialized validation row
```

`match` is treated as the positive class. Therefore:

```text
match precision = tp / (tp + fp)
match recall    = tp / (tp + fn)
match F1        = 2PR / (P + R)
accuracy        = (tp + tn) / total
macro F1        = average(match F1, non-match F1)
```

Cost is summed from per-row provider usage fields:

```text
input_tokens
output_tokens
estimated_cost_usd
```

For `gpt-5.4-mini`, the clean full-validation run used 969364 input tokens,
15439 output tokens, and an estimated total cost of `$0.7964985`.

## Prior-Work Positioning

The `gpt-5.4-mini` direct result is broadly aligned with strong LLM-as-matcher
reports on WDC Products:

- Peeters and Bizer 2023 report WDC ChatGPT direct matching around the mid/high
  0.80s F1 depending on prompt and demonstrations.
- AnyMatch 2024 reports MatchGPT GPT-4 on WDC at about 0.8583 F1 and a
  GPT-3.5-style setting around 0.7651 F1.
- Steiner and Bizer 2026 is more relevant as nearby distillation/LLM-labeling
  work than as a direct-baseline comparison; their reported WDC distilled
  student values are lower than this direct LLM baseline.

The thesis should frame this as contextual alignment, not as a leaderboard
claim, because prompts, WDC variants, split construction, and sample sizes
differ across papers.

## Next Step

Phase 5 can now use the clean `gpt-5.4-mini` direct validation baseline as the
repeated-inference quality/cost reference while training and evaluating the
128-budget compact student variants.

## Follow-Up: Teacher Labels And Targets

After confirming the direct baseline, we regenerated the 128-budget training
teacher labels with the same `openai/gpt-5.4-mini` model so Phase 5 is
methodologically cleaner.

| Cache | Rows | Valid | Rejected | LLM labels | Gold labels | Cost |
|---|---:|---:|---:|---|---|---:|
| `train_128.random.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl` | 128 | 128 | 0 | 52 match / 76 non-match | 64 match / 64 non-match | `$0.0447` |
| `train_128.llm_active_bucketed_v1.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl` | 128 | 128 | 0 | 49 match / 79 non-match | 44 match / 84 non-match | `$0.0356` |

The Phase 4 targets were rebuilt from those caches:

- `data/cache/wdc_products/targets/train_128.llm_random.openai-gpt-5-4-mini.targets.jsonl`
- `data/cache/wdc_products/targets/train_128.llm_active_bucketed_v1.openai-gpt-5-4-mini.targets.jsonl`

Stale model-less GPT-4o-mini teacher-label caches and generic LLM target files
were removed to avoid accidentally training Phase 5 students from the wrong
teacher. The old GPT-4o-mini direct baseline result remains available only as a
comparison artifact.

Validation after the refresh:

- `scripts/run_phase03_reproducible.sh validate`: passed.
- `scripts/run_phase04_targets.sh test`: passed.
- `.venv/bin/python -m unittest discover -s tests`: 21 tests passed.
