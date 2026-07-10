# Phase 3 Flow And Input/Output Report

## Purpose

This report explains the Phase 3 data flow for sanity checking. Phase 3 creates
fixed pair-selection manifests, sends selected or evaluation pairs through an
answer-only LLM matcher, caches valid/invalid outputs, and prepares auditable
cost/quality metadata for later target building and analysis.

Live OpenRouter calls have been completed for the 128-budget pilot with
`openai/gpt-5.4-mini`: random and active teacher-label caches exist, and the
full validation direct LLM baseline exists with token/cost/metric summaries.

## High-Level Flow

```mermaid
flowchart TD
    A["WDC serialized train rows<br/>label + target_label = gold"] --> B["Selection manifest builder"]
    B --> B1["train_128.random.jsonl<br/>balanced random control"]
    B --> B2["train_128.llm_active_bucketed_v1.jsonl<br/>4 bucket active selection"]

    B1 --> C["Teacher-label generator"]
    B2 --> C
    C --> P["Answer-only prompt builder<br/>gold label hidden"]
    P --> L["OpenRouter LLM provider"]
    L --> R["Raw LLM answer + usage/cost metadata"]
    R --> S["Strict parser<br/>match / non_match only"]
    S --> T["Teacher-label cache<br/>valid rows"]
    S --> U["Reject cache<br/>invalid/failed rows"]

    T --> V["Cache validator + cost summary"]
    U --> V

    W["WDC validation/test rows<br/>label + target_label = gold"] --> X["Direct LLM matcher"]
    X --> P
    X --> Y["Direct prediction cache"]
    Y --> Z["Direct cost + metrics summary"]

    T --> AA["Phase 4 target builder"]
    AA --> AB["llm_random targets"]
    AA --> AC["llm_active_bucketed_v1 targets"]
    Z --> AD["Phase 5/7 baseline comparison"]
    AB --> AE["Compact student training"]
    AC --> AE
    AE --> AD
```

## Flow 1: Serialized WDC Rows

| Item | Details |
|---|---|
| Input path | `data/cache/wdc_products/serialized/train.jsonl`, `validation.jsonl`, `test.jsonl` |
| Producer | Phase 1 WDC serialization code |
| Main consumer | selection builder, direct LLM matcher |
| Key fields | `pair_id`, `split`, `label`, `target_label`, `input_text`, `record_a`, `record_b`, `metadata` |
| Gold label location | `label` as `0/1`; `target_label` as `match` or `non-match` |
| LLM visibility | Gold label is not placed in the prompt |

Example conceptual row:

```json
{
  "pair_id": "a#b",
  "split": "train",
  "label": 1,
  "target_label": "match",
  "input_text": "Task: decide whether Record A and Record B...",
  "record_a": {"attributes": {"title": "..."}},
  "record_b": {"attributes": {"title": "..."}},
  "metadata": {"dataset": "wdc_products"}
}
```

Sanity checks:

- `pair_id` must be stable and unique within a split.
- Gold labels are retained for audit/evaluation, not for prompting.
- `input_text` must include both records with field names.

## Flow 2: Selection Manifest Builder

| Item | Random Control | Active Heuristic |
|---|---|---|
| Code | `data/select_active_pairs.py` | `data/select_active_pairs.py` |
| Input | `data/cache/wdc_products/low_label/train_128.jsonl` | `data/cache/wdc_products/serialized/train.jsonl` |
| Output | `data/cache/wdc_products/selection_manifests/train_128.random.jsonl` | `data/cache/wdc_products/selection_manifests/train_128.llm_active_bucketed_v1.jsonl` |
| Strategy | preserve existing balanced sample | select from four label-free WDC candidate buckets |
| Uses gold label for selection? | yes, indirectly, because existing low-label sample is stratified | no |
| Current row count | 128 | 128 |
| Default active allocation | not applicable | 32 rows each from easy-match, hard-match, easy-non-match, hard-negative candidate buckets |
| Gold-label audit | expected balanced from low-label sampler | audit after selection only; not an input to active scoring |

Added fields:

| Field | Meaning |
|---|---|
| `selection_strategy` | `random` or `llm_active_bucketed_v1` |
| `selection_rank` | 1-based rank inside the selected budget |
| `selection_score` | active score; `null` for random |
| `selection_bucket` | active-only bucket name |
| `selection_bucket_rank` | 1-based rank inside the active bucket |
| `selection_bucket_quota` | requested number of rows for that bucket |
| `selection_seed` | fixed seed, currently `42` |
| `selection_budget` | selected budget, currently `128` |
| `selection_uses_gold_label` | whether the selection procedure used gold labels |
| `selection_features` | active-only feature breakdown |

Sanity checks:

- Manifests are fixed before live LLM labels or student results are inspected.
- The active bucket ratio is predeclared as 25 percent per bucket by default.
- The within-bucket score is a transparent heuristic, not a learned selector.
- Gold labels remain in the row for later audit, but active scoring does not use them.
- The active manifest is not guaranteed to be class-balanced.

## Flow 3: Prompt Builder

| Item | Details |
|---|---|
| Code | `supervision/prompts.py` |
| Input | one selected train pair or one validation/test pair |
| Output | answer-only prompt string |
| Mode | `teacher_label` or `direct_prediction` |
| Allowed answers | `match`, `non_match` |
| Gold label visibility | hidden from prompt |

Prompt behavior:

- Includes `pair_id`.
- Includes serialized Record A and Record B.
- Asks for exactly one label.
- Forbids explanation, JSON, and punctuation.

Sanity checks:

- Prompt must not include `label`, `target_label`, or `gold_label`.
- Same prompt family is used for teacher labeling and direct LLM matching.
- Prompt version is fixed as `answer_only_v1`.

## Flow 4: OpenRouter LLM Provider

| Item | Details |
|---|---|
| Code | `supervision/llm_providers.py` |
| Input | system prompt + user prompt |
| Output | normalized `LLMResponse` |
| Default provider | OpenRouter chat completions |
| Default model | `openai/gpt-5.4-mini` |
| Default temperature | `0.0` |

Provider output fields:

| Field | Meaning |
|---|---|
| `raw_answer` | raw model text |
| `input_tokens` | provider usage prompt/input tokens |
| `output_tokens` | provider usage completion/output tokens |
| `estimated_cost_usd` | provider cost or configured pricing estimate |
| `response_model` | model returned by provider |
| `provider_response_id` | provider response id |
| `metadata` | usage and provider details |

Sanity checks:

- Temperature must remain `0.0` unless changed before results are inspected.
- Token/cost metadata should be retained even if answer parsing fails.
- Live calls require `OPENROUTER_API_KEY`.

## Flow 5: Teacher-Label Generator

| Item | Details |
|---|---|
| Code | `supervision/generate_teacher_labels.py` |
| Input | selected-pair manifest JSONL |
| Output valid | teacher-label cache JSONL |
| Output invalid | reject cache JSONL |
| Resume behavior | reuses cached valid rows for same prompt/model |

Expected inputs:

```text
data/cache/wdc_products/selection_manifests/train_128.random.jsonl
data/cache/wdc_products/selection_manifests/train_128.llm_active_bucketed_v1.jsonl
```

Expected outputs:

```text
data/cache/wdc_products/teacher_labels/train_128.random.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl
data/cache/wdc_products/teacher_labels/train_128.random.openrouter.openai-gpt-5-4-mini.answer_only_v1.rejects.jsonl
data/cache/wdc_products/teacher_labels/train_128.llm_active_bucketed_v1.openrouter.openai-gpt-5-4-mini.answer_only_v1.labels.jsonl
data/cache/wdc_products/teacher_labels/train_128.llm_active_bucketed_v1.openrouter.openai-gpt-5-4-mini.answer_only_v1.rejects.jsonl
```

Valid cache row meaning:

| Field | Meaning |
|---|---|
| `label` | parsed LLM teacher label |
| `gold_label` | dataset gold label copied for audit/evaluation |
| `raw_answer` | raw model output |
| `valid` | parser result |
| `selection_*` | selection metadata copied from manifest |
| `input_tokens`, `output_tokens`, `estimated_cost_usd` | cost fields |

Important distinction:

```text
label      = LLM teacher output
gold_label = original WDC truth
```

Sanity checks:

- Invalid LLM answers go to reject cache.
- Valid rows must have `label`.
- Invalid rows must not have `label`.
- `gold_label` is used for teacher-noise analysis, not for prompting.
- Cache filenames should include the selection strategy.

## Flow 6: Strict Parser And Reject Routing

| Item | Details |
|---|---|
| Code | `supervision/prompts.py`, `supervision/generate_teacher_labels.py` |
| Input | `raw_answer` |
| Valid outputs | `match`, `non_match`, quoted forms, `non-match` normalized |
| Invalid outputs | explanations, punctuation, uncertainty, malformed text |

Examples:

| Raw answer | Parsed label | Valid? |
|---|---|---|
| `match` | `match` | yes |
| `"non-match"` | `non_match` | yes |
| `match.` | null | no |
| `The answer is match` | null | no |
| `uncertain` | null | no |

Sanity checks:

- Parser is intentionally strict to keep invalid-output rate measurable.
- Reject rows preserve errors for later inspection.

## Flow 7: Cache Validator And Cost Summary

| Item | Details |
|---|---|
| Code | `supervision/validate_teacher_labels.py`, `analysis/cost_summary.py` |
| Input | teacher-label, reject, or direct-prediction JSONL |
| Output | printed JSON summary |

Summary fields:

| Field | Meaning |
|---|---|
| `rows` | schema-valid rows summarized |
| `valid_count`, `invalid_count`, `invalid_rate` | validity stats |
| `duplicate_pair_ids` | repeated pair IDs |
| `label_distribution` | distribution of valid LLM labels |
| `gold_label_distribution` | gold-label distribution when present |
| `selection_strategy_distribution` | random vs active rows |
| `input_tokens`, `output_tokens` | token totals |
| `estimated_total_cost_usd` | total estimated cost |
| `estimated_cost_per_valid_label_usd` | cost divided by valid rows |

Sanity checks:

- Validate caches before target building.
- Check active vs random label distributions separately.
- Check invalid rate before trusting teacher labels.
- Check duplicate pair IDs before training.

## Flow 8: Direct LLM Matcher

| Item | Details |
|---|---|
| Code | `supervision/direct_llm_matcher.py` |
| Input | validation/test serialized JSONL |
| Output | direct LLM prediction cache + cost JSON |
| Purpose | repeated-inference cost and quality baseline |

Expected first input:

```text
data/cache/wdc_products/serialized/validation.jsonl
```

Expected outputs:

```text
outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.predictions.jsonl
outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json
```

Cache row meaning:

```text
label      = direct LLM prediction
gold_label = original WDC truth
```

Sanity checks:

- Use full validation split if budget allows.
- Otherwise declare fixed `--limit N --sample-seed 42` before inspecting results.
- Direct LLM matcher does not train a student.
- Direct LLM cost grows with every prediction, unlike distilled-student inference.

### Direct Metric Provenance For Thesis Writing

The direct LLM metric block in the `.cost.json` file is computed from the
direct-prediction JSONL rows, not manually entered. Each prediction row stores:

| Field | Meaning |
|---|---|
| `label` | parsed direct LLM output: `match` or `non_match` |
| `gold_label` | WDC dataset truth copied from the serialized validation/test row |
| `valid` | whether the raw LLM answer was parsed into the answer-only label set |

The metric computation uses only valid prediction rows:

```python
predictions = [row.label == "match" for row in valid_rows]
labels = [row.gold_label == "match" for row in valid_rows]
```

Then `utils/metrics.py` treats `match` as the positive class:

| Count | Meaning |
|---|---|
| `tp` | predicted `match`, gold `match` |
| `fp` | predicted `match`, gold `non_match` |
| `tn` | predicted `non_match`, gold `non_match` |
| `fn` | predicted `non_match`, gold `match` |

For the clean GPT-5.4-mini validation direct baseline:

```text
tp = 438
fp = 65
tn = 1935
fn = 62
```

The reported metrics therefore come from:

```text
match precision = tp / (tp + fp) = 438 / (438 + 65) = 0.8708
match recall    = tp / (tp + fn) = 438 / (438 + 62) = 0.8760
match F1        = 2PR / (P + R) = 0.8734
accuracy        = (tp + tn) / total = (438 + 1935) / 2500 = 0.9492
macro F1        = average(match F1, non-match F1) = 0.9208
```

Cost values in the same `.cost.json` summary come from summing the per-row
OpenRouter usage fields stored in the prediction JSONL:

```text
input_tokens
output_tokens
estimated_cost_usd
```

For the clean GPT-5.4-mini validation direct baseline, the summary is:

```text
rows = 2500
valid_count = 2500
invalid_count = 0
input_tokens = 969364
output_tokens = 15439
estimated_total_cost_usd = 0.7964985
```

Code path:

```text
supervision/direct_llm_matcher.py
  -> _direct_metrics(...)
  -> utils/metrics.py::compute_metrics(...)
  -> analysis/cost_summary.py::summarize_rows(...)
  -> write_summary_json(...)
```

### Direct Baseline Compared With Prior Work

Use this table as thesis-writing context for whether the direct LLM baseline is
plausible. These comparisons are not strict apples-to-apples because WDC
variants, split construction, sample sizes, prompts, and model access differ
across papers. The safest wording is that the GPT-5.4-mini direct result is
within the range of strong LLM-as-matcher reports on WDC Products.

| Source | Setup | Dataset/eval note | Match F1 |
|---|---|---|---:|
| This thesis, old direct baseline | `openai/gpt-4o-mini`, answer-only direct matcher | WDC validation, 2500 pairs | 0.7636 |
| This thesis, current direct baseline | `openai/gpt-5.4-mini`, answer-only direct matcher | WDC validation, 2500 pairs | 0.8734 |
| Peeters and Bizer 2023 | ChatGPT zero-shot, best basic prompt | WDC sampled validation, 433 pairs | 0.8624 |
| Peeters and Bizer 2023 | ChatGPT zero-shot plus rules | WDC sampled validation, 433 pairs | 0.8829 |
| Peeters and Bizer 2023 | ChatGPT plus 20 related demonstrations | WDC sampled validation, 433 pairs | 0.9020 |
| AnyMatch 2024 | MatchGPT with GPT-3.5-Turbo03 | WDC in multi-dataset zero-shot table | 0.7651 |
| AnyMatch 2024 | MatchGPT with GPT-4 | WDC in multi-dataset zero-shot table | 0.8583 |
| AnyMatch 2024 | AnyMatch small model | WDC zero-shot | 0.6331 |
| Steiner and Bizer 2026 | Distilled student from GPT-5.2 teacher | WDC student F1, not direct LLM | 0.7217 |
| Steiner and Bizer 2026 | Distilled student from Qwen 3.6 Plus teacher | WDC student F1, not direct LLM | 0.7249 |

Interpretation for writing:

- `gpt-4o-mini` was plausible for a cheaper direct LLM baseline but much more
  conservative on matches: match recall was 0.6880.
- `gpt-5.4-mini` improved match recall to 0.8760 and match F1 to 0.8734,
  placing it near strong direct LLM WDC reports.
- The comparison should be framed as contextual alignment, not a leaderboard
  claim, because the thesis uses its own fixed validation split and answer-only
  prompt.
- For the thesis cost story, the important question remains whether low-budget
  LLM-labeled students can approach this direct LLM quality at much cheaper
  repeated inference cost.

References to cite/check during thesis writing:

- Peeters and Bizer, 2023, "Using ChatGPT for Entity Matching".
- AnyMatch, 2024.
- Steiner and Bizer, 2026, "Labeling Training Data for Entity Matching Using
  Large Language Models".

## Flow 9: Phase 4 Handoff

| Item | Details |
|---|---|
| Input | validated teacher-label caches |
| Consumer | `supervision/build_targets.py` extension in Phase 4 |
| Output | compact-student target JSONL |

Expected future target variants:

| Variant | Training label source |
|---|---|
| `llm_random` | valid labels from `train_128.random` teacher cache |
| `llm_active_bucketed_v1` | valid labels from `train_128.llm_active_bucketed_v1` teacher cache |
| `mixed_gold_llm_active` | optional gold seed plus active LLM labels |

Sanity checks:

- Validation/test targets must use gold labels.
- Student training targets for LLM variants must use LLM `label`, not `gold_label`.
- Invalid teacher rows should be excluded or explicitly handled before training.

## Current Artifact Status

| Artifact | Status |
|---|---|
| `train_128.random` manifest | exists, 128 rows |
| `train_128.llm_active_bucketed_v1` manifest | exists, 128 rows after local regeneration |
| Random teacher-label cache | exists, 128 valid rows, 0 rejects, `openai/gpt-5.4-mini` |
| Active teacher-label cache | exists, 128 valid rows, 0 rejects, `openai/gpt-5.4-mini` |
| Direct validation prediction cache | exists, 2500 valid rows, 0 invalid, `openai/gpt-5.4-mini` |
| Validator/cost summary code | implemented |
| Unit tests | passing as of last run: 21 tests |

## End-To-End Sanity Checklist

- [x] Manifests are fixed and versioned before LLM calls.
- [x] Prompt does not expose gold labels.
- [x] Teacher-label cache stores both LLM `label` and dataset `gold_label`.
- [x] Direct LLM cache stores both LLM `label` and dataset `gold_label`.
- [x] Invalid model outputs are measurable, not silently coerced.
- [x] Token and cost fields are present for every LLM call when provider returns usage.
- [x] Selection strategy is present in teacher-label caches.
- [x] Random and active teacher-label caches are validated separately.
- [x] Phase 4 target builder uses teacher `label` for LLM variants.
- [x] Validation/test evaluation uses gold labels.

## Commands For Sanity Check

Preferred wrapper:

```bash
scripts/run_phase03_reproducible.sh local
```

This regenerates the fixed manifests and runs unit tests without live LLM/API
calls. Live OpenRouter runs are available through explicit subcommands:

```bash
scripts/run_phase03_reproducible.sh teacher-all
scripts/run_phase03_reproducible.sh direct
scripts/run_phase03_reproducible.sh validate
```

Underlying commands:

```bash
.venv/bin/python -m data.select_active_pairs \
  --input data/cache/wdc_products/low_label/train_128.jsonl \
  --budget 128 \
  --strategy random

.venv/bin/python -m data.select_active_pairs \
  --input data/cache/wdc_products/serialized/train.jsonl \
  --budget 128 \
  --strategy llm_active_bucketed_v1 \
  --easy-match-ratio 0.25 \
  --hard-match-ratio 0.25 \
  --easy-non-match-ratio 0.25 \
  --hard-negative-ratio 0.25

.venv/bin/python -m supervision.generate_teacher_labels \
  --pairs data/cache/wdc_products/selection_manifests/train_128.random.jsonl \
  --model openai/gpt-5.4-mini \
  --temperature 0.0

.venv/bin/python -m supervision.generate_teacher_labels \
  --pairs data/cache/wdc_products/selection_manifests/train_128.llm_active_bucketed_v1.jsonl \
  --model openai/gpt-5.4-mini \
  --temperature 0.0

.venv/bin/python -m supervision.direct_llm_matcher \
  --input data/cache/wdc_products/serialized/validation.jsonl \
  --model openai/gpt-5.4-mini \
  --temperature 0.0
```
