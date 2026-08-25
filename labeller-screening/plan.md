# WDC Labeller Screening Plan

## Purpose

Screen three fixed `gpt-5.6-sol` reasoning settings on one frozen, randomly
selected set of 300 WDC training pairs. Compare each setting's answer-only
predictions with benchmark gold labels by `pair_id` and use the result as
calibration evidence for selecting the full-label experiment's LLM labeler.

This is a preliminary screening experiment, not the thesis result matrix.

## Completion status

- All three 300-pair paid screening runs are complete.
- `sol_high` was selected after comparison with the private gold labels.
- The separately frozen WDC production vertical slice is complete: 2,500/2,500
  valid training predictions, comprising 300 verified reused predictions and
  2,200 new calls.
- The production run recorded zero invalid results, zero retries, and USD
  2.693225 cumulative cost under its USD 5 ceiling.
- Published labels:
  `../data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv`.

## Frozen screening design

- Dataset: the WDC hardest variant already selected in the Phase-1 contract.
- Source split: training only. Validation and test labels remain untouched.
- Sampling: uniform random sampling without replacement, seed `42`, `n=300`.
- Sample reuse: all three settings receive the exact same byte-identical input.
- Natural prevalence: do not balance or manually select categories. Record the
  sampled class counts in the manifest.
- Gateway: OpenRouter Chat Completions with `OPENROUTER_API_KEY`.
- Routing: OpenAI upstream only; parameter enforcement enabled; provider and
  model fallbacks disabled.
- Settings:
  1. `sol_high`: `openai/gpt-5.6-sol`, reasoning effort `high`.
  2. `sol_max`: `openai/gpt-5.6-sol`, reasoning effort `max`.
  3. `sol_pro_max`: `openai/gpt-5.6-sol-pro`, reasoning effort `max`.
- Prompt: one fixed prompt and structured answer schema for every setting.
- Output: one final CSV per setting containing only `pair_id,result`.
- Primary screening metric: positive/match F1. Also report match precision and
  recall, macro F1, accuracy, invalid rate, and confusion counts.

## Phase A — Prepare and freeze the sample

Run `prepare_sample.py` against the normalized WDC training JSONL.

The model-facing JSONL must contain exactly `pair_id` and `input_text`. It must
not contain `label`, `target_label`, records, entity IDs, cluster IDs, metadata,
hard-negative flags, or any other truth-bearing selection field.

Write gold labels separately as `pair_id,gold_label`. The manifest records the
seed, sample size, class counts, source hash, output hashes, and sampled-ID hash.
Refuse duplicate IDs, malformed rows, non-training rows, undersized sources, or
an attempt to overwrite a different frozen sample.

Acceptance gates:

- Exactly 300 unique IDs.
- Deterministic reproduction with seed 42.
- No truth-bearing fields in model input.
- Separate gold and input hashes recorded.

## Phase B — Run the three settings

Run each setting independently with `run_setting.py`. A dry run validates the
inputs and request configuration without making network calls. Paid execution
requires both an API key in the environment and the explicit
`--confirm-paid-screening` flag and a positive per-setting
`--spend-ceiling-usd`. Before each call, the runner reserves a conservative
maximum request cost using the frozen token prices and output-token cap; an
ambiguous transport failure is charged that reserve. This prevents subsequent
calls from crossing the approved ceiling. OpenRouter-reported charged cost is
used when present, and the routing price cap fails closed if the frozen price is
no longer available.

Each Chat Completions request is standalone, uses the same instructions and
strict JSON schema, and receives only one blinded pair.
Journal each attempt so interrupted runs can resume. Retry only transient
transport failures or malformed structured output, with a cumulative fixed cap;
provider refusals and incomplete responses stop for review. No majority voting,
label correction, fallback model, or gold access is allowed.

Acceptance gates:

- Exactly 300 valid unique predictions per setting.
- Final prediction CSV has only `pair_id,result`.
- IDs match the frozen input exactly.
- Request/audit sidecar contains provenance and usage, but no gold label.
- No paid call without explicit confirmation.

## Phase C — Compare after all settings finish

Run `compare_results.py` only after all three final CSVs exist. It joins each
prediction to the private gold file by `pair_id`, never row position. Missing,
extra, duplicate, or invalid predictions fail closed.

Write a deterministic JSON report and CSV summary. Rank settings by match F1,
then match recall, then setting name. Also record pairwise disagreement counts
so the researcher can inspect where settings differ.

Acceptance gates:

- Three complete result sets over the same 300 IDs.
- Metrics recomputed from prediction and gold values.
- Screening limitations state that the seed-42 sample has natural WDC class
  prevalence and is only a 300-pair preliminary estimate.
- Human review is required before freezing the Phase-1 labeler choice.

## Phase D — Completed WDC full-training vertical slice

After the researcher selected `sol_high`, a narrow contract was frozen at
`../plans/260820-1507-full-label-er-migration/research/wdc-sol-high-vertical-slice-contract.md`.
`run_full_wdc.py` verified and reused the 300 compatible screening attempts,
then called only the other 2,200 official WDC training rows. Gold labels,
validation rows, and test rows were never part of the request path.

Publication required exact coverage of all 2,500 unique frozen training IDs.
The completed CSV contains only `pair_id,result`; the adjacent attempt, audit,
run-manifest, input, and input-manifest files preserve provenance and measured
cost. No new paid WDC labeling run is authorized by this completed phase.

## Commands

```bash
.venv/bin/python labeller-screening/prepare_sample.py
.venv/bin/python labeller-screening/run_setting.py --setting sol_high
.venv/bin/python labeller-screening/run_setting.py --setting sol_high --confirm-paid-screening --spend-ceiling-usd YOUR_REVIEWED_CEILING
.venv/bin/python labeller-screening/run_setting.py --setting sol_max --confirm-paid-screening --spend-ceiling-usd YOUR_REVIEWED_CEILING
.venv/bin/python labeller-screening/run_setting.py --setting sol_pro_max --confirm-paid-screening --spend-ceiling-usd YOUR_REVIEWED_CEILING
.venv/bin/python labeller-screening/compare_results.py
.venv/bin/python labeller-screening/run_full_wdc.py
.venv/bin/python -m unittest discover -s labeller-screening/tests -v
```

## Scope boundary

This folder did not choose the labeler automatically: the researcher selected
`sol_high` after screening and separately approved the completed WDC production
run. It does not call validation/test data or alter the fixed 3×3×2 thesis
matrix. Datasets 2–3, Models 2–3, other paid labeling cells, compact-model
training, and final-test predictions remain blocked by the broader Phase-1
contract and their own explicit approvals.
