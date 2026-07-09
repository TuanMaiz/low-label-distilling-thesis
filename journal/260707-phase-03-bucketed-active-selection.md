# Phase 3 Bucketed Active Selection

Date: 2026-07-07
Plan: `plans/260704-distiller-wdc-agent-execution/plan.md`
Tags: `wdc-products`, `active-selection`, `llm-labeling`, `phase-03`

## Context

The first active selector was `llm_active_hybrid`, a single blended heuristic
score over WDC product-pair features. During review we decided that this is hard
to defend because the top-level constants look arbitrary.

## What Changed

- Added `llm_active_bucketed_v1` in `data/select_active_pairs.py`.
- The active budget is allocated across four candidate buckets by default:
  - `easy_match_candidate`.
  - `hard_match_candidate`.
  - `easy_non_match_candidate`.
  - `hard_negative_candidate`.
- Default ratios are 25 percent each. At budget 128 this creates 32 rows per
  bucket.
- Ratio parameters are exposed in the builder function and CLI:
  `--easy-match-ratio`, `--hard-match-ratio`, `--easy-non-match-ratio`, and
  `--hard-negative-ratio`.
- Teacher-label schemas and cache generation now preserve `selection_bucket`,
  `selection_bucket_rank`, and `selection_bucket_quota`.
- The reproducibility wrapper now defaults to `ACTIVE_STRATEGY=llm_active_bucketed_v1`.

## Decision

Use the bucketed selector as the main Phase 3 active manifest. Keep
`llm_active_hybrid` only as an older optional comparison.

The bucket labels are candidates, not gold classes. The selector does not use
gold labels for scoring; gold labels are retained only for audit.

## Verification

- `scripts/run_phase03_reproducible.sh local`
- Result: 128 random rows, 128 bucketed active rows, 32 rows per bucket.
- Audit-only gold distribution for the bucketed active manifest: 44 matches and
  84 non-matches.
- Unit tests: 18 passed.
- `git diff --check`: clean.
