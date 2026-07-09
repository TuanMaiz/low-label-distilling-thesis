# 2026-07-07 - Phase 3 Active Selection Implementation

## What Changed

- Implemented fixed selected-pair manifest generation for Phase 3.
- Added `data/select_active_pairs.py` with two strategies:
  - `random`: preserves the existing balanced low-label sample as a manifest.
  - `llm_active_hybrid`: ranks serialized training pairs with label-free WDC
    difficulty heuristics.
- Added selection metadata to LLM cache schemas and teacher-label generation:
  `selection_strategy`, `selection_rank`, `selection_score`, `selection_seed`,
  and `selection_uses_gold_label`.
- Added selection-strategy summaries to cache/cost summaries.
- Extended unit tests for manifest creation and metadata propagation.

## Generated Manifests

- `data/cache/wdc_products/selection_manifests/train_128.random.jsonl`
- `data/cache/wdc_products/selection_manifests/train_128.llm_active_hybrid.jsonl`

The active selector does not use gold labels for scoring. Gold labels remain in
the manifest only because the serialized training rows carry them for later
audit/evaluation. The `llm_active_hybrid` `train_128` manifest has 33 matches
and 95 non-matches by gold-label audit.

## Next Step

Run live OpenRouter teacher labeling for both fixed manifests, then validate
invalid rate, label distribution, selection-strategy distribution, and cost
before Phase 4 target building.
