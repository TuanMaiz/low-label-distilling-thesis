# Phase 2: Category-rule dataset builder

## Context Links

- Pattern: `supervision/build_quick_refined_dataset.py`, `tests/test_quick_refined_dataset.py`
- Inputs: `data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl`,
  `data/cache/wdc_products/majority_labels/llm_unrefined.csv`,
  `outputs/uncertainty_refinement/wdc-evidence-check/diagnoses.csv`

## Overview

Priority P1. Pending. Apply the category rule and the action rule by code and
build one training file per arm. No paid call, no gold.

## Requirements

- New `supervision/build_category_treated_dataset.py` with `build` and `verify`.
- Rule function: diagnosis row + parsed record fields → (verdict, step);
  exactly the table in `plan.md`.
- `build --arm A|B|C` → `data/cache/wdc_products/refinement/category-rule-<arm>/`:
  `train.jsonl` (same row format as quick-tier-v1), `decisions.csv` (all 2,500
  pairs: verdict, step, action, original label, final label, included,
  duplicate_of), `manifest.json` (input and guideline hashes, counts by action,
  step and class).
- Upweight copies: `pair_id = "<id>#dup1"`, `duplicate_of = "<id>"`, text and
  label identical to the source row.
- Gold denylist: refuse `serialized/train.jsonl`, `full_label_targets/gold.jsonl`,
  validation and test files.
- `verify`: counts match manifest; every row traceable to source; rebuild is
  byte-identical.

## Related Code Files

- Create: `supervision/build_category_treated_dataset.py`, `tests/test_category_treated_dataset.py`

## Implementation Steps

1. Rule function with synthetic tests (one per step, price-only conflict,
   invalid diagnosis kept).
2. Builder and verifier with tests (each action, duplicates, denylist refusal,
   determinism).
3. Build A, B, C; verify; rebuild and compare hashes; report counts against the
   preview in `plan.md`.
4. Check the trainer accepts `#dup1` IDs (one-epoch CPU smoke on a tiny file).

## Todo List

- [ ] Rule function and tests
- [ ] Builder, verifier, tests
- [ ] Arms built and verified
- [ ] Trainer smoke with duplicates

## Success Criteria

- Verified datasets for A, B, C; counts by action and step reported; full test suite passes.

## Risk Assessment

- Upweight set mostly non-match (47 of 52 in the preview). Mitigation: report class mix; stated as a limitation.
- 51 teacher matches removed at step 3. Mitigation: report positives lost; version 2 targets exactly these pairs.

## Next Steps

- Phase 3.
