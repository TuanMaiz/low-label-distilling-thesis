# Phase 4: Build treated datasets

## Context Links

- Pattern: `supervision/build_quick_refined_dataset.py`, `tests/test_quick_refined_dataset.py`
- Inputs: `data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl`,
  `data/cache/wdc_products/majority_labels/llm_unrefined.csv`, Phase 3 `final_verdicts.csv`

## Overview

Priority P1. Pending. Apply the action rule by code to saved verdicts and build
one training file per arm.

## Requirements

- New `supervision/build_judge_treated_dataset.py` with `build` and `verify`.
- `build --arm A|B|C|D` → `data/cache/wdc_products/refinement/judge-<arm>/`:
  `train.jsonl` (same row format as quick-tier-v1), `decisions.csv` (all
  2,500 pairs: action, original label, final label, included, duplicate_of),
  `manifest.json` (input hashes, guideline hash, counts by action and class).
- Upweight copies: `pair_id = "<id>#dup1"`, `duplicate_of = "<id>"`, text and
  label identical to the source row.
- Arm D: random pairs not upweighted in C, matching C's upweight class mix,
  seed from the contract.
- Gold denylist: refuse `serialized/train.jsonl`, `full_label_targets/gold.jsonl`,
  validation and test files.
- `verify`: counts match manifest; every row traceable to source; rebuild is byte-identical.

## Related Code Files

- Create: `supervision/build_judge_treated_dataset.py`, `tests/test_judge_treated_dataset.py`

## Implementation Steps

1. Builder and verifier with synthetic tests (each action, duplicates, class
   mix of D, denylist refusal, determinism).
2. Build arms; verify; rebuild and compare hashes.
3. Check the trainer accepts `#dup1` IDs (one-epoch CPU smoke on a tiny file).

## Todo List

- [ ] Builder, verifier, tests
- [ ] Arms built and verified
- [ ] Trainer smoke with duplicates

## Success Criteria

- Verified datasets for each arm; counts by action reported.

## Risk Assessment

- C upweight set dominated by one class (expected: mostly non-match). Mitigation: report class mix; D matches it.
- Many removals of matches. Mitigation: report positives lost; stated as a limitation, not tuned.

## Next Steps

- Phase 5.
