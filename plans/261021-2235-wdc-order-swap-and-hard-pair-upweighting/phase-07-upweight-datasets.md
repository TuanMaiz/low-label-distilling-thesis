# Phase 7: Build upweight and control datasets

## Context Links

- Pattern: `supervision/build_quick_refined_dataset.py`, `tests/test_quick_refined_dataset.py`
- Source: `data/cache/wdc_products/majority_labels/train.jsonl` (Phase 4)
- Frozen set: `outputs/uncertainty_refinement/wdc-qwen-upweight/selection/<rule>.csv`

## Overview

Priority P1. Pending. Build two training files: majority targets plus the
frozen set duplicated once, and majority targets plus a random same-size,
same-class-mix set duplicated once.

## Key Insights

- Duplication (k = 2) is JTT-style upsampling; it adds 10% more steps, so the random control separates "right pairs" from "more steps".
- `load_target_rows` does not reject duplicate IDs, but copies still get their own IDs for traceability and for target validators.

## Requirements

- New `supervision/build_upweight_dataset.py` with subcommands `build` and `verify`.
- `build --selection <csv> --name <rule>`: all 2,500 source rows in source order, then one copy per selected pair with `pair_id = "<id>#dup1"`, `duplicate_of = "<id>"`; text, label and order identical to the source row.
- `build --random-control --match-class-mix <csv> --seed 20261021`: draw from pairs **not** in the frozen set, per class, as many as the frozen set has of that class.
- Outputs per dataset directory: `train.jsonl`, `duplicates.csv`, `manifest.json` (source and selection hashes, rule, K, k, seed, counts by class).
- `verify`: originals complete and byte-identical to the source; exactly 250 copies; each copy equals its source except `pair_id` and `duplicate_of`; class counts match the manifest; rebuild gives identical bytes.

## Related Code Files

- Create: `supervision/build_upweight_dataset.py`, `tests/test_upweight_dataset.py`
- Outputs: `data/cache/wdc_products/refinement/upweight-<rule>-k2/`, `data/cache/wdc_products/refinement/upweight-random-k2/`
- Modify: `scripts/run_wdc_qwen_upweight.sh` (map `upweight` and `control` arms to these files)

## Implementation Steps

1. Builder and verifier with tests (synthetic rows).
2. Build both datasets; run `verify`; rebuild and compare hashes.
3. Runner `preflight upweight` and `preflight control` pass.

## Todo List

- [ ] Builder and verifier
- [ ] Both datasets built and verified
- [ ] Runner preflight passes

## Success Criteria

- Two verified datasets of 2,750 rows each, matching manifests and class mixes.

## Risk Assessment

- Copies often land in the same batch. Mitigation: normal shuffling; accepted.

## Security Considerations

- None.

## Next Steps

- Phase 8.
