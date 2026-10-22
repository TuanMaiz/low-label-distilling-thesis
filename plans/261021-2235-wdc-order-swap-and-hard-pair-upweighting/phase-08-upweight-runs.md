# Phase 8: Upweight and control runs

## Context Links

- Runner: `scripts/run_wdc_qwen_upweight.sh`
- Contract: `./contract.md` (shared recipe, metrics)
- Baseline: the Phase 4 chosen run

## Overview

Priority P1. Pending. Train the upweight arm and the random control with the
chosen recipe and logging, then compare both with the baseline on validation.

## Requirements

- Same recipe, seed, schedule, early stopping and GPU type as the Phase 4 chosen run.
- Validation gold for checkpoint and threshold only. No test.
- New `analysis/compare_arms.py`: inputs two prediction files on the same pairs (and labels); outputs ΔF1 with paired bootstrap 95% CI (2,000 resamples, seed 0), McNemar exact p-value, counts of pairs fixed and broken, and their ID lists.
- Comparisons: upweight vs baseline, control vs baseline, upweight vs control.
- Order-flip probe on validation for both arms.
- Fixed and broken validation pairs exported for researcher reading (validation is the human development set).

## Related Code Files

- Create: `analysis/compare_arms.py`, `tests/test_compare_arms.py`
- Outputs: `outputs/uncertainty_refinement/wdc-qwen-upweight/{upweight,control}/`, `.../comparisons/validation.json`

## Implementation Steps

1. `compare_arms.py` and tests (known small cases for bootstrap and McNemar).
2. Train upweight, then control; evaluate, probe, package each.
3. Run comparisons; write a dated journal entry with the numbers and CIs.

## Todo List

- [ ] `compare_arms.py` and tests
- [ ] Upweight run
- [ ] Control run
- [ ] Comparisons and journal

## Success Criteria

- Arms comparable (same recipe and hardware); results reported with CIs; no gain claimed when a CI crosses 0.

## Risk Assessment

- Differences inside noise. Mitigation: report as a null result; a second seed for baseline and upweight only if budget allows and the contract is amended first.

## Security Considerations

- None.

## Next Steps

- Phase 9.
