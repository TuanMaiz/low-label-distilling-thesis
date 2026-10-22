# Phase 3: Random record-order swap option

## Context Links

- `models/generative_reranker_student.py`: `ERGenerativeRerankerDataset`, `prepare_reranker_token_rows` (overflow check)
- `models/student_config.py`: `StudentConfig`, loader
- `analysis/order_swap_probe.py`: `swap_record_order`
- `experiments/trainer.py`: `Trainer.train` (epoch loop)
- Ditto `entry_swap` (swap with probability 1/2)

## Overview

Priority P1. Pending. Opt-in training option that swaps the Record A and
Record B bodies with probability 0.5, independently per pair and per epoch.

## Key Insights

- Random swapping keeps the number of training steps unchanged (unlike doubling the data), so the comparison with the baseline is fair.
- Validation and test inputs always stay in the original order; the probe measures flip rates afterwards.
- Adding optional fields with defaults leaves existing config files, and their hashes, unchanged.

## Requirements

- `StudentConfig` fields: `pair_order_augmentation: "none" | "random_swap" = "none"`, `pair_order_swap_probability: float = 0.5`. Validate: probability in [0, 1]; `random_swap` only for `generative_reranker`.
- Move `swap_record_order` into `models/generative_reranker_student.py`; `analysis.order_swap_probe` re-exports it so its tests and imports keep working.
- New `PairOrderAugmentedDataset(original, swapped, probability, seed)`:
  - Wraps two `ERGenerativeRerankerDataset`s built from the same rows (swapped one uses `swap_record_order` on `input_text`); labels identical.
  - `set_epoch(epoch)`; `__getitem__(i)` returns the swapped item when `random.Random(f"{seed}:{epoch}:{i}").random() < probability`.
  - Reports per-epoch swap counts.
- `Trainer.train` calls `train_loader.dataset.set_epoch(epoch)` at epoch start when the method exists (no-op otherwise).
- `train_configured_student` wraps only the training dataset when the config asks for it.
- `training_summary.json` records the option, probability and per-epoch swap counts.
- New config `configs/students/qwen3_reranker_0_6b_random_swap.json`: identical to the baseline config plus the two fields (same `student_id`, separate output directory, as with the retired variant config).

## Related Code Files

- Modify: `models/student_config.py`, `models/generative_reranker_student.py`, `experiments/train_student.py`, `experiments/trainer.py`, `analysis/order_swap_probe.py` (import only)
- Create: `configs/students/qwen3_reranker_0_6b_random_swap.json`, `tests/test_pair_order_augmentation.py`

## Implementation Steps

1. Config fields and validation; tests that existing configs load unchanged.
2. Move `swap_record_order`; keep the probe's public name.
3. Implement `PairOrderAugmentedDataset` and the `set_epoch` call.
4. Wire into `train_configured_student`; summary fields.
5. Tests:
   - `none`: training items identical to `ERGenerativeRerankerDataset` today.
   - `random_swap`: about 50% swapped (within ±5% on 2,500 synthetic rows), reproducible across instances, different between epochs, labels unchanged.
   - Validation dataset never wrapped.
   - Overflow check still applies to swapped inputs.
6. Focused tests, then the full suite (including `tests/test_order_swap_probe.py`).

## Todo List

- [ ] Config fields
- [ ] Shared swap helper
- [ ] Augmented dataset and `set_epoch`
- [ ] Summary fields and new config file
- [ ] Tests pass with the full suite

## Success Criteria

- Default config produces identical training inputs to today; the swap config is reproducible per seed and epoch.

## Risk Assessment

- Tokenizing both orders doubles cached inputs. Mitigation: 2,500 short pairs; acceptable.
- Interaction with the preflight that pins the baseline config. Mitigation: the swap config is used only by the new runner (Phase 4).

## Security Considerations

- None.

## Next Steps

- Phase 4.
