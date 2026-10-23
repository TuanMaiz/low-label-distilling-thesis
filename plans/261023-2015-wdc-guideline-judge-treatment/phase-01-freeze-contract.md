# Phase 1: Freeze contract

## Context Links

- Guideline: `supervision/judge_guideline.md` (rule source for version 1)
- Decisions: `docs/research/decisions.md` D18–D21
- Diagnoses: `outputs/uncertainty_refinement/wdc-evidence-check/diagnoses.csv`

## Overview

Priority P1. Completed 2026-10-24. Write down every choice before any dataset exists.

## Requirements

`contract.md` beside this plan with:

1. **Inputs and hashes:** `diagnoses.csv`, `selected_pairs.json`,
   `majority_labels/llm_unrefined.csv`, the train inputs JSONL, the guideline.
2. **Category rule:** the step table in `plan.md`, including the price-field
   filter for step 2.
3. **Action rule:** table in `plan.md`; upweight = one duplicate (`<pair_id>#dup1`).
4. **Arms:** A, B, C.
5. **Training recipe:** identical to `scripts/run_wdc_qwen_variant_prompt.sh`
   (Qwen3-Reranker-0.6B, original instruction, LoRA, 10 epochs, lr 2e-4,
   seed 42, early stopping patience 3, fp16) on one GPU type.
6. **Metrics:** match F1 primary; precision, recall, macro F1, accuracy. Paired
   bootstrap ΔF1 (2,000 resamples, seed 0, 95% percentile CI) and exact
   McNemar on validation.
7. **Gold boundary and test caveat:** as in `plan.md`.

## Implementation Steps

1. Researcher signs off the guideline.
2. Draft `contract.md`; researcher reviews; record `sha256sum` in `plan.md`.
3. Add a dated decision (D22) to `docs/research/decisions.md`: version 1 uses
   the category rule on saved diagnoses; judge panel deferred.

## Todo List

- [x] Guideline signed off
- [x] Contract written and hashed
- [x] D22 recorded

## Success Criteria

- Every later choice is predeclared here.

## Risk Assessment

- Rule edited after seeing validation. Mitigation: any change after Phase 3 starts is a dated amendment and a new arm name.

## Next Steps

- Phase 2.
