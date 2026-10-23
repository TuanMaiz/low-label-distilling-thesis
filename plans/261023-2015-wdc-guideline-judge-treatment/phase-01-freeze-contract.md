# Phase 1: Sign off guideline and freeze contract

## Context Links

- Guideline: `supervision/judge_guideline.md` (draft v2)
- Decisions: `docs/research/decisions.md` D18–D21
- Reviewer prep: `reviewers-question-mock/q01–q04`

## Overview

Priority P1. Pending. Settle every choice that could otherwise be tuned on
results, then hash the guideline and the contract.

## Requirements

`contract.md` beside this plan with:

1. **Guideline:** final version, its SHA-256, status "signed off" with date.
2. **Selection:** the 158 pairs exactly as in `outputs/uncertainty_refinement/wdc-evidence-check/selected_pairs.json` (hash).
3. **Audit sample:** 100 pairs where teacher and OOF student agree, drawn uniformly with seed 20261023, stratified by teacher label in train proportion.
4. **Judge:** model, reasoning setting, provider pinning (OpenAI only, no fallback), max tokens, 3 independent passes on the 158, 1 pass on the audit sample.
5. **Aggregation:** per pair, majority verdict across 3 passes; no majority (three different verdicts) → 0. Invalid outputs after one predeclared retry → 0.
6. **Action rule:** table in `plan.md`; upweight = one duplicate (`<pair_id>#dup1`).
7. **Arms:** A, B, C, and D if chosen; D's random set seed 20261024, drawn from pairs not upweighted in C, matching C's upweight class mix.
8. **Training recipe:** identical to `scripts/run_wdc_qwen_variant_prompt.sh` (Qwen3-Reranker-0.6B, original instruction, LoRA, 10 epochs, lr 2e-4, seed 42, early stopping patience 3, fp16) on one GPU type.
9. **Metrics:** match F1 primary; precision, recall, macro F1, accuracy. Paired bootstrap ΔF1 (2,000 resamples, seed 0, 95% percentile CI) and exact McNemar on validation.
10. **Gold boundary and test caveat:** as in `plan.md`.
11. **Spend ceiling:** USD 15 for Phase 3.

## Implementation Steps

1. Researcher answers the open questions in `plan.md`.
2. Update the guideline (step 4, factor), mark it signed off, record its hash.
3. Draft `contract.md`; researcher reviews; record `sha256sum` in `plan.md`.
4. Add a dated decision (D22) to `docs/research/decisions.md`.

## Todo List

- [ ] Open questions answered
- [ ] Guideline signed off and hashed
- [ ] Contract written and hashed
- [ ] D22 recorded

## Success Criteria

- Every later choice is predeclared here or made by a rule written here.

## Risk Assessment

- Guideline edited after outputs are seen. Mitigation: any change after Phase 3 starts is a dated amendment and requires a new judge run directory.

## Next Steps

- Phase 2 (code may start in parallel; no paid call before this phase is done).
