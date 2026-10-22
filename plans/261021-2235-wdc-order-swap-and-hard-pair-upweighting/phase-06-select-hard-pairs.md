# Phase 6: Select hard-but-useful pairs

## Context Links

- Contract: `./contract.md` (items 4–6)
- Pair scores: `outputs/uncertainty_refinement/wdc-qwen-upweight/<chosen arm>/train_pair_scores.jsonl`
- Reference: OOF version fixed in the contract (v2 from Phase 5, or the existing OOF)
- Teacher: `data/cache/wdc_products/majority_labels/{train.jsonl,consistency.csv}`
- Order signal (optional): `outputs/uncertainty_refinement/wdc-qwen-oof-v2/order_signals.csv`
- Evidence diagnoses: `outputs/uncertainty_refinement/wdc-evidence-check/diagnoses.csv`
- RHO-LOSS (Mindermann et al., ICML 2022): `documents/rho-loss-2022.pdf`

## Overview

Priority P1. Pending. Compute three predeclared rules, review gold-free
summaries, and freeze one rule and its 250-pair set for Phase 7.

## Key Insights

- RHO-style priority is high when the current model has not learnt a pair but a model trained without it supports its label.
- Risk for this task: the reference student shares the student's blind spot on identity details, so exactly the wanted pairs can look like noise. Teacher-trust judges the label with the teacher instead.
- `data/cache/wdc_products/serialized/train.jsonl` and `full_label_targets/gold.jsonl` carry gold labels; this phase must not read them.

## Requirements

Per training pair i, teacher majority label y_i:
- `current_loss(i)` = mean `loss` over epochs 1–2 in `train_pair_scores.jsonl`.
- `reference_loss(i)` = −log P_ref(y_i), from `assigned_label_probability` (existing OOF) or the equivalent v2 column; probabilities clipped to [1e-7, 1 − 1e-7].
- **RHO-style:** `priority = current_loss − reference_loss`; all pairs eligible; top 250 by priority.
- **Teacher-trust:** pool = `consistency == "3/3"`; rank by (`flip` desc if the order signal exists, then `current_loss` desc); top 250.
- **Combined:** same pool as teacher-trust; rank by RHO priority; top 250.
- Ties: `pair_id` ascending.
- Per-set gold-free summary: size; match share vs train 19.8%; Jaccard overlap between sets; share with an evidence diagnosis and its case counts; median token length; share with student flip (if available).
- Optional researcher review of 20 sampled pairs per set (seeded), recorded as a design note.

## Related Code Files

- Create: `supervision/select_hard_pairs.py`, `tests/test_select_hard_pairs.py`
- Outputs: `outputs/uncertainty_refinement/wdc-qwen-upweight/selection/{rho,teacher_trust,combined}.csv` (`pair_id`, label, scores, rank), `selection_summary.json` (with input hashes)

## Implementation Steps

1. Implement loaders for the allowed inputs only; refuse any path in a predeclared gold denylist (`serialized/train.jsonl`, `full_label_targets/gold.jsonl`, `serialized/validation.jsonl`, `serialized/test.jsonl`).
2. Implement the three rules and the summary.
3. Tests on synthetic inputs: formulas, clipping, pool filter, tie-break, K, determinism, denylist refusal, missing order signal.
4. Run; review the summary with the researcher.
5. Researcher picks one rule; record the rule, reason and the set's SHA-256 in `contract.md`.

## Todo List

- [ ] Selection module and tests
- [ ] Three sets and summary
- [ ] Researcher review
- [ ] Rule and set frozen in the contract

## Success Criteria

- Frozen rule and set hash recorded before Phase 7; no gold file opened (denylist test).

## Risk Assessment

- One class dominates a set. Mitigation: report it; the Phase 7 control matches the class mix.
- Teacher systematic errors pass the teacher-trust filter. Mitigation: measured in Phase 9; stated as a limitation.

## Security Considerations

- Gold denylist enforced in code and tested.

## Next Steps

- Phase 7.
