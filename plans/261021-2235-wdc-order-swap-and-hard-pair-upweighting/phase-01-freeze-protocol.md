# Phase 1: Freeze protocol and predeclare arms

## Context Links

- Findings: `plans/journals/2026-10-20-record-order-swap-findings.md`
- Research: `plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/research/2026-10-20-order-instability-capacity-vs-training.md`, `.../2026-10-17-finding-useful-hard-pairs.md`
- Records: `docs/research/decisions.md`, `docs/research/ideas.md`, `docs/research/open-questions.md`
- Deferred items: `REVISIT.md` (swapped teacher pass; LLM-labeled validation)

## Overview

Priority P1. Pending. Write and hash the experiment contract before code that
produces results, so no later choice can be tuned on train gold or test.

## Key Insights

- Reweighting was dropped on 2026-09-18 to cut runs; this reopens it as upweighting only.
- Validation gold for early stopping and thresholds stays allowed (gold boundary in `plan.md`).
- Single runs cannot resolve ~1-point differences; paired tests are part of the contract.

## Requirements

`contract.md` beside this plan, with these sections and values:

1. **Labels.** Three-pass majority labels (`majority_labels/llm_unrefined.csv`, 495 match / 2,005 non-match) for all LLM arms.
2. **Shared recipe.** Qwen3-Reranker-0.6B, `configs/students/qwen3_reranker_0_6b.json` (original instruction), LoRA r=8, alpha=16, dropout 0.05, q/k/v/o; batch 1, grad accumulation 16, 10 epochs, lr 2e-4, weight decay 0.01, warmup ratio 0.1, seed 42, early stopping patience 3 on validation macro F1, threshold by `select_decision_threshold`. Same GPU type for all Phase 4 and Phase 8 runs.
3. **Recipe rule (Phase 4).** Adopt `random_swap` (p = 0.5) only if (a) its validation match F1 ≥ baseline − 0.005 and (b) its overall validation flip rate is lower than the baseline's. Otherwise keep `none`.
4. **Selection rules (Phase 6).** Formulas exactly as in Phase 6; current-loss window = epochs 1–2; K = 250; deterministic tie-break by `pair_id`.
5. **Reference model.** Use the Phase 5 OOF (v2) if it completes before Phase 6 starts; otherwise the existing OOF (`outputs/uncertainty_refinement/wdc-qwen-oof/oof_predictions.csv`). Recorded in the contract at Phase 6 start.
6. **Rule freeze.** Researcher picks one rule after the gold-free Phase 6 summary; reason written in the contract before Phase 7.
7. **Arms (Phase 8).** Upweight (rule R, K = 250, k = 2); random control (250 pairs, same class mix, seed 20261021); baseline = Phase 4 chosen run.
8. **Metrics.** Match F1 primary; precision, recall, macro F1, accuracy. Flip = decision differs between orders at the run's own threshold; flip rate overall, by outcome, and by |margin| decile (margin = original log-odds − logit(threshold)). Paired bootstrap of ΔF1: 2,000 resamples, seed 0, 95% percentile CI. McNemar exact test on per-pair correctness.
9. **Gold boundary and authorizations.** As in `plan.md`.

## Related Code Files

- Create: `plans/261021-2235-wdc-order-swap-and-hard-pair-upweighting/contract.md`
- Modify: `docs/research/decisions.md`, `docs/research/ideas.md`, `docs/research/open-questions.md`, this plan's `plan.md` (contract SHA-256)

## Implementation Steps

1. Draft `contract.md` from the requirements.
2. Researcher reviews; resolve any change before hashing.
3. Add to `docs/research/decisions.md` (dated 2026-10-21, researcher decisions): upweighting reopened; swap experiment; gold boundary; K = 250, k = 2, one rule; two items deferred to `REVISIT.md`. Mark anything unconfirmed `[CẦN XÁC NHẬN]`.
4. Add the order-flip signal and RHO-style selection to `ideas.md` as "under test"; supervisor approval to `open-questions.md`.
5. Record `sha256sum contract.md` in `plan.md`; commit plan, contract and records.

## Todo List

- [ ] Contract drafted
- [ ] Researcher confirmation
- [ ] Research records updated
- [ ] Contract hashed and committed

## Success Criteria

- Every later choice is predeclared here or made on validation by a rule written here.
- Contract hash in `plan.md` matches the committed file.

## Risk Assessment

- Scope creep (more rules, more k values). Mitigation: one rule, one k (decided).
- Contract edited after results. Mitigation: any edit after Phase 4 starts is a dated amendment with reason.

## Security Considerations

- No secrets; API keys stay in environment variables.

## Next Steps

- Phases 2 and 3 (can run in parallel).
