---
title: "wdc-order-swap-and-hard-pair-upweighting"
description: "Test record-order swap training and upweighting of hard-but-useful pairs for the WDC–Qwen student, keeping train gold out of every method decision."
status: backlog
priority: P1
effort: "~3–4 days code + ~9–10 GPU-h (RTX 3090 equivalent)"
branch: analysis/wdc-qwen-order-swap-probe
tags: [experimental, uncertainty-refinement, wdc, qwen]
blockedBy: []
blocks: []
created: 2026-10-21
---

# wdc-order-swap-and-hard-pair-upweighting

> **Backlog (researcher, 2026-10-22).** Out of scope for the current thesis
> question. Swap training and the order-flip detector are deferred (`docs/research/ideas.md` I16).
> RHO-style / teacher-trust selection over all train pairs (K = 250) is replaced by
> category-based treatment of the teacher–student disagreement pairs. See `REVISIT.md`.

## Overview

Two lines of work on WDC–Qwen, after the 2026-10-20 swap probe
(`plans/journals/2026-10-20-record-order-swap-findings.md`):

1. **Record-order swap.** The student is not order-invariant: errors flip far
   more than correct pairs at equal confidence. Test random A/B swap training
   (Ditto-style) and record-order flips as an extra uncertainty signal.
2. **Hard-but-useful upweighting.** The only new treatment. Select pairs the
   student has not learnt but whose teacher label looks trustworthy, then
   duplicate them (k = 2) against a same-size random control.

Treatments prune/relabel stay as they are; this plan adds upweighting only.

## Gold boundary (applies to every phase)

| Gold | Allowed use |
|---|---|
| Train gold | Phase 9 only (final analysis). Never in a method decision |
| Validation gold | Checkpoint and threshold selection; choosing among variants predeclared in Phase 1 |
| Test gold | Phase 9 only, once, for frozen arms |

Defense (researcher): validation and test are human-checked development and
evaluation sets; labeling them costs less than labeling train. Known weak point:
WDC validation (2,500) equals train size, so Phase 9 adds a small-subset check.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Freeze protocol and predeclare arms](./phase-01-freeze-protocol.md) | Pending |
| 2 | [Per-pair score logging](./phase-02-per-pair-logging.md) | Pending |
| 3 | [Random record-order swap option](./phase-03-swap-augmentation.md) | Pending |
| 4 | [Baseline and swap runs](./phase-04-baseline-and-swap-runs.md) | Pending |
| 5 | [Student order-flip signal (OOF)](./phase-05-order-flip-signals.md) | Pending |
| 6 | [Select hard-but-useful pairs](./phase-06-select-hard-pairs.md) | Pending |
| 7 | [Build upweight and control datasets](./phase-07-upweight-datasets.md) | Pending |
| 8 | [Upweight and control runs](./phase-08-upweight-runs.md) | Pending |
| 9 | [Final evaluation](./phase-09-final-evaluation.md) | Pending |

Dependencies: 1 → {2, 3} → 4 → 6 → 7 → 8 → 9. Phase 5 needs 1, 2 and the
Phase 4 recipe decision; Phase 6 uses its output when it finishes first.

## Decisions (researcher, 2026-10-21)

- Hard-set size K = 250 (10% of train); upweight factor k = 2.
- One selection rule goes to Phase 8 (two runs: upweight + random control).
- Deferred to `REVISIT.md`: swapped-order teacher pass; LLM-labeled validation.

## Success Criteria

- [ ] Phase 1 contract recorded and hashed before any GPU run.
- [ ] Logging and swap options are opt-in; flag-off training is unchanged (tests).
- [ ] Recipe (with or without swap) chosen on validation by the predeclared rule only.
- [ ] Hard-pair rule frozen without train gold; set size, class mix and overlap reported.
- [ ] Upweight and random control trained with the same recipe; paired bootstrap and McNemar reported.
- [ ] Test evaluated once for frozen arms; train-gold analysis done last.

## Budget and authorizations

| Item | Cost | Needs |
|---|---|---|
| Phase 4: two runs | ~3.4 GPU-h | Researcher GPU approval |
| Phase 5: three OOF folds | ~2–3 GPU-h | Researcher GPU approval |
| Phase 8: two runs | ~3.4 GPU-h | Researcher GPU approval |
| Phase 9: test inference | < 1 GPU-h | Explicit test authorization |

No paid API calls in this plan.

## Dependencies and outputs

Reuse: `experiments/trainer.py`, `experiments/train_student.py`,
`models/generative_reranker_student.py`, `models/student_config.py`,
`analysis/order_swap_probe.py`, `scripts/run_wdc_qwen_oof.sh`,
`experiments/wdc_qwen_oof.py`, `utils/classification_threshold.py`,
`data/cache/wdc_products/majority_labels/`.
Outputs: ignored `outputs/uncertainty_refinement/wdc-qwen-upweight/`,
`outputs/uncertainty_refinement/wdc-qwen-oof-v2/`,
`data/cache/wdc_products/refinement/upweight-*/`.

## Open questions

1. Does the supervisor need to approve upweighting and swap as new arms?
