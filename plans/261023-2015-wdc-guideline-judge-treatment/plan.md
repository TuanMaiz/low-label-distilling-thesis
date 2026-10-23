---
title: "wdc-guideline-judge-treatment"
description: "Treat the 158 WDC uncertain training pairs with a researcher-written guideline applied by an LLM judge, then test whether the treated labels train a better Qwen student than unrefined and uniform-removal baselines."
status: pending
priority: P1
effort: "~2–3 days code + ~7 GPU-h (RTX 3090 equivalent) + ~USD 11 API"
branch: feat/wdc-guideline-judge
tags: [experimental, uncertainty-refinement, wdc, qwen, llm-judge]
blockedBy: []
blocks: []
created: 2026-10-23
---

# wdc-guideline-judge-treatment

## Overview

Main thesis question (decisions D18–D20): find uncertain LLM-labelled training
pairs and treat them. Detection is fixed: 149 OOF teacher–student
disagreements plus 13 teacher `2/3` pairs, 158 unique pairs. Treatment: an LLM
judge applies `supervision/judge_guideline.md` (researcher-written, no domain
profile) to each pair, seeing records only. A fixed rule maps the verdict and
the teacher majority label to an action:

| Judge `verdict` | Teacher majority label | Action |
|---|---|---|
| 0 (not decidable) | any | remove |
| 1 or 2 | same as verdict | upweight (duplicate once, k = 2) |
| 1 or 2 | different | flip to the verdict |

Supersedes the treatment part of backlog plan `261021-2235-wdc-order-swap-and-hard-pair-upweighting`;
reuses its gold boundary, k = 2 duplication, random control and paired tests.

## Gold boundary (every phase)

| Gold | Allowed use |
|---|---|
| Train gold | Phase 6 final analysis only. Never in a method decision |
| Validation gold | Checkpoint and threshold selection; paired comparison of predeclared arms |
| Test gold | Phase 6 only, once, for frozen arms, with explicit authorization |

WDC test already influenced an earlier decision (prompt rollback, D13), so it
is not a clean confirmation set. DBLP-ACM is the cleaner confirmation later.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Sign off guideline and freeze contract](./phase-01-freeze-contract.md) | Pending |
| 2 | [Judge runner, offline](./phase-02-judge-runner.md) | Pending |
| 3 | [Paid judge run and aggregation](./phase-03-paid-judge-run.md) | Pending |
| 4 | [Build treated datasets](./phase-04-build-datasets.md) | Pending |
| 5 | [Train arms](./phase-05-train-arms.md) | Pending |
| 6 | [Compare, test once, final analysis](./phase-06-evaluate.md) | Pending |

Dependencies: 1 → 2 → 3 → 4 → 5 → 6. Phase 2 code can start before sign-off
but no paid call runs until Phase 1 is done.

## Arms

| Arm | Training data | Rows |
|---|---|---|
| A | Teacher majority labels, unchanged | 2,500 |
| B | A minus all 158 uncertain pairs (uniform removal) | 2,342 |
| C | A with judge-routed remove / flip / upweight | depends on judge |
| D (optional) | A plus a random upweight set, same size and class mix as C's | 2,500 + C's upweight count |

B tests the claim "not all uncertain pairs are the same"; D separates "the right
pairs" from "more training steps".

## Success criteria

- [ ] Guideline signed off and hashed before any paid call.
- [ ] Judge outputs pass literal-quote and verdict/step consistency checks; inter-pass agreement reported.
- [ ] Action rule applied by code only to saved outputs; rebuild is byte-identical.
- [ ] Arms trained with one shared recipe on one GPU type.
- [ ] C vs A, C vs B (and C vs D) reported with paired bootstrap ΔF1 and McNemar on validation.
- [ ] Test run once for frozen arms; train-gold analysis done last.

## Budget and authorizations

| Item | Cost | Needs |
|---|---|---|
| Phase 3 judge: 158 pairs × 3 passes | ~USD 9 | Researcher paid authorization + ceiling |
| Phase 3 audit: 100 agreed pairs × 1 pass | ~USD 2 | Same authorization |
| Phase 5: 3–4 runs × ~1.7 GPU-h | ~5–7 GPU-h | Researcher GPU approval |
| Phase 6: test inference | < 1 GPU-h | Explicit test authorization |

Cost basis: evidence check cost about USD 0.019 per pair; the judge output is
shorter, so this is an upper estimate.

## Reuse

`supervision/run_evidence_check.py` (selection, payload, quote validation,
dispatch markers, cost), `supervision/build_quick_refined_dataset.py` (builder
and verify pattern), `scripts/run_wdc_qwen_variant_prompt.sh` (training recipe),
`experiments/train_student.py`, `experiments/evaluate_student.py`,
`utils/classification_threshold.py`.

## Open questions (Phase 1)

1. Guideline step 4: option A or B; block-rule factor 2.
2. Judge model: same as teacher (Sol-high) or a different one.
3. Arm D: run or skip.
4. Supervisor sign-off needed before the paid run?
5. Work on a new branch `feat/wdc-guideline-judge` instead of the swap-probe branch?
