---
title: "wdc-guideline-judge-treatment"
description: "Treat the 158 WDC uncertain training pairs by applying the guideline's decision steps to the existing evidence-check categories, then test whether the treated labels train a better Qwen student than unrefined and uniform-removal baselines."
status: pending
priority: P1
effort: "~1–2 days code + ~5 GPU-h (RTX 3090 equivalent); no API cost"
branch: feat/wdc-guideline-judge
tags: [experimental, uncertainty-refinement, wdc, qwen]
blockedBy: []
blocks: []
created: 2026-10-23
updated: 2026-10-24
---

# wdc-guideline-judge-treatment

## Overview

Main thesis question (decisions D18–D20): find uncertain LLM-labelled training
pairs and treat them. Detection is fixed: 149 OOF teacher–student
disagreements plus 13 teacher `2/3` pairs, 158 unique pairs.

**Version 1 (this plan): category rule.** Instead of a new paid judge run, the
decision steps of `supervision/judge_guideline.md` are applied by code to the
saved evidence-check diagnoses (researcher, 2026-10-24). Free, deterministic,
and every quote was already verified literally.

| Guideline step | Rule on saved diagnosis | Verdict |
|---|---|---|
| 1 | has `intra_record_contradiction` | 0 |
| 2 | has `cross_record_conflict` with at least one quote outside `price`/`priceCurrency` | 2 (non-match) |
| 3 | has `ambiguous_variant_bundle` or `ambiguous_role_scope` | 0 |
| 4 | one-sided information does not block (option B) | – |
| 5 | noise ignored (categories cannot show it hides evidence) | – |
| 6 | none of the above → match | 1 |
| – | invalid diagnosis (3 pairs) | keep unchanged |

The guideline has no fixed price threshold (researcher, 2026-10-24): supporting
information such as price is weighed by judgement, which the category rule
cannot do. A price-only conflict still never counts as a conflict at step 2,
because supporting information alone never decides non-match.

Action rule (code, outside the diagnosis):

| Verdict | Teacher majority label | Action |
|---|---|---|
| 0 | any | remove |
| 1 or 2 | same as verdict | upweight (duplicate once, k = 2) |
| 1 or 2 | different | flip to the verdict |

Preview on the saved diagnoses (2026-10-23): 93 remove,
52 upweight (47 non-match, 5 match), 10 flip (all match → non-match), 3 keep.

Known limits of version 1: step 3 removes every ambiguous pair (86, of which 51
are teacher matches, about 10% of positives); step 6 means "no problem found",
not "identity confirmed"; the diagnoses come from the teacher's model family
(`reviewers-question-mock/q02`).

**Version 2 (later, optional):** a paid judge panel (different model families)
on the step-3 pairs only, compared against version 1 as an ablation. Not
authorized; needs pricing and a separate decision.

## Gold boundary (every phase)

| Gold | Allowed use |
|---|---|
| Train gold | Phase 4 final analysis only. Never in a method decision |
| Validation gold | Checkpoint and threshold selection; paired comparison of predeclared arms |
| Test gold | Phase 4 only, once, for frozen arms, with explicit authorization |

WDC test already influenced an earlier decision (prompt rollback, D13), so it
is not a clean confirmation set. DBLP-ACM is the cleaner confirmation later.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | [Freeze contract](./phase-01-freeze-contract.md) | Pending |
| 2 | [Category-rule dataset builder](./phase-02-category-rule-builder.md) | Pending |
| 3 | [Train arms](./phase-03-train-arms.md) | Pending |
| 4 | [Compare, test once, final analysis](./phase-04-evaluate.md) | Pending |

Dependencies: 1 → 2 → 3 → 4. Phase 2 code can start before Phase 1 is frozen;
no dataset is published before it is.

## Arms

| Arm | Training data | Rows |
|---|---|---|
| A | Teacher majority labels, unchanged | 2,500 |
| B | A minus all 158 uncertain pairs (uniform removal) | 2,342 |
| C | A with category-rule remove / flip / upweight | about 2,459 (preview) |

B tests the claim "not all uncertain pairs are the same". A random-upweight
control (former arm D) is skipped for now (researcher, 2026-10-23).

## Success criteria

- [ ] Contract hashed before any dataset is published.
- [ ] Rule applied by code only to saved diagnoses; rebuild is byte-identical.
- [ ] Arms trained with one shared recipe on one GPU type.
- [ ] C vs A and C vs B reported with paired bootstrap ΔF1 and McNemar on validation.
- [ ] Test run once for frozen arms; train-gold analysis done last.

## Budget and authorizations

| Item | Cost | Needs |
|---|---|---|
| Phase 3: 3 runs × ~1.7 GPU-h | ~5 GPU-h | Researcher GPU approval (own funds) |
| Phase 4: test inference | < 1 GPU-h | Explicit test authorization |

No paid API calls in this plan. Supervisor approval not required (researcher,
2026-10-23).

## Reuse

`outputs/uncertainty_refinement/wdc-evidence-check/diagnoses.csv`,
`supervision/build_quick_refined_dataset.py` (builder and verify pattern),
`scripts/run_wdc_qwen_variant_prompt.sh` (training recipe),
`experiments/train_student.py`, `experiments/evaluate_student.py`,
`utils/classification_threshold.py`.

## Open questions (Phase 1)

1. Guideline sign-off.
