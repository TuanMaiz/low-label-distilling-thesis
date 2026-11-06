---
title: "companies-treatment-arms"
description: "Repeat the WDC treatment arms (B remove detected, C category rule) on MaDI-Bench Companies: OOF detection, evidence check, category rule, train B and C."
status: in-progress
priority: P2
effort: "GPU: 3 OOF folds + arms B and C on the V100 (estimate ~5 runs of the Companies recipe); OpenRouter: evidence check on the detected pairs, about USD 2–4 (WDC: 158 pairs, USD 3.25)"
branch: feat/wdc-guideline-judge
tags: [companies, treatment, oof, evidence-check]
blockedBy: []
blocks: []
created: 2026-11-06
---

# companies-treatment-arms

Same method as WDC (plan `261023-2015-wdc-guideline-judge-treatment`), on
Companies. Contract: `contract.md`. Arms G and A are already trained and frozen
(test once done 2026-11-03).

## Why

On WDC treatment did not help; the gold audit showed most changed pairs were
already right (labels 96.8% correct on train). Companies labels are 92.9%
correct with a systematic error (127 gold matches the teacher rejects), so there
is more room to fix. Caveat: the evidence check uses the teacher's model family
and the guideline's "same specific entity", which agree with the teacher on
parent/subsidiary pairs; treatment may move labels away from the gold convention.

## Phases

1. **Folds (done, offline):** `prepare_oof_folds` on the gold-free inputs and
   majority labels → `data/cache/madi_companies/.../oof/majority-3fold`
   (3 folds, about 656 held out each; verified). Git-ignored like all Companies data.
2. **OOF training (GPU):** `scripts/run_madi_companies_oof.sh` preflight,
   `train-fold 1..3 --confirm-oof-training`, `merge`, `package-results`.
   Same recipe, V100, fp16 as the arms.
3. **Selection + evidence check (OpenRouter, paid):**
   `supervision/run_evidence_check_companies.py prepare` (free) shows the count;
   `run --confirm-paid-evidence-check`.
4. **Category rule (free):** `supervision/build_category_treated_dataset_companies.py
   build/verify --arm B|C` → `refinement/category-rule-{B,C}`.
5. **Train B and C (GPU):** `scripts/run_madi_companies_arms.sh` with arms `B|C`,
   validation, then test once, packaging.
6. **Analysis:** paired C vs A, C vs B, B vs A (validation, then test); gold audit
   of the rule's decisions on train (as WDC).

## Built (2026-11-06, offline, no paid calls)

- Folds built and verified.
- `scripts/run_madi_companies_oof.sh` (build/verify folds, plan, preflight,
  train-fold, merge via the WDC merge function, package-results).
- `supervision/run_evidence_check_companies.py`: the WDC prompt word for word
  except its quotable-field list (researcher, 2026-11-06). Flex tier, 2/10 price cap.
- `supervision/build_category_treated_dataset.py`: parser, supporting fields and
  contract are now parameters with the WDC defaults; the three WDC bundles still
  verify byte-identically.
- `supervision/build_category_treated_dataset_companies.py`.
- `scripts/run_madi_companies_arms.sh` accepts `B|C`; the G and A plan commands
  are byte-identical to before.
- Tests: `tests/test_companies_treatment.py` (synthetic data).

## Needs researcher sign-off before the paid step

See contract section 9.
