---
title: "geo-er-treatment-arms"
description: "Treatment arms B (remove detected) and C (category rule) on Geo-ER Pittsburgh, same method as Companies (plan 261106) and WDC (plan 261023)."
status: in-progress
priority: P2
effort: "GPU: 3 OOF folds + arms B and C on the V100; OpenRouter: evidence check on the detected pairs (about USD 0.02 per pair, WDC pilot rate)"
branch: feat/wdc-guideline-judge
tags: [geo-er, treatment, oof, evidence-check]
blockedBy: []
blocks: []
created: 2026-11-11
---

# geo-er-treatment-arms

Researcher, 2026-11-11: run B and C on all three datasets. Contract: `contract.md`.
Arms G and A are not trained yet; they run in the same GPU session as the OOF folds.

## Phases

1. **Folds (done, offline):** `prepare_oof_folds` → `data/cache/geo_er/.../oof/majority-3fold`
   (3 folds, about 834 held out each; verified). Git-ignored.
2. **GPU:** `scripts/run_geo_er_arms.sh` G and A; `scripts/run_geo_er_oof.sh`
   preflight, `train-fold 1..3 --confirm-oof-training`, `merge`, `package-results`.
3. **Evidence check (paid):** `supervision/run_evidence_check_v2.py prepare|run --dataset geo_er_pit`.
4. **Category rule (free):** `supervision/build_category_treated_dataset_geo_er.py build --arm B|C`.
5. **Label audit before training (free, recommended):** compare the B/C bundle
   labels with train gold; on WDC the rule made labels worse (10/10 flips wrong).
6. **GPU:** arms B and C (`scripts/run_geo_er_arms.sh B|C`), validation, test once.
7. **Analysis:** paired C vs A, C vs B, B vs A; gold audit.

## Built (2026-11-11, offline)

- Folds; `scripts/run_geo_er_oof.sh` (Companies OOF runner with Geo-ER paths).
- `geo_er_pit` profile in `supervision/run_evidence_check_v2.py` (fields name,
  latitude, longitude, address, postalCode; the `Pair: distance_m` line is sent
  but not quotable).
- `supervision/build_category_treated_dataset_geo_er.py` (supporting fields:
  latitude, longitude).
- `scripts/run_geo_er_arms.sh` accepts `B|C`; G and A plan commands unchanged.
- Tests: `tests/test_geo_er_treatment.py`.
