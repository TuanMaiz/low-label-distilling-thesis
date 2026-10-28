---
title: "madi-companies-dataset3"
description: "Prepare MaDI-Bench Companies (entity matching subtask) as dataset 3 and train the gold reference arm, then the LLM-label arm once teacher labels exist."
status: in-progress
priority: P1
effort: "~1 GPU-h per arm (V100, fp16); teacher labelling about USD 1–2 per pass (estimate scaled from WDC)"
branch: feat/wdc-guideline-judge
tags: [dataset3, madi-bench, companies, qwen]
blockedBy: []
blocks: []
created: 2026-10-28
---

# madi-companies-dataset3

Decision record (researcher, 2026-10-28): `docs/research/dataset3-madi-companies.md`.

## Done (2026-10-28)

- Source fetched at MaDI-Bench commit `062058d`, stored git-ignored in
  `data/raw/madi_companies/062058d/` with `SHA256SUMS`. No licence file in the
  repository.
- Frozen profile `configs/datasets/madi_companies.json`; preparation
  `data/prepare_madi_companies.py` (`prepare`, `verify`, `materialize-test`).
  Train 1,968 (698 match), validation 948 (335 match); test 599 locked.
  The 3 broken DBpedia URIs dropped as documented. Gold schema mapping applied;
  values raw except strip, blank → `<missing>`, FullContact literal `null`
  city (92) → `<missing>`.
- Outputs (git-ignored, `data/cache/madi_companies/madi-bench-companies-base-062058d/`):
  `serialized/{train,validation}.jsonl`, `full_label_targets/gold.jsonl`,
  `arm-manifests/G.json`, gold-free `teacher_labels/inputs.jsonl`, `stats.json`,
  `manifest.json`.
- Student config `configs/students/qwen3_reranker_0_6b_madi_companies.json`
  (instruction says "company"); runner `scripts/run_madi_companies_arms.sh`
  with per-source-pair F1 in packaging. Runbook section added.

## Arms

| Arm | Training labels | Status |
|---|---|---|
| G | Benchmark human labels (1,968) | Ready to train |
| A | Teacher three-pass majority | Blocked: needs teacher labels |

Same recipe as WDC (contract 261023 section 5 and A3): V100-SXM2-32GB, fp16,
seed 42, LoRA r=8, 10 epochs max, patience 3, macro-F1 threshold on the full
validation set. One model on both source pairs; report F1 per source pair and
their mean.

## Gold boundary

Train gold: arm G training and final analysis only. Validation gold: checkpoint
and threshold selection. Test: once per frozen arm. Validation contains known
label errors (decision record); stated as a limitation.

## Next steps

1. **Teacher labelling (paid, needs authorization):** the existing labelling
   pipeline (`supervision/run_full_labeling.py`) only accepts the DBLP-ACM
   profile type. Either generalise `data/dataset_profiles.py` or add a small
   Companies labelling entry point that reads `teacher_labels/inputs.jsonl`.
   Three passes, majority vote → `majority_labels/{train.jsonl,manifest.json}`.
2. Train G (can start now), then A.
3. Optional: email the authors about licensing (decision record).
