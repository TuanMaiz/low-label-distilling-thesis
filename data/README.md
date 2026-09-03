# Dataset Pipeline

This directory holds the dataset loading and pair-serialization code for the
Entity Resolution experiments.

Active modules:

- `schema.py`: generic ER record and pair contracts.
- `er_dataset_loader.py`: WDC Products pair-wise loader.
- `dataset_profiles.py`: strict explicit benchmark profile contracts.
- `loaders/dblp_acm.py`: frozen-source DBLP-ACM adapter.
- `prepare_benchmark.py`: deterministic atomic benchmark preparation CLI.
- `serialize_pairs.py`: teacher-, direct-LLM-, and student-ready text
  serialization.

Prepared artifacts should go under an ignored cache/raw-data directory such as
`data/cache/` or `data/raw/`, not into source files. The active WDC cache is
`data/cache/wdc_products/`. The full-label migration uses complete dataset
training splits. Training targets are produced either from benchmark gold
labels or from LLM-generated labels; the retired low-label and active-selection
implementations remain available in Git history.

The frozen DBLP-ACM source snapshot is stored only in ignored raw storage at
`data/raw/dblp_acm/`. Its committed executable contract is
`configs/datasets/dblp_acm.json`, backed by the locally calculated observation
manifest under `configs/datasets/observations/`. Reproduce the read-only audit
with:

```bash
.venv/bin/python scripts/inspect_dblp_acm_source.py \
  --archive data/raw/dblp_acm/dblp_acm_exp_data.zip \
  --source-root data/raw/dblp_acm/archive-2026-09-01/exp_data \
  --direct-root data/raw/dblp_acm/acquisition-2026-09-01 \
  --observed-on 2026-09-01
```

This audit does not prepare normalized test data or authorize paid labeling,
training, or evaluation.

Prepare its train and validation splits with the frozen profile:

```bash
.venv/bin/python -m data.prepare_benchmark \
  --dataset-config configs/datasets/dblp_acm.json \
  --source-root data/raw/dblp_acm/archive-2026-09-01/exp_data \
  --output-root data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f
```

Add `--verify-only` to audit the frozen source and all prepared hashes without
rewriting the cache. Only `train.jsonl` and `validation.jsonl` are materialized;
the test split remains represented by its locked hash/header/row-count contract.

## Offline labeling readiness

Phase 3 prepares gold-free full-training inputs and exercises the complete
dataset-aware supervision path with a deterministic fake client:

```bash
.venv/bin/python -m supervision.run_full_labeling \
  --pairs data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/serialized/train.jsonl \
  --dataset-profile configs/datasets/dblp_acm.json \
  --labeler-config configs/labelers/dblp_acm_sol_high.json \
  --output-dir data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/teacher_labels/fake_sol_high_phase3 \
  --expected-count 7417 \
  --fake
```

This command is offline-only: it accepts no paid client, makes zero API calls,
and does not publish production targets. Its output stays under ignored cache
storage. Current provider pricing, paid authorization, and a spend ceiling must
be reviewed separately before a future production-labeling phase.

## Offline integration verification

The DBLP integration handoff is recorded at
`plans/260820-1507-full-label-er-migration/execution/260831-1541-dblp-acm-integration/reports/verification.md`.
The real prepared cache now supports `--verify-only` after later phases add
separate directories such as `teacher_labels/`; verification still rejects any
drift or extra file in the preparation-owned root files and `serialized/` tree.

CPU-only Qwen readiness can be inspected without loading Torch or a model:

```bash
bash scripts/run_dblp_acm_qwen_vertical_slice.sh config
bash scripts/run_dblp_acm_qwen_vertical_slice.sh identity
bash scripts/run_dblp_acm_qwen_vertical_slice.sh plan
```

These commands do not authorize production labels or GPU execution. Any
`full_label_targets_fake_*` directory is synthetic integration evidence, not a
production training target.
