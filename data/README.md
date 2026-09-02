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
