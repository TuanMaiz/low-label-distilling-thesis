# Active Dataset Pipeline

This directory holds the WDC Products dataset and serialization code for the
current cost-aware active LLM-labeling thesis.

Active modules:

- `schema.py`: generic ER record and pair contracts.
- `er_dataset_loader.py`: WDC Products pair-wise loader.
- `low_label_sampler.py`: deterministic balanced low-label subsets and the base
  random-selection control.
- `serialize_pairs.py`: teacher-, direct-LLM-, and student-ready text
  serialization.

Prepared artifacts should go under an ignored cache/raw-data directory such as
`data/cache/` or `data/raw/`, not into source files. The active WDC cache is
`data/cache/wdc_products/`. Active-selection manifests should live under
`data/cache/wdc_products/selection/` and must be fixed before teacher labels or
student results are inspected.
