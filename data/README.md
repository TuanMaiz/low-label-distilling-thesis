# Active Dataset Pipeline

This directory holds the WDC Products dataset and serialization code for the
current cost-aware LLM-label distillation thesis.

Active modules:

- `schema.py`: generic ER record and pair contracts.
- `er_dataset_loader.py`: WDC Products pair-wise loader.
- `low_label_sampler.py`: deterministic balanced low-label subsets.
- `serialize_pairs.py`: teacher-, direct-LLM-, and student-ready text
  serialization.

Prepared artifacts should go under an ignored cache/raw-data directory such as
`data/cache/` or `data/raw/`, not into source files. The active WDC cache is
`data/cache/wdc_products/`.
