# Dataset Pipeline

This directory holds the dataset loading and pair-serialization code for the
Entity Resolution experiments.

Active modules:

- `schema.py`: generic ER record and pair contracts.
- `er_dataset_loader.py`: WDC Products pair-wise loader.
- `serialize_pairs.py`: teacher-, direct-LLM-, and student-ready text
  serialization.

Prepared artifacts should go under an ignored cache/raw-data directory such as
`data/cache/` or `data/raw/`, not into source files. The active WDC cache is
`data/cache/wdc_products/`. The full-label migration uses complete dataset
training splits. Training targets are produced either from benchmark gold
labels or from LLM-generated labels; the retired low-label and active-selection
implementations remain available in Git history.
