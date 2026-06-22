# Active Dataset Pipeline

This directory now holds the Phase 01 dataset and serialization code for the
reasoning-rationale distillation thesis.

Active modules:

- `schema.py`: generic ER record and pair contracts.
- `er_dataset_loader.py`: WDC Products pair-wise loader.
- `low_label_sampler.py`: deterministic balanced low-label subsets.
- `serialize_pairs.py`: teacher- and mT5-ready text serialization.
- `febrl/`: optional sanity-check dataset support, not the main thesis dataset.

Legacy Wikidata collection scripts and old multilingual-name documentation were
moved to `../legacy/wikidata/`.

Prepared artifacts should go under an ignored cache/raw-data directory such as
`data/cache/` or `data/raw/`, not into source files.
