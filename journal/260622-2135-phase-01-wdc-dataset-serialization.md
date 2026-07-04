---
title: "Phase 01 WDC Dataset and Serialization"
date: 2026-06-22
phase: 1
status: completed
commit: 77c510aa801b79f88838d024b3320843de3877e4
plan: ../../plans/260619-reasoning-rationale-distillation/phase-01-dataset-and-serialization.md
tags: [entity-resolution, wdc-products, low-label, serialization, rationale-distillation]
---

# Phase 01 WDC Dataset and Serialization

## Context

Phase 01 implements the data foundation for the active thesis pivot:
reasoning-rationale distillation for low-label entity resolution. The old
FEBRL/noise and Wikidata/name-matching directions were intentionally not used
as the main dataset path because they do not create a strong enough test for
low-label generalization.

The selected primary dataset is WDC Products pair-wise entity matching. The
chosen configuration is:

- 80 percent corner-cases.
- Small development set.
- 100 percent unseen-products test split.
- Seed 42 for low-label sampling.

This setup is deliberately hard. It keeps the first thesis experiment close to
the main research question: whether rationale supervision helps compact models
when labels are scarce and test products are not simply memorized from train.

## What Changed

The committed Phase 01 implementation is:

```text
77c510a feat(data): add WDC phase 1 pipeline
```

Committed files:

```text
code/.gitignore
code/data/schema.py
code/data/er_dataset_loader.py
code/data/low_label_sampler.py
code/data/serialize_pairs.py
code/experiments/prepare_low_label_data.py
code/tests/test_phase01_wdc.py
```

The implementation creates a complete path from the official WDC pair-wise zip
archive to teacher-ready and mT5-ready JSONL files.

## Schema Decision

`code/data/schema.py` was rewritten around a generic ER contract:

- `GenericERRecord`
- `GenericERPair`

The schema no longer assumes person names, Wikidata IDs, family relations, or
FEBRL-style patient fields. Each record now has a stable ID, optional entity ID,
source name, and an attribute dictionary. This keeps the active pipeline usable
for WDC now and for Magellan, DeepMatcher, or transfer datasets later.

The pair schema stores:

- `pair_id`
- `record_a`
- `record_b`
- `label`
- `split`
- `metadata`

This is the core Phase 01 contract for later teacher prompting, rationale
validation, and student training.

## WDC Loader

`code/data/er_dataset_loader.py` loads the official WDC Products pair-wise
files from either:

- an extracted folder containing `*.json.gz` files, or
- the official zip file directly.

The loader identifies WDC files by filename pattern. For the selected setup it
uses:

- train: `wdcproducts80cc20rnd000un_train_small.json.gz`
- validation: `wdcproducts80cc20rnd000un_valid_small.json.gz`
- test: `wdcproducts80cc20rnd100un_gs.json.gz`

The loader preserves the original WDC product attributes:

- `title`
- `description`
- `brand`
- `price`
- `priceCurrency`

It also preserves useful WDC metadata:

- `cluster_id_left`
- `cluster_id_right`
- `is_hard_negative`

One important interpretation was confirmed during the phase: `is_hard_negative`
is not a general difficulty label. It is a WDC-provided boolean that only marks
whether a negative pair was selected by a similarity metric rather than random
negative sampling. Positive examples still have `label = 1`; WDC does not
provide row-level `hard_match` or `easy_match` labels in the pair-wise file.

## Low-Label Sampling

`code/data/low_label_sampler.py` creates deterministic low-label training
subsets:

- 16 labels.
- 32 labels.
- 64 labels.
- 128 labels.
- full training set.

The low-label subsets are balanced by design. Each budget has half matches and
half non-matches. This avoids tiny training sets collapsing into one class and
makes teacher rationale generation easier to compare across budgets.

Generated counts:

| Subset | Pairs | Matches | Non-Matches |
|---|---:|---:|---:|
| train_16 | 16 | 8 | 8 |
| train_32 | 32 | 16 | 16 |
| train_64 | 64 | 32 | 32 |
| train_128 | 128 | 64 | 64 |
| train_full | 2,500 | 500 | 2,000 |

The full set is not balanced because it reflects the WDC split distribution.
Only the low-label budgets are force-balanced.

## Serialization Format

`code/data/serialize_pairs.py` converts structured WDC rows into a textual
`input_text` field. This is the string that teacher prompts and seq2seq models
will consume.

Example shape:

```text
Task: decide whether Record A and Record B refer to the same real-world entity.

Record A:
- title: HDD 35 4TB Seagate IronWolf Pro NAS ST4000NE001
- brand: <missing>
- description: <missing>
- price: 154.10
- priceCurrency: <missing>

Record B:
- title: HD 3,5 4TB 7200RPM IRONWOLF PRO 128 MB SATA3 SEAGATE
- brand: <missing>
- description: <missing>
- price: 153.99
- priceCurrency: EUR
```

Each JSONL training row includes:

- `pair_id`
- `split`
- `label`
- `target_label`
- `input_text`
- `record_a`
- `record_b`
- `metadata`

`label` is numeric after serialization, while `target_label` is text:

```text
1 -> match
0 -> non-match
```

This keeps the file useful for both classifier-style baselines and seq2seq
training.

## Preparation CLI

`code/experiments/prepare_low_label_data.py` is the end-to-end command for this
phase.

Primary command used:

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
.venv/bin/python -m experiments.prepare_low_label_data \
  --wdc-root data/raw/wdc_products/80pair.zip \
  --output-dir data/cache/wdc_products \
  --corner-cases 80 \
  --train-size small \
  --test-unseen 100 \
  --seed 42
```

The CLI can also download the official pair-wise archive with `--download`,
but the generated raw archive is ignored by git.

Generated artifact paths:

```text
code/data/cache/wdc_products/stats.json
code/data/cache/wdc_products/serialized/train.jsonl
code/data/cache/wdc_products/serialized/validation.jsonl
code/data/cache/wdc_products/serialized/test.jsonl
code/data/cache/wdc_products/low_label/train_16.jsonl
code/data/cache/wdc_products/low_label/train_32.jsonl
code/data/cache/wdc_products/low_label/train_64.jsonl
code/data/cache/wdc_products/low_label/train_128.jsonl
code/data/cache/wdc_products/low_label/train_full.jsonl
```

The cache and WDC zip are ignored through `code/.gitignore`:

```text
data/cache/
data/raw/wdc_products/*.zip
```

## Dataset Counts

The generated full split counts are:

| Split | Pairs | Matches | Non-Matches |
|---|---:|---:|---:|
| train | 2,500 | 500 | 2,000 |
| validation | 2,500 | 500 | 2,000 |
| test | 4,500 | 500 | 4,000 |

The test split is harder than train and validation because it uses 100 percent
unseen products and contains a larger number of hard negatives.

## Verification

Focused Phase 01 tests were added in `code/tests/test_phase01_wdc.py`.

The tests cover:

- loading a tiny synthetic WDC-style zip;
- selecting train, validation, and test files by WDC filename convention;
- converting rows into generic ER pairs;
- balanced low-label sampling;
- serialization with explicit field names.

Verification command:

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
.venv/bin/python -m unittest tests.test_phase01_wdc
```

Result:

```text
Ran 3 tests in 0.023s
OK
```

Before committing, the staged diff was also scanned for obvious secrets. No
secret-like strings were found.

## Important Decisions

The first important decision was to use WDC Products as the main dataset rather
than FEBRL. WDC gives a better low-label and unseen-entity story, while FEBRL is
better kept as optional sanity-check or historical background.

The second decision was to use WDC's official pair-wise split rather than
creating new train, validation, and test splits. This avoids inventing a split
protocol and preserves the benchmark's intended difficulty dimensions.

The third decision was to make low-label subsets balanced, even though the full
WDC split is imbalanced. This makes the smallest budgets meaningful for teacher
rationale generation and reduces the chance that early experiments are dominated
by label imbalance rather than rationale quality.

The fourth decision was to preserve `is_hard_negative` exactly as WDC provides
it. Any future `hard_match`, `easy_match`, or richer difficulty taxonomy should
be explicitly marked as derived, not WDC-provided.

## Risks And Notes

The schema rewrite is intentionally aligned with the active thesis pivot, but
it means older code that imported person-specific classes from `data/schema.py`
will need migration or legacy imports. The repository already contains broader
pivot-cleanup changes, so this is consistent with the active direction rather
than a standalone backward-compatible change.

Teacher generation must avoid test leakage. Phase 02 should read only:

```text
code/data/cache/wdc_products/low_label/train_*.jsonl
```

It should not generate teacher rationales from:

```text
code/data/cache/wdc_products/serialized/test.jsonl
```

The WDC archive and generated cache are deliberately not committed. Anyone
reproducing the artifacts should run the preparation CLI.

## Next Step

Phase 02 should define the rationale schema and teacher-generation contract.
The immediate output should be cached teacher rationales for the low-label
training files only, with validation that every rationale references real input
attributes and does not hallucinate field names.
