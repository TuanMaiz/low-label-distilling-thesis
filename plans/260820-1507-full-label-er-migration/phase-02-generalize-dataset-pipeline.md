---
phase: 2
title: "Generalize Dataset Pipeline"
status: pending
priority: P1
effort: "4-6d"
dependencies: [1]
---

# Phase 2: Generalize Dataset Pipeline

## Overview

Replace the WDC-only loader/path assumptions with a registry-driven normalized
pair pipeline for the three frozen datasets, while keeping upstream splits and
dataset-specific fields auditable.

## Context Links

- Contract: `./phase-01-freeze-experiment-contract.md`
- Existing WDC loader: `/mnt/d/study/cao-hoc/luan-van/code/data/er_dataset_loader.py`
- Existing schema/serializer: `/mnt/d/study/cao-hoc/luan-van/code/data/schema.py`, `/mnt/d/study/cao-hoc/luan-van/code/data/serialize_pairs.py`

## Requirements

- Each `configs/datasets/*.json` declares ID, `included`, source,
  version, splits, fields, checksums, and an existing loader backend name.
  Registry discovery asserts exactly three configs with `included: true`.
- Each adapter maps official train/validation/test rows to `GenericERPair`
  without changing labels, preserves raw IDs/source fields, and emits stable
  dataset-namespaced `pair_id` values.
- Cache grammar: `data/cache/{dataset_id}/{version}/{split}/...`; never reuse a
  bare pair ID or cache across datasets/versions.
- Audit row count, class balance, duplicate IDs, missing labels/records, split
  overlap, serialization determinism, source and normalized checksums. Detect
  overlap with a side-order-invariant canonical pair fingerprint, not pair ID
  alone. Report entity/record overlap across splits and enforce each dataset's
  Phase-1 policy: fail, allow-and-report, or not-applicable.
- Acquisition is idempotent and license-aware; checksum mismatch fails closed.

## Architecture

`included dataset JSON files -> dataset_registry -> named loader backend ->
GenericERPair -> deterministic JSONL + stats/manifest`. Dataset-specific parsing
stays behind existing backend-pattern adapters; no global experiment config.

## Related Code Files

- Create: `/mnt/d/study/cao-hoc/luan-van/code/data/dataset_registry.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/configs/datasets/{three_dataset_ids}.json`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/data/loaders/__init__.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/data/loaders/wdc_products.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/data/loaders/{non_wdc_dataset_1}.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/data/loaders/{non_wdc_dataset_2}.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/data/prepare_benchmarks.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/data/schema.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/data/serialize_pairs.py`
- Refactor/compatibility shim: `/mnt/d/study/cao-hoc/luan-van/code/data/er_dataset_loader.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_dataset_registry.py`
- Extend: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_phase01_wdc.py`

## Tests Before

Add fixtures for all three adapters and first write failing tests for canonical
split/count mapping, namespaced IDs, deterministic serialization/checksums,
reversed-side fingerprint collisions, entity-overlap policy, unknown datasets,
malformed labels, checksum failure, and included-config counts other than three.

## Implementation Steps

1. Define dataset JSON schema, backend dispatch, included discovery, exact-three
   assertion, and normalized manifest schema from Phase 1 decisions.
2. Extract WDC parsing into an adapter without semantic changes; prove fixture
   parity with the existing loader before adding datasets.
3. Implement both frozen non-WDC adapters using official splits and licenses.
4. Namespace pair IDs with dataset/version; compute a canonical fingerprint
   from ordered normalized record identities so `(A,B)` equals `(B,A)`.
5. Write deterministic split JSONL, stats, acquisition provenance, source hash,
   normalized hash, and schema version atomically.
6. Add a `--verify-only` mode that rehashes artifacts without downloading.
7. Update data README/AGENTS guidance only after all adapter tests pass.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Each frozen dataset fixture, all official splits | Exact normalized rows/labels |
| Same raw pair ID in two datasets | Distinct namespaced IDs |
| Pair/reversed pair in train and test | Fingerprint gate fails |
| Entity appears across splits | Apply frozen fail/report/N-A policy |
| Changed source byte/checksum | Acquisition fails, cache untouched |
| Repeated preparation | Byte-identical JSONL/manifests |
| Unknown registry ID/version | Fail before I/O |
| 2 or 4 included dataset configs | Registry fails with actionable IDs |

## Success Criteria

- [ ] Three adapters reproduce frozen split counts/balance and checksums.
- [ ] Registry discovers exactly three included configs through named backends.
- [ ] No duplicate/cross-split canonical pair fingerprints; entity overlap is
  reported and conforms to policy; artifacts are deterministic and
  dataset/version namespaced.
- [ ] Existing and new dataset tests plus full suite pass.

## Risk Assessment

Different benchmark schemas and split conventions can create incomparable or
leaky data. Mitigate with thin adapters, official split preservation, fixture
parity, explicit manifests, and hard overlap gates.

## Security/Data Integrity

Download only frozen upstream URLs, validate licenses/checksums before parsing,
write atomically, and never interpolate unvalidated dataset IDs into paths.

## Next Steps

Phases 3 and 4 consume the verified normalized artifacts and may proceed in
parallel.
