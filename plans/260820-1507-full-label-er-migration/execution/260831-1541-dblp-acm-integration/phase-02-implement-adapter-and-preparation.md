---
phase: 2
title: "Implement Adapter and Preparation"
status: completed
priority: P1
effort: "1-2d"
dependencies: [1]
---

# Phase 2: Implement Adapter and Preparation

Implementation was independently tested and researcher-approved on 2026-09-02.
Phase 3 subsequently began from this approved preparation boundary.

## Overview

Implement a thin structured DBLP-ACM adapter and a deterministic explicit
single-dataset preparation path for train and validation only. The global
registry's exact-three assertion is deferred until all three datasets are
selected. Exact source acquisition is allowed only when Phase 1 records approval.

## Requirements

- Functional: consume the frozen local observation manifest, verify its source
  hashes/headers, resolve observed pair foreign keys, and normalize only the
  reviewed train/validation mappings to `GenericERPair`; validate the locked
  test source contract without materializing ordinary-cache test JSONL.
- Functional: emit dataset/version-namespaced IDs, deterministic serialized
  JSONL, stats, and a manifest containing source and normalized hashes.
- Functional: enforce every schema, label, key, duplicate, missingness, and
  overlap rule recorded by the locally frozen observation manifest.
- Non-functional: repeated preparation is byte-identical and atomic; explicit
  DBLP preparation does not require exactly three dataset configs.
- Non-functional: existing WDC prepared files remain byte-for-byte unchanged.
- Non-functional: resolved inputs/outputs remain under explicit allowed
  dataset/version roots; reject `..`, symlinks, aliases, and any protected WDC
  overlap before staging or mutation.

## Architecture

`configs/datasets/dblp_acm.json -> explicit profile loader ->
data/loaders/dblp_acm.py -> GenericERPair -> profile-aware serializer ->
data/cache/dblp_acm/{version}/serialized/{train,validation}.jsonl +
stats/manifest`. The test contract remains in the profile/manifest only.

Identity and fingerprint behavior follow the approved profile: source-qualified
record IDs, dataset/version/split-qualified pair IDs, and source-byte hashes.

## Related Code Files

- Create: `data/loaders/__init__.py`
- Create: `data/loaders/dblp_acm.py`
- Create: `data/dataset_profiles.py`
- Create: `data/prepare_benchmark.py`
- Consume unchanged: `configs/datasets/dblp_acm.json`
- Modify: `data/serialize_pairs.py`
- Modify only if the adapter proves a schema gap: `data/schema.py`
- Preserve as compatibility path: `data/er_dataset_loader.py`
- Create: `tests/test_dblp_acm_loader.py`
- Create: `tests/test_dataset_preparation.py`
- Create: `tests/fixtures/dblp_acm/**`
- Update: `data/README.md`

## Tests Before

Write failing fixture tests before adapter code:

1. Exact source headers, hashes, counts, split mapping, class balance, row order,
   missingness, and value ranges from the frozen local observation manifest.
2. Identifier, foreign-key, label-domain, duplicate, and cross-split behavior
   exactly matches the frozen observation manifest.
3. Exact reviewed namespaced identities and observed attribute order.
4. Locally observed overlap values and their reviewed policy are enforced;
   altered values do not silently pass a frozen-source audit.
5. Observed duplicate-content behavior is preserved under the reviewed identity
   rule, whatever the local audit establishes.
6. Any checksum/header/integrity mutation relative to the frozen
   observation fails before publication.
7. Atomic-state cases: absent output stages, fsyncs, renames; exact existing
   verifies/returns; partial/different output fails unchanged; orphan staging is
   reported and never adopted silently.
8. Hostile source/output roots using traversal, symlinks, aliases, or protected
   WDC overlap fail before file creation.
9. Test source hash/count/schema is audited, but no normalized test JSONL/path is
   emitted and no downstream profile exposes test to labeling/training/evaluation.
10. Regenerate WDC serialization into a temporary root with the compatibility
    path and byte-compare each result to committed artifacts.

## Implementation Steps

1. Define a minimal profile schema for ID/version/backend, file contracts,
   fields/order, entity terminology, split aliases, expected audits, and cache
   paths. Separate portable repo-relative artifact identities from resolved
   runtime paths. Load an explicit config; do not implement exact-three discovery.
2. Implement CSV parsing in `data/loaders/dblp_acm.py` according to the locally
   frozen headers, roles, fields, identities, relationships, and row policy. Do
   not impute source values.
3. Build the reviewed namespaced identities and preserve the provenance fields
   required by the local observation contract.
4. Thread optional `attribute_order` and profile context through
   `serialize_pair`, `pair_to_training_row`, `write_serialized_pairs`, and
   `preview_serialized_pair`. Retain current defaults so WDC bytes do not move.
5. Implement the explicit preparation CLI with source-root, dataset-config,
   output-root, and `--verify-only`. Canonicalize/validate safe roots before
   source reads or writes. If Phase 1 acquisition is not approved, allow only
   synthetic fixtures and report `fixture-ready, source verification blocked`.
6. Compute split stats, missingness, class balance, source hashes, output hashes,
   duplicate/canonical-pair audit, and record-overlap matrices. Write to a
   same-filesystem staging directory; fsync files/directory and rename atomically.
   Exact existing output verifies/returns; partial/different output fails
   unchanged; orphan staging is reported for manual inspection.
7. Store test source contract evidence in the manifest without calling the
   serializer or publishing `test.jsonl`.
8. Regenerate WDC through the compatibility serializer into a temporary root and
   byte-compare it to every committed WDC serialized artifact. Do not edit the
   WDC loader, cache, targets, config, settings, or runner.

## Success Criteria

- [x] Adapter/preparer reproduces the locally frozen train/validation counts,
  class counts, and order; the observed test source is audited only and no
  normalized test JSONL is materialized.
- [x] Normalized records use the locally frozen attribute order and render
  observed missing values with the reviewed explicit representation.
- [x] All identities are namespaced and deterministic according to the locally
  reviewed identity and duplicate policy.
- [x] Manifest reproduces locally frozen overlap values and applies the reviewed
  overlap policy without inventing a resplit.
- [x] `--verify-only` independently rederives expected normalized bytes and
  rechecks source/profile/observation hashes without downloading or rewriting artifacts.
- [x] Atomic-state and hostile-path tests pass, including traversal, symlink,
  alias, protected-WDC overlap, partial output, and orphan staging cases.
- [x] Single-dataset preparation succeeds with only the DBLP profile present;
  no exact-three global assertion is weakened or implemented prematurely.
- [x] Fresh temporary WDC regeneration is byte-identical to committed
  serialization; all protected WDC files remain unmodified.
- [x] If acquisition is not approved, the phase reports fixture-ready/blocked
  and does not claim source-verified readiness or completion.

## Risk Assessment

CSV quoting/Unicode and observed split/identity behavior can silently change
identities or counts. Mitigate with fixtures derived from the frozen local
observation, exact hashes, atomic output, and a manifest that makes audits
visible. Path confusion could overwrite
protected evidence; canonical-root enforcement and hostile-path tests are hard
gates. Optional serializer parameters could change WDC bytes; temporary
regeneration and byte parity are required.
