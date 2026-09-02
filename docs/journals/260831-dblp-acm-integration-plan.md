# DBLP-ACM integration planning decision

## Decision

Use the locally acquired structured DeepMatcher DBLP-ACM snapshot as Dataset 2.
Its exact bytes, schemas, official split mapping, source-qualified identities,
attribute order, missing-value rendering, and attribution wording were reviewed
and frozen on 2026-09-02. Its age remains a stated limitation; no claim of
modern-data recency is made.

## Implementation boundary

The existing generic pair schema, target publication, trainer/evaluator,
metrics, cost accounting, and checkpoint code remain reusable. DBLP needs a
thin CSV adapter, dataset-aware attribute ordering, and new offline
profile-driven labeling/preflight entrypoints. The completed WDC preflight,
runner, Qwen config, screening code/settings, and artifacts are hash-bound and
must remain byte-identical, so limited adjacent DBLP code is safer than
refactoring historical verification paths.

The plan will prepare only the splits approved after local observation. The
test-designated candidate file remains locked and unmaterialized, and no paid
OpenRouter call, GPU training, validation prediction, or production target
publication is authorized.

## Phase 1 gate outcome

On 2026-09-01 the researcher rejected all inherited snapshot observations. The
official ZIP and independent file downloads were then acquired locally,
confirmed byte-identical, and inspected by a reproducible script. The resulting
manifest/config were approved as the executable frozen source contract.

## Phase 2 implementation outcome

The adapter consumes the frozen observation/profile, validates all five source
files, and materializes only train and validation as `GenericERPair` JSONL. The
result contains 7,417 train pairs (1,332 matches) and 2,473 validation pairs
(444 matches). Test remains locked to hash, size, header, and row count; no
normalized `test.jsonl` exists.

Publication uses same-filesystem staging, fsync, and atomic rename. Repeated
preparation is byte-identical. `--verify-only` independently regenerates the
expected normalized bytes and compares them with the publication rather than
trusting its manifest. It rejects profile/observation drift, forged data plus a
forged manifest, traversal, source/output aliases, symlinks, WDC overlap,
partial output, and orphan staging.

The independent review initially found two important gaps: verification trusted
mutable manifest hashes, and symlink aliases could be followed. Both were fixed
and covered by regression tests. It also prompted explicit acquisition gating,
profile-to-observation reconciliation, frozen identity/serialization contracts,
and fuller ID/year/missingness/canonical-pair audit output.

Final Phase 2 evidence: 152/152 repository tests, 12/12 labeler-screening tests,
real DBLP `--verify-only`, and a non-skipped fresh WDC byte-compatibility check
all pass. Phase 3 remains pending researcher review; no paid call, GPU run,
validation prediction, target publication, or test evaluation occurred.

## Review outcome

Hard-mode review found the most important hidden constraint: modifying the
completed WDC execution files would invalidate their artifact contracts. It
also tightened outbound payload isolation, exact OpenRouter origin checks,
crash accounting, atomic preparation, target-publisher integration, locally
derived training scheduling, and the gold-first two-arm lifecycle.
