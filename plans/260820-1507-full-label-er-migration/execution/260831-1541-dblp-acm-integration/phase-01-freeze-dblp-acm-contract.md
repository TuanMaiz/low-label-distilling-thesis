---
phase: 1
title: "Freeze DBLP-ACM Contract"
status: completed
priority: P1
effort: "0.5-1d"
dependencies: []
---

# Phase 1: Freeze DBLP-ACM Contract

## Overview

Acquire the candidate DeepMatcher files into ignored raw storage, calculate all
dataset facts locally, and turn that observation into a reviewable contract.
The host has no immutable release tag and the preprocessed-files license needs
confirmation.

## Requirements

- Functional: before acquisition retain only the candidate source URL and
  filenames. Calculate hashes, schemas, split mapping/counts/balance,
  missingness, identity behavior, and overlap from the downloaded bytes.
- Functional: distinguish DeepMatcher structured data from the dirty variant and
  the Leipzig raw collection; never combine their counts or hashes.
- Non-functional: acquisition is limited to the five candidate filenames and
  ignored raw storage. No paid API request, model execution, normalized test
  artifact, or final-test access occurs.
- Non-functional: mark source mutability, license/attribution, and public-model
  contamination as review/limitation items rather than silently resolving them.

## Architecture

The Markdown contract is the human scientific authority. A small JSON dataset
profile later becomes the executable authority. Code may hash both for
provenance but must not parse Markdown as configuration.

## Related Code Files

- Create: `plans/260820-1507-full-label-er-migration/research/dblp-acm-dataset-contract.md`
- Create: `configs/datasets/dblp_acm.json`
- Create: `configs/labelers/dblp_acm_sol_high.json`
- Modify after review: `plans/260820-1507-full-label-er-migration/research/experiment-contract.md`
- Modify after implementation is meaningful: `data/README.md`, `AGENTS.md`, `CLAUDE.md`, `../AGENTS.md`
- Preserve unchanged: `data/cache/wdc_products/**`, `configs/students/qwen3_reranker_0_6b.json`, `labeller-screening/settings.json`

## Implementation Steps

1. Record only the candidate structured `DBLP-ACM/exp_data` source and five
   filenames; mark every dataset observation pending.
2. Download those files to ignored raw storage. Calculate SHA-256, sizes,
   headers/columns, counts, class behavior, missingness, value ranges,
   identifier/foreign-key integrity, duplicates, and split overlap locally.
3. After inspection, propose split mapping, record/pair identity grammar, row
   ordering, and label normalization from the values actually observed.
4. Propose attribute order, missing-value representation, entity noun, and
   record-source names only after the local headers and values are observed.
5. Calculate record and pair overlap for every locally observed split, then
   propose the overlap/duplicate policy for review.
6. Propose the dataset role and contamination limitation only after the local
   snapshot is shown to match the intended benchmark variant.
7. Freeze a separate DBLP JSON-Schema publication prompt/version. Reuse the
   selected `openai/gpt-5.6-sol` model, high reasoning, and OpenAI-only requested
   routing, but state that the domain prompt makes this a distinct labeling
   condition. Perform only zero-cost prompt/parser/payload review—no DBLP gold
   rescreening.
8. Present the local observation plus source mutability and license/attribution
   facts for explicit human review.
9. After approval of the locally observed contract, mark the DBLP
   contract/profile frozen and update the parent experiment contract's Dataset
   2 row. Retain only the reviewed locked-source contract for any test-designated
   file; omit an ordinary-cache test output.

## Success Criteria

- [x] Before download, the active config contained no inherited hashes, schemas,
  columns, counts, class balance, missingness, overlap, or derived schedule.
- [x] A local observation manifest calculates those fields from the newly
  downloaded bytes.
- [x] Human confirms the logical version and whether the mutable URL plus hashes
  is an acceptable source freeze.
- [x] Human confirms license/attribution handling for DeepMatcher-preprocessed
  files; raw Leipzig CC BY 4.0 is not overclaimed.
- [x] Human explicitly authorized fresh acquisition for local inspection.
- [x] DBLP prompt/version and JSON-Schema behavior are frozen separately while
  model, high reasoning, and OpenAI-only requested routing are recorded as the
  reused screening decisions.
- [x] The contract explicitly says no DBLP rescreening, paid labeling, training,
  validation/test prediction, normalized test artifact, or direct baseline is
  authorized.
- [x] Parent contract is updated only after the review facts are approved.

## Risk Assessment

The main risk is treating an observed mutable snapshot or raw-data license as a
formal release/license for the preprocessed benchmark. Mitigate with exact
hashes, separate provenance statements, human review, and fail-closed
acquisition. A second risk is assigning a benchmark role before confirming the
locally acquired variant; defer that characterization to the review.
