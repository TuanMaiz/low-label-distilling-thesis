---
phase: 5
title: "Refactor Experiment Runner"
status: pending
priority: P1
effort: "5-7d"
dependencies: [3, 4]
---

# Phase 5: Refactor Experiment Runner

## Overview

Create a dataset-neutral orchestration script for the fixed 18-cell matrix and
three direct baselines. Deleted runner behavior is available only in Git
history; cleanup removes its exclusive tests/documentation.

## Context Links

- Targets: `./phase-03-build-full-label-targets.md`
- Compact models: `./phase-04-finalize-compact-cross-encoder-models.md`
- Narrow WDC/Qwen smoke contract: `./research/wdc-qwen-training-vertical-slice-contract.md`
- Deleted Phase-05 runner: Git history only.

## Narrow WDC/Qwen boundary

`scripts/run_wdc_qwen_vertical_slice.sh` is not the global 18-cell runner. Its
current authorization ends after RTX-3090 setup, fail-closed preflight, and a
tiny balanced smoke train/evaluate/reload check. The smoke evaluates only its
fixture, with zero warmup solely to make the single optimizer step nonzero.
The old full-run Qwen config and hyperparameters remain frozen. Full gold and
LLM-hard training, official full-validation predictions, and all test
predictions require later explicit approval.

## Requirements

- Discover included configs or accept explicit config-file lists, assert exactly
  3 datasets × 3 compact models, and enumerate
  `{gold,llm_hard}` = 18 unique train/evaluate cells. Direct LLM matcher runs
  once per dataset on frozen evaluation scope and is never treated as a
  model-training arm.
- Stable path grammar: `outputs/full_label/{experiment_id}/{dataset_id}/
  {model_id}/{label_source}/`; direct artifacts live under the
  exact path `outputs/full_label/{experiment_id}/{dataset_id}/direct_llm/
  {labeler_id}/{scope_id}/`.
- Run manifests record Git commit plus hashes of `experiment-contract.md`, used
  dataset/model configs, targets, prompt version, and runtime. Skip only exact
  complete matches;
  mismatch fails unless an explicit recoverable archive/force workflow is used.
- Validation selects threshold/checkpoint. Test evaluation is separately
  allowed once with explicit `--allow-final-test`. Direct accuracy uses the
  exact compact-model test IDs; a smaller cost-only sample is explicitly
  non-comparable. Model train/evaluate stages make no LLM-labeling calls.

## Architecture

`run_full_label_experiments.sh {preflight|direct-cost|train|validate|test|package|all}`
delegates computation to Python modules, writes atomic run manifests and
completion markers, and supports exact stage-boundary resume. `all` stops after
validation unless `--allow-final-test` is explicitly passed; then `test`
evaluates compact-model and direct accuracy on common IDs. `experiment_id` is the
stable plan slug by default or an explicit stable CLI value.

## Related Code Files

- Create: `/mnt/d/study/cao-hoc/luan-van/code/scripts/run_full_label_experiments.sh`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/experiments/full_label_matrix.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/train_student.py` (legacy identifier)
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/evaluate_student.py` (legacy identifier)
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/supervision/direct_llm_matcher.py`
- Reuse: `/mnt/d/study/cao-hoc/luan-van/code/utils/artifact_contract.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/runtime_provenance.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/checkpoint_manifest.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_full_label_runner.py`
- Migrate generic runtime assertions from `/mnt/d/study/cao-hoc/luan-van/code/tests/test_phase05_runtime.py` under new names; the deleted runner contract test and Phase-05 docs remain Git-history-only.

## Tests Before

Create shell/Python fixture tests before runner code: exact matrix cardinality
and uniqueness; deterministic paths; direct baseline separation; exact-manifest
skip/resume; changed target/config/runtime rejection; partial-stage restart;
test-stage lock; LLM-call prohibition during model runs; package completeness;
`all` stopping before test; common compact-model/direct test IDs;
and explicit cost-sample non-comparability.

## Implementation Steps

1. Discover included config files (or use explicit lists) and validate 18 cells.
2. Implement preflight: Git commit, experiment-plan hash, six target manifests,
   three dataset and model config hashes, runtime identity, and ceilings.
3. Implement stage functions and atomic run manifests/completion markers.
4. Add pre-test cost-only direct sampling on frozen validation IDs and final-test
   direct accuracy on the exact compact-model test-ID manifest; keep
   artifacts/scopes distinct.
5. Add validation-only flow; permit final `test` only after all 18 validation
   cells, manual checklist confirmation, and explicit `--allow-final-test`.
6. Make `all` stop after validation when the flag is absent; record the flag,
   validation-manifest hash, and exact test scope in the run manifest.
7. Add exact resume/archive semantics and compact artifact packaging; never
   mix experiment IDs, input hashes, or partial cells.
8. Remove/update deleted-runner-exclusive tests/docs, then dry-run the state
   machine on tiny fixtures and one real dataset/model/source smoke cell.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Matrix listing | Exactly 18 cells + 3 direct jobs |
| Re-run completed exact cell | Skip without mutation |
| Target/config/runtime changes | Refuse reuse |
| Interrupted cell | Restart at declared stage boundary |
| `test` without manual check/explicit flag | Refuse |
| `all` without `--allow-final-test` | Validate only; no accuracy test |
| Direct accuracy IDs differ from model IDs | Refuse comparison |
| Model-training stage attempts LLM call | Test/preflight fails |
| Same raw pair/model across datasets | Artifacts remain isolated |

## Success Criteria

- [ ] New runner dry-runs 18 cells and 3 baselines deterministically.
- [ ] Resume, mismatch, test-lock, and packaging tests pass fail-closed.
- [ ] A real smoke cell completes and obsolete runner-only tests/docs are gone.

## Risk Assessment

Long GPU runs amplify identity and resume bugs. Mitigate with a new small runner,
matrix tests, stage-boundary markers, immutable hashes, test locking, and a
single-cell smoke before full scheduling.

## Security/Data Integrity

Provider secrets are available only to labeling/direct stages; model jobs do
not receive them. Quote paths, validate IDs, hash stage inputs, and package only
declared artifacts—never environment files.

## Next Steps

Run the fixed matrix; Phase 6 aggregates only complete manifest-matching cells.
