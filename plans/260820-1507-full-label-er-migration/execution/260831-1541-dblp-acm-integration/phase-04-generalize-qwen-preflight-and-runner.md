---
phase: 4
title: "Generalize Qwen Preflight and Runner"
status: pending
priority: P1
effort: "1.5-2.5d"
dependencies: [2, 3]
---

# Phase 4: Generalize Qwen Preflight and Runner

## Overview

Build a new DBLP/profile-driven validation-only Qwen preflight and runner beside
the immutable, hash-bound WDC workflow. Limited duplication is accepted until a
future versioned global runner replaces both. Prove readiness with fixtures and
CPU-side checks only; do not train, load CUDA models, or predict validation.

## Requirements

- Functional: preflight accepts explicit dataset/profile, target directory,
  validation path, model config, locally frozen train/validation counts and
  class counts, output root, and entity instruction.
- Functional: validate both target arms for aligned IDs/input text, validation
  IDs/labels/source identity, train-validation pair separation, input-length
  audit contract, test lock, and no LLM access.
- Functional: cross-split separation uses the canonical identity and overlap
  policy selected from the local observation manifest; presentation IDs cannot
  hide a prohibited repeated source pair.
- Functional: runner paths remain
  `outputs/full_label/{experiment}/{dataset}/{model}/{arm}` and expose setup,
  preflight, smoke, confirmed train-arm, verify, and package actions without
  authorizing any of them here beyond offline preflight fixtures.
- Non-functional: no hard-coded relationship between train and validation counts;
  no global 18-cell/exact-three logic. New DBLP files may duplicate the minimum
  necessary verified logic but never import by refactoring hash-bound WDC files.
- Non-functional: do not edit/wrap/refactor
  `experiments/wdc_qwen_preflight.py`,
  `scripts/run_wdc_qwen_vertical_slice.sh`, the WDC Qwen config, or completed
  artifact-contract inputs/results.

## Architecture

`dataset profile + DBLP Qwen execution profile -> new DBLP/profile preflight ->
new parameterized shell dispatcher -> existing train_student/evaluate_student/
artifact-contract/checkpoint utilities`.

DBLP gets a student config containing only model-side identity/architecture,
tokenizer, labels, input limit, LoRA, and instruction fields. Dataset identity
and runtime/training settings live in an execution profile. Two equivalence
checks are required: every student field except the publication instruction
matches the frozen WDC Qwen config, and DBLP runtime hyperparameters match the
approved vertical-slice policy. Optimizer and warmup step counts are derived
only after the local training-row count is frozen.

Portable artifact identities are repo-relative; resolved runtime paths are
recorded separately. Runner lifecycle is fixed:
`gold -> verify/package/checksum -> llm_hard -> verify/package/checksum`.

## Related Code Files

- Create: `experiments/dblp_acm_qwen_preflight.py`
- Create: `scripts/run_dblp_acm_qwen_vertical_slice.sh`
- Create: `configs/executions/dblp_acm_qwen_vertical_slice.json`
- Create: `configs/students/qwen3_reranker_0_6b_dblp_acm.json`
- Reuse: `experiments/train_student.py`, `experiments/evaluate_student.py`, `experiments/trainer.py`
- Reuse: `utils/artifact_contract.py`, `utils/checkpoint_manifest.py`, `utils/runtime_provenance.py`, `utils/torch_runtime.py`, `utils/peft_runtime.py`
- Preserve without edit/wrap/refactor; test externally: `experiments/wdc_qwen_preflight.py`, `scripts/run_wdc_qwen_vertical_slice.sh`, `configs/students/qwen3_reranker_0_6b.json`
- Create: `tests/test_dblp_acm_qwen_preflight.py`
- Run unchanged: `tests/test_wdc_qwen_preflight.py`

## Tests Before

1. DBLP preflight accepts the locally frozen train/validation counts without a
   hard-coded relationship or WDC count assumption.
2. The locally frozen validation balance is enforced; wrong split/source,
   duplicate IDs, label corruption, or train-validation pair overlap fails.
   A fixture violates the locally frozen cross-split pair policy under distinct
   presentation IDs and must fail via canonical source identity.
3. Test path/data is absent from every default command, contract, and package;
   attempts to pass it without a future final-test flag fail.
4. Model-training actions require explicit full-training confirmation before
   CUDA/model imports or output mutation.
5. Dataset/profile/config/hash mismatch and partial/corrupt output states fail
   closed; exact completed stages resume/verify idempotently.
6. Generic result verification rejects missing/duplicate/non-finite predictions,
   checkpoint/summary mismatch, and stale archive members.
7. Student-config equivalence covers every model-side field except the declared
   instruction and rejects dataset/runtime fields. Separate
   execution-profile tests freeze only training constants approved after the
   observed dataset contract and derive step counts from the frozen row count.
8. Lifecycle rejects `llm_hard` before gold verify/package/checksum and rejects
   stale/missing per-arm checksums.
9. A checkout-relocation fixture proves repo-relative artifact identities stay
   stable while resolved runtime paths change safely.
10. Existing 21 focused WDC tests pass and Git/blob hashes prove the WDC
    preflight, runner, and config were not edited.

## Implementation Steps

1. Treat the current WDC preflight/runner/config as read-only reference. Create
   new DBLP/profile-driven modules alongside them; accept limited duplication
   rather than invalidating existing artifact-contract hashes.
2. Define an explicit execution-profile schema for dataset ID, source identity,
   expected train/validation counts and balance, target/validation paths,
   student config, output grammar, training constants, derived optimizer/warmup
   schedule, lifecycle state, and portable repo-relative identities.
3. Treat counts independently and enforce the locally reviewed cross-split
   identity/overlap policy from normalized source/manifest provenance rather
   than presentation IDs. Continue to ban test inputs and LLM calls.
4. Add the DBLP student config with publication instruction. Check that all
   other model-side fields equal WDC and that no dataset/runtime field appears;
   separately check execution-profile hyperparameter/schedule equivalence.
5. Implement one parameterized shell runner using execution-profile paths and
   existing Python trainer/evaluator utilities. Keep destructive/recovery,
   confirmation, checksum, and packaging behavior fail-closed. Enforce gold
   train/verify/package/checksum before allowing llm-hard, then the same
   verification/package/checksum for llm-hard.
6. Store portable repo-relative path identities separately from canonicalized
   runtime paths, apply safe-root rules, and test a relocated checkout.
7. Exercise only list/config/preflight/state/package-fixture actions on CPU.
   Do not call setup/smoke/train/evaluate against official DBLP artifacts in
   this phase.
8. Run WDC focused/full regressions and verify Git/blob hashes for the untouched
   WDC workflow files.

## Success Criteria

- [ ] DBLP preflight validates locally frozen train/validation counts and class
  behavior without assuming their relationship or materializing the test split.
- [ ] Canonical-source overlap test rejects a repeated raw pair even when its
  split-qualified target IDs differ.
- [ ] Runner dry-run/listing resolves deterministic DBLP paths and both arms,
  while all GPU/training actions remain explicit and unexecuted.
- [ ] Student-model and execution-runtime equivalence are separate; execution
  profile freezes optimizer and warmup steps derived from the observed count.
- [ ] Lifecycle and relocation tests enforce gold-first verified packaging,
  checksums, stable portable identities, and safe resolved runtime paths.
- [ ] Confirmation, resume, corruption, contract-hash, checkpoint, and packaging
  fixture tests fail closed.
- [ ] Existing WDC preflight/runner/config are unedited by Git/blob hash and all
  21 focused tests plus the full suite pass.
- [ ] No CUDA requirement, model weight download, training checkpoint,
  validation prediction, test access, or LLM call occurs.

## Risk Assessment

Touching hash-bound WDC code would invalidate historical contracts, so it is a
hard prohibition. Limited new-file duplication is accepted until verifier
versioning/global runner migration is designed. A dataset-specific instruction
or derived schedule could be misreported as the WDC config; mitigate with two
separate equivalence checks, distinct hashes, and explicit execution identity.
