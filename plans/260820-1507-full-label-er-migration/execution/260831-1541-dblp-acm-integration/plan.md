---
title: "DBLP-ACM Dataset Integration"
description: "Prepare structured DeepMatcher DBLP-ACM for the existing full-label ER pipeline with a thin adapter and narrowly scoped dataset-aware generalization."
status: in-progress
priority: P1
effort: "4.5-8.5d"
branch: "refactor/full-label-er-migration"
tags: [full-label, entity-resolution, dblp-acm, dataset-integration, tdd]
blockedBy: []
blocks: []
created: "2026-08-31T08:41:42.967Z"
createdBy: "ck:plan"
source: skill
---

# DBLP-ACM Dataset Integration

## Overview

Integrate the structured DeepMatcher DBLP-ACM benchmark as the second
dataset without disturbing the completed WDC experiment. The implementation is
limited to pipeline readiness: freeze and verify the source contract, prepare
train/validation while auditing test contract facts only, make serialization
and supervision dataset-aware, and prepare a DBLP Qwen preflight/runner path. It does not make paid calls,
train a model, evaluate the test split, or freeze the broader global 3×3 contract.

The design deliberately reuses `GenericERPair`, the full-label target publisher,
the trainer/evaluator, metrics, OpenRouter accounting, and checkpoint machinery.
The only dataset-specific parsing belongs in a thin adapter; the remaining work
removes product/WDC assumptions through explicit profiles rather than cloning a
second workflow.

## Scope

### In scope

- Structured DeepMatcher `DBLP-ACM/exp_data`, with its supplied train/valid/test
  pair files and `tableA.csv`/`tableB.csv` record tables.
- One DBLP-ACM dataset profile, a thin loader, deterministic normalized output,
  train/validation audit manifests, and an explicit single-dataset preparation
  command. Source acquisition occurs only if Phase 1 explicitly approves it.
- Dataset-aware attribute ordering, a new neutral JSON-Schema labeling protocol,
  full-training input preparation, cache identity, and DBLP Qwen preflight.
- Tests that prove DBLP behavior and byte-for-byte compatibility of frozen WDC
  serialized data/config/artifact inputs.
- A dry-run/handoff showing that later paid labeling and GPU work are ready but
  remain gated.

### Out of scope

- Selecting or integrating dataset 3.
- Enforcing the parent plan's exact-three included-dataset gate during this
  explicitly scoped single-dataset preparation.
- A new 300-pair gold-based labeler screening for DBLP-ACM; Sol-high is reused
  only after the DBLP/global contract is frozen.
- Paid labeling, target publication from live labels, model training, GPU use,
  validation predictions, test evaluation, or a direct-LLM baseline.
- Materializing normalized/labeled DBLP test JSONL in the ordinary cache. This
  plan freezes only the test source hash/count contract.
- Editing, wrapping, or refactoring hash-bound WDC preflight/runner code, Qwen
  config, screening settings/code, completed targets, contracts, or artifacts.

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Benchmark variant | Frozen DeepMatcher structured `exp_data` snapshot | Exact locally calculated hashes bind the mutable host snapshot |
| Dataset role | Dataset 2: structured bibliographic ER | Adds domain and schema variety relative to WDC without claiming recency |
| Adapter boundary | `data/loaders/dblp_acm.py` | Dataset schema translation stays isolated from generic training code |
| Preparation mode | Explicit `--dataset-config` single-dataset command | Enables incremental integration without weakening the later exact-three global gate |
| Entity identity | `dblp:{id}` / `acm:{id}` | Source-qualified IDs prevent table collisions; duplicate content remains distinct |
| Pair identity | Dataset/version/split plus both source IDs | Deterministic, namespaced, and source-order preserving |
| Split mapping | `train.csv` → train; `valid.csv` → validation; test locked | Preserves official files without resplitting or exposing test rows |
| Serialization | `title, authors, venue, year`; blanks as `<missing>` | Matches observed columns while WDC defaults remain byte-identical |
| Labeler setting | Selected GPT-5.6 Sol model, high reasoning, OpenAI-only requested routing | Reuses the screened model/reasoning/routing decision without claiming the full WDC labeling condition is identical |
| DBLP labeling protocol | New neutral JSON-Schema request/response stack and separately frozen DBLP publication prompt/version | Avoids using the legacy plain-text `supervision/prompts.py` protocol or modifying screened WDC code |
| Outbound identity | System message is the frozen instruction; user message is exactly `input_text`; `pair_id` stays local | Prevents source IDs/correlation data from entering provider messages |
| Provider origin | Exact `https://openrouter.ai/api/v1` only | Reject alternate scheme/host/userinfo/query/redirect before secret resolution or network |
| Qwen instruction | DBLP student config holds model-side fields/instruction only | Dataset identity and runtime schedule belong in the execution profile; WDC config remains immutable |
| Target builder | Reuse unchanged unless a failing DBLP fixture proves a defect | Current API already takes dataset ID/version/count and paths |
| Preflight | New DBLP/profile-driven files alongside immutable WDC files | Existing WDC scripts are artifact-contract hash-bound; limited duplication is safer until a versioned global runner exists |

## Frozen Local Source Observation

The official archive and five independent HTTPS file downloads were acquired
on 2026-09-01 and were byte-identical. The deterministic local observation is
recorded in
`configs/datasets/observations/dblp_acm_2018_06_29_a15b752f.json` and implemented
by `configs/datasets/dblp_acm.json`. It now supplies the hashes, schemas, counts,
class balance, missingness, identifier/foreign-key checks, duplicates, overlap,
and normalization rules approved and frozen by the researcher on 2026-09-02.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Freeze DBLP-ACM Contract](./phase-01-freeze-dblp-acm-contract.md) | Completed — locally observed and researcher-approved 2026-09-02 |
| 2 | [Implement Adapter and Preparation](./phase-02-implement-adapter-and-preparation.md) | Implementation complete — awaiting researcher review |
| 3 | [Generalize Dataset-Aware Supervision](./phase-03-generalize-dataset-aware-supervision.md) | Pending |
| 4 | [Generalize Qwen Preflight and Runner](./phase-04-generalize-qwen-preflight-and-runner.md) | Pending |
| 5 | [Verify Compatibility and Handoff](./phase-05-verify-compatibility-and-handoff.md) | Pending |

## Dependencies

- Parent plan: `../../plan.md`, especially global Phase 1 (contract), Phase 2
  (three-dataset pipeline), Phase 3 (six targets), and Phase 5 (matrix runner).
- This child plan advances DBLP-ACM readiness but does not block the already
  completed WDC-Qwen slice and does not complete any parent exact-three gate.
- Fresh source acquisition, local inspection, and human approval of the
  observation, normalization, identity, and attribution wording are complete.
- Phase 2 prepared only the ignored train/validation cache; the researcher must
  review it before Phase 3 changes the supervision workflow.
- Paid DBLP labeling and GPU execution remain downstream of separate explicit
  authorization; they are not phases in this plan.

## Deliverables

- Frozen DBLP-ACM source/profile contract and acquisition notes.
- Deterministic adapter/preparation implementation and audit manifest.
- Dataset-aware serialization and supervision changes with WDC parity tests.
- Parameterized Qwen preflight/runner readiness without training.
- Verification report and runbook commands for the next authorized step.

## Acceptance Boundary

After Phase 1 freezes the fresh local observation, this plan is complete when a
clean checkout can acquire that recorded source snapshot, prepare/verify train and validation,
produce blinded train inputs in dry-run mode, run the full synthetic
runner-to-target-validator integration, demonstrate DBLP Qwen-preflight
compatibility, and prove the WDC baseline unchanged. If acquisition is denied
or deferred, the honest terminal state is **fixture-ready, source verification
blocked**, not source-ready or plan-complete. Completion never means paid DBLP
labels, production targets, checkpoints, validation metrics, or test artifacts
exist.

## Whole-plan test commands

The first two commands below are current Phase-2 tests. The following
supervision and Qwen-preflight modules are future Phase-3/4 commands and do not
exist until those phases are implemented.

Use the repository-managed environment only:

```bash
.venv/bin/python -m unittest tests.test_dblp_acm_loader -v
.venv/bin/python -m unittest tests.test_dataset_preparation -v
.venv/bin/python -m unittest tests.test_dataset_aware_supervision -v
.venv/bin/python -m unittest tests.test_dblp_acm_qwen_preflight -v
.venv/bin/python -m unittest tests.test_phase01_wdc tests.test_wdc_target_alignment tests.test_wdc_qwen_preflight -v
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s labeller-screening/tests -v
```

Baseline before implementation: 132 repository tests, 12 labeler-screening
tests, and 21 focused WDC orchestration/recovery tests passing.

## Risks

- **Mutable upstream:** calculate hashes from the fresh acquisition, freeze the
  reviewed local observation, then refuse later drift.
- **WDC reproducibility regression:** make dataset behavior opt-in; retain WDC
  code immutable and regenerate WDC serialization into a temporary directory for
  byte comparison against committed artifacts.
- **Leakage misstatement:** calculate overlap locally before making any
  entity-disjoint or pair-disjoint claim.
- **Benchmark role:** characterize breadth/difficulty only after confirming the
  locally acquired variant and reviewing external evidence.
- **Public-benchmark contamination:** closed-model training-data overlap is
  unknown; disclose it as a limitation rather than implying absence.
- **Gold leakage into labeling:** local blinded inputs contain only `pair_id` and
  `input_text`; the provider payload has a frozen system instruction and a user
  message exactly equal to `input_text`. Tests reject IDs/truth/metadata.
- **Unsafe filesystem targets:** resolve every path under explicit
  dataset/version roots and reject traversal, symlinks, aliases, and WDC overlap.
- **Charged-call ambiguity:** durably record an inflight request identity before
  any future call and fail closed on unresolved inflight state; exactly-once
  billing is impossible without provider idempotency.
- **Unproven upstream:** describe routing as OpenAI-only **requested** unless a
  response contains trustworthy upstream evidence.

## Red Team Review

The hard-mode review produced 23 raw findings. After deduplication and a cap of
15 actionable decisions, all of the following were accepted:

1. Never modify/wrap/refactor hash-bound WDC preflight, shell runner, Qwen
   config, settings, screening code, or completed artifacts.
2. Create one canonical neutral JSON-Schema production labeling stack for DBLP;
   do not use the legacy plain-text prompt path or weaken historical settings.
3. Reuse the selected model/high reasoning/OpenAI-only requested routing, but
   freeze a separate DBLP publication prompt/version and avoid claiming the
   labeling condition is identical.
4. Keep `pair_id` local; make the outbound user message exactly `input_text` and
   test the real payload.
5. Enforce the exact OpenRouter HTTPS origin before secrets/network and bind all
   protocol/code hashes into cache/run identity.
6. Constrain resolved outputs to explicit dataset/version roots and reject
   traversal, symlinks, aliases, or protected-WDC overlap.
7. Prepare only train and validation; freeze test hash/count without producing
   normalized/labeled test artifacts.
8. Detect train/validation overlap from normalized canonical source identities,
   not split-qualified pair IDs.
9. Separate student model configuration from execution identity/runtime; derive
   and freeze the optimizer/warmup schedule only from the locally observed count.
10. Require a zero-cost end-to-end synthetic run whose actual artifacts feed the
    untouched publisher and independent validator.
11. Pre-journal future charged requests; unresolved inflight state requires
    manual reconciliation because exactly-once billing cannot be guaranteed.
12. Specify an atomic preparation state machine and regenerate WDC into temp for
    byte comparison.
13. Separate acquisition authorization from observed-contract freeze; no
    dataset facts enter the active profile until locally calculated and reviewed.
14. Freeze lifecycle order `gold -> verify/package/checksum -> llm_hard ->
    verify/package`, portable repo-relative identities, and relocation tests.
15. Preserve global exact-three validation for the parent plan while allowing an
    explicit single-dataset preparation path here.

One suggestion—to split this into separate adapter, labeling, and runner plans—
was rejected because the user requested full offline pipeline readiness. Paid
labeling, GPU execution, and test evaluation remain excluded, keeping the child
plan bounded.

## Whole-Plan Consistency Sweep

`plan.md` and all five phase files were reread after the red-team revisions.
Stale shared-core/refactor-WDC, legacy plain-text prompt, test-materialization,
outbound-pair-ID, and unconditional source-readiness language was reconciled.
Unresolved internal contradictions: **zero**.

## Validation Log

- **2026-08-31 — planning validation:** the original plan passed an internal
  consistency review, but its inherited DBLP observations were rejected by the
  researcher on 2026-09-01 and are no longer active contract evidence.
- **Red team:** 23 raw findings were deduplicated into 15 accepted corrections;
  one scope-splitting recommendation was rejected with rationale above.
- **Planning consistency at 2026-08-31:** zero `[UNVERIFIED]` markers and zero
  unresolved plan contradictions; no implementation had occurred at that
  historical checkpoint.
- **2026-09-01 — Phase 1 local observation:** inspector tests pass 6/6, the
  repository suite passes 138/138, and labeler-screening passes 12/12. Two
  independent inspector runs exactly match the 5,139-byte committed manifest.
  Test inspection is limited to hash, size, header, and row count.
- **2026-09-02 — Phase 2 implementation checkpoint:** Phase 1 is complete and
  Phase 2 implementation is awaiting researcher review; Phases 3-5 remain
  pending. Repository tests pass 152/152, labeler-screening passes 12/12, real
  DBLP `--verify-only` passes, and fresh WDC serialization is byte-identical.
  No paid call, GPU action, validation prediction, normalized test artifact, or
  test evaluation occurred.
