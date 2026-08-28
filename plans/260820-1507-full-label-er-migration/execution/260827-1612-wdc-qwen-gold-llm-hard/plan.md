---
title: "WDC-Qwen Gold vs LLM-Hard Full Training"
description: "Implement and execute one provenance-bound WDC-Qwen validation experiment for each frozen training-label arm."
status: in_progress
priority: P1
effort: "1-2 implementation days plus GPU runtime"
branch: "refactor/full-label-er-migration"
tags: [experimental, entity-resolution, cross-encoder, qwen, full-label]
blockedBy: []
blocks: []
created: 2026-08-27
createdBy: "ck:plan"
source: skill
---

# WDC-Qwen Gold vs LLM-Hard Full Training

## Overview

Implement the first complete compact-model comparison in the active migration:
train `Qwen/Qwen3-Reranker-0.6B` once from the 2,500-row benchmark-gold target
and once from the 2,500-row Sol-high `llm_hard` target, then reload each best
checkpoint and evaluate it on the official 2,500-row validation split.

The reviewed Tesla T4 smoke is plumbing evidence only. Full arms require fresh
RTX 3090 setup and preflight provenance. The 4,500-row test split stays locked.

## Frozen Boundaries

- Arm order: `gold`, then `llm_hard`.
- One completed run per arm; no seed/repeat dimension.
- Only target path, label-source identity, and output path may differ.
- Full-run warmup ratio: `0.10`; smoke-only `0.0` does not carry forward.
- No tuning from smoke or validation results.
- No publication/upstream target validator in the GPU training path.
- No LLM calls, direct-LLM baseline, test access, or other datasets/models.

## Architecture

```text
reviewed T4 smoke
  -> authorize validation-only full slice
  -> implement fail-closed per-arm commands
  -> verify locally without GPU training
  -> fresh RTX 3090 setup + preflight
  -> gold train/reload/validation
  -> llm_hard train/reload/validation
  -> verify and package immutable artifacts
```

## Phases

| Phase | Name | Status |
|---|---|---|
| 1 | [Authorize Full Validation Experiment](./phase-01-authorize-full-validation-experiment.md) | Completed |
| 2 | [Implement Fail-Closed Two-Arm Runner](./phase-02-implement-two-arm-runner.md) | Completed |
| 3 | [Verify Implementation and Rental Handoff](./phase-03-verify-rental-handoff.md) | In progress — CPU verification passed; commit/push pending |
| 4 | [Execute and Preserve RTX 3090 Results](./phase-04-execute-preserve-results.md) | Pending |

## Dependencies

- Parent plan: `plans/260820-1507-full-label-er-migration/plan.md`.
- Published inputs: WDC `gold.jsonl`, `llm_hard.jsonl`, and official
  `validation.jsonl`.
- Frozen model config: `configs/students/qwen3_reranker_0_6b.json`.
- Existing trainer, evaluator, artifact-contract, and checkpoint-manifest APIs.
- Researcher approval is required when Phase 1 changes the narrow contract from
  smoke-only to full validation-only execution.

## Success Criteria

- [x] Full-run authorization is explicit while test access remains forbidden.
- [x] Separate gold and `llm_hard` commands share every frozen setting.
- [x] Exact completed outputs verify and skip; mismatched or partial outputs fail.
- [ ] Both arms record matching RTX 3090 runtime properties and resolved precision.
- [ ] Both arms produce 2,500 valid validation predictions and verified checkpoints.
- [ ] Results package is hash-verifiable and preserves both merged checkpoints.

## Unresolved Questions

None. The model, targets, hyperparameters, order, validation scope, and stop
conditions are already frozen by the parent contract and reviewed smoke.

## Validation Log

- Mode: hard, using two completed research reviews and a planner synthesis.
- Standard verification tier applies because the plan has four phases.

## Red Team Review

### Session — 2026-08-27

**Findings:** 15 deduplicated (8 accepted, 7 rejected)

| Finding | Disposition | Plan response |
|---|---|---|
| First-write contracts could accept dirty files | Accept, simplified | Use the approved pushed commit and Git cleanliness; add no duplicate hash authority |
| Gold and `llm_hard` pair alignment is assumed | Accept | Add one small offline alignment script and focused test |
| Merged checkpoints could be omitted from package | Accept | Preserve both complete checkpoint-manifest trees |
| Threshold artifacts could disagree | Accept | Cross-check summary, persisted threshold, predictions, and metrics |
| Actual epochs/optimizer steps are not persisted | Accept | Extend the existing training summary |
| Timing scopes are ambiguous | Accept | Persist named trainer, action, inference, and evaluation scopes |
| Rental paths use workstation absolutes | Accept | Use repository-relative runtime paths |
| Gold could be lost while `llm_hard` runs | Accept | Copy a verified per-arm package off-rental before continuing |
| Reintroduce model-revision pinning | Reject | Researcher explicitly removed this dimension; preserve merged models instead |
| Add a cryptographic execution-session ID | Reject | Matching recorded runtime properties are sufficient |
| Add another validation-file hash gate | Reject | Git plus the existing artifact contract already cover the committed path |
| Add a dependency-locking project | Reject | Record resolved packages; avoid expanding this slice |
| Add a general path/symlink security framework | Reject | Fixed repository paths and explicit package members are sufficient here |
| Make epoch checkpoint promotion transactional | Reject | Preserve failure and restart; no mid-run recovery is authorized |
| Restore publication/upstream validation | Reject | Training consumes committed final targets directly |

### Whole-Plan Consistency Sweep

- Files reread: `plan.md` and all four phase files.
- Decision deltas: Git owns committed-file identity; one small alignment script;
  no session ID; no extra validation hash; full checkpoints preserved.
- Reconciled stale references: session-proof language, checkpoint deduplication,
  absolute rental inputs, summary schema, and post-gold preservation.
- Unresolved contradictions: 0.
