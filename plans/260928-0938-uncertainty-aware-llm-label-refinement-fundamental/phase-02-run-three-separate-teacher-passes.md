---
title: "Phase 2: Run Three Separate Teacher Passes"
status: todo
priority: P1
---

# Phase 2: Run Three Separate Teacher Passes

## Overview

Produce three independent GPT-5.6 Sol-high judgments for each training pair by
running the existing labeler separately under three distinct pass names. Keep
the implementation intentionally small: reuse the current prompt, config,
OpenRouter client, input preparation, resume behavior, cost accounting, and
output schema. Do not build a new multipass orchestrator.

The immediate executable scope is WDC only. It already has a complete first
Sol-high pass, so this phase reuses that result as `pass_01` and pays for two
new complete passes, `pass_02` and `pass_03`. DBLP-ACM and Dataset 3 will use
the same three-directory convention later, after their production labeling
contracts and budgets are approved.

This phase produces teacher evidence only. Majority voting, consistency
calculation, gold comparison, OOF training, detection, and refinement belong to
later phases.

## Context Links

- [Foundational plan](./plan.md)
- [Phase 1 contract](./phase-01-start.md)
- [Existing WDC Sol-high contract](../260820-1507-full-label-er-migration/research/wdc-sol-high-vertical-slice-contract.md)
- [Existing full WDC runner](../../labeller-screening/run_full_wdc.py)
- [Existing completed pass](../../data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv)
- [OpenRouter screening settings](../../labeller-screening/settings.json)

## Scope Boundary

In scope:

- Reuse the existing WDC result as pass 1.
- Add one minimal no-reuse option to the existing WDC runner.
- Run two new independent paid passes into separate output directories.
- Verify each pass has exactly 2,500 unique valid training-pair labels.
- Record per-pass request counts, invalid rows, retries, tokens, and cost.

Out of scope:

- A general multipass orchestration service.
- Majority/consistency code.
- Prompt variants.
- Gold-label comparison.
- Validation/test access.
- OOF or compact-model training.
- DBLP-ACM or Dataset 3 paid calls in this immediate phase.

## Why One Minimal Code Change Is Required

`labeller-screening/run_full_wdc.py` always reuses the original 300 screening
responses. Running it unchanged under new filenames would copy those same 300
labels into passes 2 and 3, falsely making one part of the dataset appear
perfectly self-consistent.

The underlying `run_setting(...)` function already supports a fresh run when
its reuse paths are omitted. The only required behavior change is therefore a
small explicit flag in `run_full_wdc.py`, for example:

```text
--no-reuse-existing
```

When the flag is present:

- Do not validate or load the 300 screening attempts as reusable predictions.
- Report `reused_completed_rows = 0`.
- Report `new_request_count = 2500`.
- Call `run_setting(...)` without reuse paths.

When the flag is absent, existing behavior must remain unchanged. No other
labeling abstraction is needed.

## Pass Naming and Storage

Do not move or rename the completed pass-1 artifacts. Record the mapping:

```text
pass_01 (existing)
data/cache/wdc_products/teacher_labels/full_sol_high/

pass_02 (new)
data/cache/wdc_products/teacher_labels/full_sol_high_pass_02/

pass_03 (new)
data/cache/wdc_products/teacher_labels/full_sol_high_pass_03/
```

Each new run keeps its normal internal artifact names under its own directory,
including predictions, attempts, audit, run, completion, and any resumable
inflight evidence. Directory-level naming is safer and simpler than renaming
only the final CSV because one paid pass produces several related files.

Later datasets follow the same pattern:

```text
teacher_labels/sol_high_pass_01/
teacher_labels/sol_high_pass_02/
teacher_labels/sol_high_pass_03/
```

Each pass must have an isolated output directory. Never point two passes at the
same attempts, predictions, audit, or inflight files.

## Scientific Invariants

All three passes must use the same:

- Teacher model and OpenAI-only OpenRouter routing.
- Reasoning effort.
- Prompt and response schema.
- Training-pair text and ordered pair IDs.
- Retry and timeout settings.
- Gold-free request construction.

The only intended differences are:

- Pass identity.
- Provider request execution time/request IDs.
- Output directory.
- Resulting stochastic teacher judgment, if any.

No human label, validation label, test label, previous-pass label, or consensus
label may appear in a request.

## Implementation Steps

### 1. Register the existing result as pass 1

- Verify the existing WDC Sol-high predictions still contain 2,500 unique
  official training pairs and valid binary labels.
- Record that 300 responses originated from the completed screening run and
  2,200 from the completed production extension.
- Do not make any API call and do not copy or move its artifacts.
- Create only a small human-readable pass map if later implementation needs one.

### 2. Add and test `--no-reuse-existing`

- Add the optional flag to `labeller-screening/run_full_wdc.py`.
- Preserve the current default behavior exactly.
- When enabled, omit both reuse paths from `run_setting(...)`.
- Extend focused tests for:
  - default 300-row reuse behavior;
  - zero-reuse dry-run counts;
  - rejection of mixed/partial reuse arguments if applicable;
  - no gold/validation/test path access.

No new production runner, service, database, or generic scheduler should be
introduced for this phase.

### 3. Dry-run pass 2

Render a dry run using the same source/input/settings and a new output path:

```bash
.venv/bin/python labeller-screening/run_full_wdc.py \
  --output-dir data/cache/wdc_products/teacher_labels/full_sol_high_pass_02/predictions \
  --no-reuse-existing
```

The dry run must report:

```text
full_training_rows = 2500
reused_completed_rows = 0
new_request_count = 2500
dry_run = true
```

It must make zero API calls.

### 4. Review price and authorize pass 2

- Review current OpenRouter pricing immediately before execution.
- Set a positive pass-specific USD ceiling with headroom over the dry-run
  estimate; do not reuse an old pricing assumption silently.
- Confirm the API key exists without printing it.
- Obtain explicit researcher approval for the paid pass.
- Add `--confirm-paid-labeling` and the approved ceiling only after review.

Command shape:

```bash
.venv/bin/python labeller-screening/run_full_wdc.py \
  --output-dir data/cache/wdc_products/teacher_labels/full_sol_high_pass_02/predictions \
  --no-reuse-existing \
  --spend-ceiling-usd <approved-pass-2-ceiling> \
  --confirm-paid-labeling
```

### 5. Verify pass 2 before proceeding

- Require 2,500 unique pair IDs and valid labels.
- Require zero missing and zero extra official training pairs.
- Confirm the run did not reuse pass-1 attempts.
- Review invalid rows, retries, stop reason, tokens, and total cost.
- Resume the same pass-2 directory if interrupted; do not start a replacement
  pass under a new name unless the existing pass is conclusively unusable.

Pass 3 remains blocked until pass 2 is complete and reviewed.

### 6. Dry-run, authorize, and execute pass 3

Repeat exactly the pass-2 procedure with only the output path and approved
ceiling changed:

```bash
.venv/bin/python labeller-screening/run_full_wdc.py \
  --output-dir data/cache/wdc_products/teacher_labels/full_sol_high_pass_03/predictions \
  --no-reuse-existing
```

After dry-run and explicit approval:

```bash
.venv/bin/python labeller-screening/run_full_wdc.py \
  --output-dir data/cache/wdc_products/teacher_labels/full_sol_high_pass_03/predictions \
  --no-reuse-existing \
  --spend-ceiling-usd <approved-pass-3-ceiling> \
  --confirm-paid-labeling
```

Verify it independently using the same checks as pass 2.

### 7. Produce a lightweight three-pass inventory

Without calculating consensus yet, record for each pass:

- Pass name and artifact path.
- Model, prompt, reasoning, and routing identity.
- Input count and input digest already emitted by the runner.
- Completed/invalid/retry counts.
- API calls, tokens, and USD cost.
- Final prediction CSV path.

Confirm that all three prediction files contain identical ordered pair-ID sets.
Do not join hidden gold and do not select a final teacher label in this phase.

## DBLP-ACM and Dataset 3 Follow-On

The same operational idea applies later: invoke the existing dataset-neutral
runner three times with `pass_01`, `pass_02`, and `pass_03` output directories.
The generic runner currently permits only `--fake`, so enabling real DBLP and
Dataset 3 calls will require a separate small, reviewed paid-dispatch phase that
reuses the existing OpenRouter JSON-Schema client and confirmation/cost gates.
Do not solve that future need by complicating the immediate WDC phase.

## Tests Before Paid Calls

- Existing labeler-screening suite passes.
- New no-reuse tests pass.
- Repository regression suite passes for touched behavior.
- Pass-specific dry run reports 2,500 new requests and zero reused rows.
- Output directory is absent or contains a valid resumable run for that same
  pass; it must not contain artifacts from another pass.
- Validation/test and gold files are not opened.

## Risks and Mitigations

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| Screening rows are accidentally reused | Artificially inflated consistency | Require zero-reuse count for passes 2-3 |
| Two passes share a directory | Results overwrite or resume the wrong pass | Use isolated pass directories and verify before confirmation |
| Exact-repeat model is deterministic | All three passes may be identical | Record the result honestly; prompt variants require a later reviewed decision |
| Interrupted paid run is restarted elsewhere | Duplicate spend and ambiguous pass identity | Resume the same directory and pass name |
| Pricing changed | Ceiling or estimate is wrong | Review current pricing immediately before each paid pass |
| Gold reaches the request path | Invalid experiment | Retain blinded inputs and focused no-gold tests |

## Todo

- [ ] Approve reuse of the completed WDC result as `pass_01`.
- [x] Implement and test the minimal `--no-reuse-existing` flag.
- [x] Dry-run and review pass 2.
- [x] Approve and complete paid pass 2.
- [x] Verify pass 2 before pass 3.
- [ ] Dry-run and review pass 3.
- [ ] Approve and complete paid pass 3.
- [ ] Verify all three pass inventories and aligned pair IDs.

## Success Criteria

- Three independently identified WDC pass artifacts exist.
- Pass 1 is the existing completed Sol-high result; passes 2 and 3 each contain
  2,500 freshly requested judgments with no reused screening responses.
- Every pass contains exactly the same 2,500 unique official training pair IDs
  and one valid label per pair.
- All calls use the same frozen teacher configuration and blinded training
  inputs.
- Paid execution stays within separately approved ceilings and records measured
  cost.
- No gold, validation, test, majority, OOF, detector, or refinement operation
  occurs in this phase.
- The next phase can consume three prediction files without re-running or
  relocating any paid artifacts.

## Next Steps

Implementation evidence (2026-09-28): the flag is implemented and the 13-test
screening suite passes. The pass-2 dry run reports 2,500 rows, zero reuse,
2,500 new requests, and `dry_run=true`. Existing pass-1 CSV contains 2,500
unique IDs with valid binary labels. No paid pass was started. The printed
pricing snapshot is dated 2026-08-22 and requires review before paid execution.

Pass-2 completion evidence (2026-09-29): 2,500/2,500 valid unique labels,
zero reused responses, zero retries, and USD 2.709705 total cost. Prediction,
attempt, audit, and run artifacts passed exact input-ID alignment, frozen-model,
provider, request-hash, cost-ceiling, and response-ID checks. The focused suite
passes 13/13 and the repository suite passes 181/181.

After researcher review of the three-pass inventory, add a small phase that
computes majority labels and `3/3` versus `2/3` consistency offline. That phase
must not make API calls and must keep hidden gold separate until evaluation.
