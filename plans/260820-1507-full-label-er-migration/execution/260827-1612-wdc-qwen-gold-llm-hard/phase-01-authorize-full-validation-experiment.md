---
phase: 1
title: "Authorize Full Validation Experiment"
status: pending
priority: P1
effort: "2-3h"
dependencies: []
---

# Phase 1: Authorize Full Validation Experiment

## Context Links

- Parent contract: `../../research/wdc-qwen-training-vertical-slice-contract.md`
- Parent plan: `../../plan.md`
- Reviewed smoke summary: `../../../../outputs/qwen3-reranker-0-6b/smoke/run/training_summary.json`
- Reviewed smoke metrics: `../../../../outputs/qwen3-reranker-0-6b/smoke/run/validation.metrics.json`

## Overview

Record the T4 smoke review and change only the narrow WDC-Qwen authorization:
permit one full gold arm and one full `llm_hard` arm with full-validation
checkpoint/threshold selection. Keep the final test locked.

## Requirements

- Functional: authorize 2,500-row training and 2,500-row validation for both arms.
- Functional: predeclare order `gold` then `llm_hard` and one completion per arm.
- Non-functional: preserve all frozen model and training settings.
- Non-functional: distinguish T4 smoke provenance from future RTX 3090 provenance.

## Architecture

The contract remains the human authorization layer. The runner implements it;
artifacts cannot broaden it. Validation is used during training to select the
best checkpoint and deterministic threshold, then used once after checkpoint
reload to publish validation metrics. It is not final-test evidence.

## Related Code Files

- Modify: `/mnt/d/study/cao-hoc/luan-van/code/plans/260820-1507-full-label-er-migration/research/wdc-qwen-training-vertical-slice-contract.md`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/plans/260820-1507-full-label-er-migration/plan.md`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/plans/260820-1507-full-label-er-migration/phase-04-finalize-compact-cross-encoder-models.md`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/plans/260820-1507-full-label-er-migration/phase-05-refactor-experiment-runner.md`
- Modify together after implementation: `/mnt/d/study/cao-hoc/luan-van/AGENTS.md`, `/mnt/d/study/cao-hoc/luan-van/code/AGENTS.md`, `/mnt/d/study/cao-hoc/luan-van/code/CLAUDE.md`
- Reference only: `/mnt/d/study/cao-hoc/luan-van/code/outputs/qwen3-reranker-0-6b/`

## Implementation Steps

1. Record the reviewed T4/FP16 smoke as successful plumbing evidence: finite
   loss, 16/16 valid predictions, verified 22-file checkpoint manifest.
2. Leave rented-3090 setup/preflight unchecked; require fresh runtime identity.
3. Authorize only WDC-Qwen gold and `llm_hard` full training and validation.
4. Freeze identical settings and restore full warmup ratio `0.10`.
5. Freeze arm order, output grammar, completion definition, and failure policy.
6. State that target publication validation stays separate from GPU training.
7. Preserve the explicit prohibition on all test predictions and LLM calls.
8. Update parent Phase 4/5 wording without marking either global phase complete.

## Todo List

- [ ] Smoke evidence recorded without promoting its F1 to an experiment result.
- [ ] Full validation-only authorization added.
- [ ] Test and LLM boundaries remain explicit.
- [ ] Parent plan and guidance agree with the narrow contract.

## Success Criteria

- [ ] Authorization changes scope only; no hyperparameter or input identity changes.
- [ ] Both arms use the same official validation truth and selection policy.
- [ ] T4 timing is excluded from RTX 3090 experimental timing.
- [ ] Phase 3 remains globally in progress for the other two datasets.

## Risk Assessment

The main risk is silently treating the successful T4 smoke as full experimental
authorization. Mitigate by recording it as plumbing evidence and requiring a
new 3090 preflight before either full arm.

## Security and Data Integrity

No API key is needed. Gold training truth enters only the gold arm. Validation
truth never enters either training target. The test split remains inaccessible.

## Next Steps

Proceed to Phase 2 only after the authorization diff is reviewed.
