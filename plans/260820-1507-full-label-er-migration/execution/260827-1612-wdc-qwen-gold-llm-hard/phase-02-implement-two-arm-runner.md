---
phase: 2
title: "Implement Fail-Closed Two-Arm Runner"
status: pending
priority: P1
effort: "6-10h"
dependencies: [1]
---

# Phase 2: Implement Fail-Closed Two-Arm Runner

## Context Links

- Authorization: `./phase-01-authorize-full-validation-experiment.md`
- Existing runner: `/mnt/d/study/cao-hoc/luan-van/code/scripts/run_wdc_qwen_vertical_slice.sh`
- Existing training API: `/mnt/d/study/cao-hoc/luan-van/code/experiments/train_student.py`
- Existing evaluation API: `/mnt/d/study/cao-hoc/luan-van/code/experiments/evaluate_student.py`

## Overview

Extend the narrow runner with explicit, confirmation-gated gold and
`llm_hard` actions. Reuse the existing trainer/evaluator; add only orchestration,
finite-loss protection, immutable per-arm contracts, and output verification.

## Requirements

- Functional: `train-gold`, `train-llm-hard`, `verify-results`, `package-arm`,
  and `package-results` actions.
- Functional: `train-llm-hard` refuses to start until gold verifies completely.
- Functional: current RTX 3090 preflight must pass before either arm.
- Functional: one small offline script confirms both committed targets contain
  2,500 unique rows with identical ordered pair IDs and non-label data.
- Non-functional: exact complete runs skip without mutation; incomplete or
  mismatched runs stop for inspection.
- Non-functional: runner contains no test path and invokes no LLM/provider code.

## Architecture

```text
action + explicit confirmation
  -> current preflight contract check
  -> arm-specific immutable contract
  -> train_student (full validation each epoch)
  -> checkpoint manifest check
  -> evaluate_student (reload best_model; persisted threshold)
  -> output verifier + atomic completion contract
```

Per-arm output roots:

```text
<output-root>/gold/run/
<output-root>/llm_hard/run/
```

Only selected target path, `variant`, and output root differ between arms.

## Frozen Full-Run Command Fields

| Field | Value |
|---|---|
| Batch / validation batch | `1` / `1` |
| Gradient accumulation / effective batch | `16` / `16` |
| Epoch limit / patience | `10` / `3` |
| Optimizer / LR / weight decay | AdamW / `2e-4` / `0.01` |
| Schedule / warmup ratio | linear / `0.10` |
| Planned steps / warmup steps | `1570` / `157` |
| Maximum length / truncation | `4096` / disabled |
| Precision / device | `auto` / `cuda` |
| Checkpoint and threshold metric | validation macro F1 |

## Related Code Files

- Modify: `/mnt/d/study/cao-hoc/luan-van/code/scripts/run_wdc_qwen_vertical_slice.sh`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/wdc_qwen_preflight.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/trainer.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/train_student.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/scripts/check_wdc_target_alignment.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_wdc_qwen_preflight.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_phase05_runtime.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_wdc_target_alignment.py`
- Reuse unchanged: `/mnt/d/study/cao-hoc/luan-van/code/experiments/evaluate_student.py`
- Reuse unchanged: `/mnt/d/study/cao-hoc/luan-van/code/utils/artifact_contract.py`
- Reuse unchanged: `/mnt/d/study/cao-hoc/luan-van/code/utils/checkpoint_manifest.py`

## Tests Before

1. Add shell-contract tests for the five new actions and explicit confirmation.
2. Assert exact arm-to-target mapping and full warmup `0.10` versus smoke `0.0`.
3. Test the alignment script against count, duplicate-ID, order, pair-ID, and
   non-label-payload mismatches; do not judge either arm's labels.
4. Add negative tests for T4/CPU/stale preflight, test-path access, and LLM code.
5. Add state tests for exact-complete skip, incomplete refusal, contract mismatch,
   and evaluation-only recovery after complete training.
6. Add corruption tests for duplicate/missing predictions, non-finite or
   unnormalized probabilities, metric mismatch, and invalid checkpoint manifest.
7. Add a training-loss test proving NaN/Inf stops before an optimizer step.

## Implementation Steps

1. Parse actions and require `--confirm-full-training` for each paid-GPU arm.
2. Run `check_wdc_target_alignment.py`; keep it a small committed-data check,
   not a publication or upstream validator.
3. Require current default 3090 output root and rerun `preflight` before training.
4. Build one shared function parameterized only by arm, target, and output path.
5. Write/check a per-arm artifact contract binding Git commit, narrow contract,
   preflight contract, runtime, input audit, target, validation, config, relevant
   code hashes, and every frozen CLI field.
6. Add immediate finite training-loss checks in `Trainer.train_epoch`.
7. Persist completed epochs and actual optimizer steps in `training_summary.json`.
8. Invoke `train_student` with the frozen full settings.
9. Verify checkpoint manifest, then invoke `evaluate_student` on validation only.
10. Verify exactly 2,500 ordered unique IDs, zero invalids, finite normalized
   probabilities, metrics recomputed from predictions, and equality among the
   summary, checkpoint, and evaluation thresholds.
11. Record train-stage wall time separately from existing trainer-only time;
   preserve inference-only and evaluation wall timing already emitted.
12. Write an atomic completion contract only after every check succeeds.
13. Permit evaluation-only recovery only when training summary and checkpoint
    manifest are complete and the arm contract still matches.
14. Preserve partial/mismatched artifacts without overwriting or auto-retrying.

## Todo List

- [ ] Tests fail before new behavior exists.
- [ ] Gold and `llm_hard` actions share one implementation path.
- [x] The small alignment script passes on the committed targets.
- [ ] Full settings and provenance are bound per arm.
- [ ] Result verifier covers IDs, probabilities, metrics, and checkpoint files.
- [ ] No test, LLM, or publication-upstream dependency enters the runner.

## Success Criteria

- [ ] Gold maps only to `gold.jsonl`; `llm_hard` maps only to `llm_hard.jsonl`.
- [ ] Both resolve the same precision, planned schedule, warmup, and runtime
  properties; actual steps may differ only through early stopping.
- [ ] No silent overwrite, mid-epoch resume, hyperparameter change, or CPU fallback.
- [ ] Complete outputs verify idempotently; corruption fails closed.

## Risk Assessment

Long runs amplify resume and provenance mistakes. The plan allows no mid-training
resume: preserve a failed attempt, review it, quarantine explicitly, and restart
under the unchanged contract only with researcher approval.

## Security and Data Integrity

Do not export OpenRouter/OpenAI keys to model-training jobs. Do not call the
separate target publication validator. Hash only declared files and never
package `.env`, ignored upstream caches, or unrelated outputs.

## Next Steps

Proceed to Phase 3 after focused and full local tests pass.
