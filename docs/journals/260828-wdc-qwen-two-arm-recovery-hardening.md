---
date: 2026-08-28
session: wdc-qwen-two-arm-recovery-hardening
---

# Journal: 2026-08-28 — WDC–Qwen Two-Arm Recovery Hardening

## Context

CPU-only review of the WDC–Qwen `gold` versus `llm_hard` runner found two
recovery paths that could combine artifacts from different attempts or
overwrite evidence from an interrupted evaluation. The fixes preserve the
approved training configuration and do not expand the experiment scope.

## What Happened

- Evaluation recovery now classifies outputs as `empty`, `complete`, or
  `partial`. Any temporary prediction or metrics file, or only one finalized
  evaluation file, is `partial` and stops for inspection. Evaluation resumes
  only when neither finalized nor temporary output exists.
- Training recovery now validates the persisted checkpoint manifest and
  requires the copy embedded in `training_summary.json` to match it exactly.
  A summary from one run therefore cannot authorize evaluation of a different
  otherwise-valid checkpoint.
- Regression tests cover temporary evaluation files, mixed finalized outputs,
  and a mismatched embedded checkpoint manifest.

## Reflection

Existence checks were insufficient at the two recovery boundaries. A temporary
file is evidence of interrupted work, not permission to overwrite it, and two
independently valid files are not necessarily evidence that they belong to the
same run. Recovery now fails closed unless the evaluation state is clean and
the training summary is cryptographically bound to the persisted checkpoint
inventory.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Stop on any partial evaluation state | Preserve interrupted-run evidence for manual inspection | Recovery cannot silently overwrite `.tmp` files or accept one-sided finalized output |
| Require exact embedded/persisted checkpoint-manifest equality | Bind the training summary to the checkpoint selected for evaluation | Cross-run summary/checkpoint substitution is rejected |
| Keep verification CPU-only in this session | The current machine is not the approved RTX 3090 runtime | Workflow logic is verified without starting experiment execution |

## Verification

- WDC–Qwen focused suite: **21/21 passed**.
- Affected workflow suites: **20/20 passed**.
- Full repository suite: **132/132 passed**.
- Labeler-screening suite: **12/12 passed**.
- Independent code review: **PASS** with no remaining blocking finding.

No GPU training, evaluation, model loading, or LLM/API call ran. The full
two-arm execution remains pending on the rented GPU.

## Next Steps

1. Commit and push the reviewed recovery changes.
2. Pull that exact commit on the rented RTX 3090 environment.
3. Run setup and preflight before explicitly starting the `gold` arm.
