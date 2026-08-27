---
date: 2026-08-27
session: wdc-qwen-two-arm-planning
---

# Journal: 2026-08-27 — WDC–Qwen Two-Arm Plan

## Context

The T4 smoke run passed LoRA training, checkpoint reload, and evaluation. The next scope is the full WDC–Qwen comparison between benchmark-gold and `llm_hard` training labels.

## What Happened

- `ck:plan` created the WDC–Qwen gold-versus-`llm_hard` implementation plan.
- Red-team suggestions for a separate validation hash and cross-arm execution-session machinery were rejected as redundant.
- Git remains the authority for committed code, target files, and the validation input.
- Pair equivalence will be checked by one small offline script: both arms must contain the same 2,500 unique pair IDs and identical non-label inputs.
- The training workflow will not invoke the upstream publication validator.
- Added `scripts/check_wdc_target_alignment.py` and its focused tests in `tests/test_wdc_target_alignment.py`.
- The real target check passed: each arm has 2,500 rows and 2,500 unique IDs; ordered pair IDs and input text align exactly; the 79 label disagreements remain permitted.
- Verification passed both the focused suite (5/5) and the full suite (118/118).
- No training, LLM/API calls, publication validation, or official validation/test predictions ran.

## Reflection

The smoke result provides sufficient evidence that the training, checkpoint, reload, and evaluation plumbing works. The plan is stronger after removing controls that duplicated Git while retaining the one comparison-specific check that matters: both arms must differ only in their training labels. That check is now implemented and verified against the committed WDC targets, so the next implementation concern is the two-arm runner rather than further target validation machinery.

## Decisions Made

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Run `gold` before `llm_hard` | Fixed, simple execution order | The completed gold artifacts must be preserved before the second arm starts. |
| Use full-run warmup `0.10` | Restore the frozen training configuration after the one-step smoke exception | Both arms receive identical full-training hyperparameters. |
| Run fresh preflight on RTX 3090 | The T4 smoke does not establish the rental runtime | Runtime provenance is captured for the actual experiment environment. |
| Evaluate validation only | Official validation is the authorized comparison split | The test split remains locked. |
| Use one offline pair-alignment script | Confirm a fair label-source comparison without publication rederivation | Both 2,500-row arms are checked locally; no upstream evidence is required. |

## Next Steps

1. Add the full `gold` and `llm_hard` runner commands.
2. Verify that both arms use identical full-training and validation settings.
3. Run setup and preflight on the rented RTX 3090 before explicit training approval.

The offline alignment step is complete. No full training was run during this session.
