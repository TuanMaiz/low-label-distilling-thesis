---
title: "soft-target-arm"
description: "Arm S on WDC: soft targets (0.5 teacher + 0.5 OOF student) for detected uncertain pairs instead of hard flip/remove/duplicate; plus extra seeds."
status: in-progress
priority: P1
effort: "Offline phases done; GPU: about 10 runs (S seed 42, then seeds 43 and 44 for A, G, C2, S); API: about USD 3.3 for the C2 evidence check"
branch: feat/soft-target-arm
tags: [treatment, soft-labels, wdc, seeds]
blockedBy: [GPU choice]
blocks: []
created: 2026-11-12
---

# soft-target-arm

Contract: WDC contract amendment A6 (`plans/261023-2015-wdc-guideline-judge-treatment/contract.md`).

## Phases

1. **Trainer soft targets (done, 5272f0f).** Optional `soft_target` per training row →
   class probabilities for cross-entropy; hard-only files unchanged. Tests run on CPU
   torch in the cloud workspace.
2. **Arm S bundle (done, 067379e).** From the frozen arm A file; 158 detected pairs get
   `0.5 × teacher + 0.5 × p_OOF`; rebuild byte-identical; tracked in git.
3. **Runner route + amendment A6 (done).** `run_wdc_qwen_treatment_arms.sh ... S`;
   other arms' commands unchanged.
4. **GPU session (blocked on GPU choice).** S seed 42; seeds 43 and 44 for A, G, C2, S.
   C2 needs the WDC evidence check v2 first (`run_evidence_check_v2 run --dataset wdc`,
   about USD 3.3) and its bundle `category-rule-C2`.
5. **Analysis and report.** S vs A, S vs G, C2 vs A; mean ± SD over seeds; report to the
   supervisor.

## Open decisions

1. Sign off amendment A6 (arm S, beta = 0.5, seeds 43 and 44).
2. GPU: V100-SXM2-32GB family, or a new card with every arm rerun.
