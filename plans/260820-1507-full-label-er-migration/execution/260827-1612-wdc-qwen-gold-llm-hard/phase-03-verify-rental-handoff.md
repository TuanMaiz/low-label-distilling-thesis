---
phase: 3
title: "Verify Implementation and Rental Handoff"
status: completed
priority: P1
effort: "3-5h"
dependencies: [2]
---

# Phase 3: Verify Implementation and Rental Handoff

## Context Links

- Runner implementation: `./phase-02-implement-two-arm-runner.md`
- Repository commands: `/mnt/d/study/cao-hoc/luan-van/code/AGENTS.md`

## Overview

Prove the runner state machine locally without model training, document exact
rental commands, and package an implementation commit before renting the 3090.

## Requirements

- Functional: dry-run fixtures exercise both arms and all failure branches.
- Functional: the handoff gives setup, preflight, gold, `llm_hard`, verify, and
  package commands in fixed order.
- Non-functional: no paid API, GPU training, official predictions, or mutation
  of existing smoke artifacts occurs during local verification.

## Architecture

Mocks and temporary fixtures validate orchestration. Real full targets are read
only by structural preflight/input audit. GPU-dependent model execution remains
deferred to Phase 4.

## Related Code Files

- Modify: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_wdc_qwen_preflight.py`
- Modify together: `/mnt/d/study/cao-hoc/luan-van/AGENTS.md`, `/mnt/d/study/cao-hoc/luan-van/code/AGENTS.md`, `/mnt/d/study/cao-hoc/luan-van/code/CLAUDE.md`
- Create after implementation: `/mnt/d/study/cao-hoc/luan-van/code/docs/journals/260827-wdc-qwen-full-validation-authorization.md`

## Implementation Steps

1. Run shell syntax and focused state-machine tests.
2. Run the full repository and labeler-screening suites.
3. Confirm the active runner contains no serialized test path or LLM client import.
4. Verify the publication builder remains unchanged and separate.
5. Run `git diff --check` and inspect staged scope for secrets/unrelated files.
6. Record the exact pushed implementation commit required on the rental machine;
   Git remains the authority for committed code and input identity.
7. Document that 3090 setup/preflight must create new runtime provenance; the
   copied T4 contract is evidence only and must not be reused.
8. Commit/push implementation before starting the rented machine.

## Verification Commands

```bash
bash -n scripts/run_wdc_qwen_vertical_slice.sh
.venv/bin/python -m unittest tests.test_wdc_qwen_preflight -v
.venv/bin/python -m unittest tests.test_wdc_target_alignment -v
.venv/bin/python -m unittest tests.test_phase05_runtime -v
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s labeller-screening/tests -v
git diff --check
```

## Todo List

- [x] Focused and full tests pass.
- [x] Static test/LLM/publication-validator boundaries pass.
- [x] Contract and guidance documentation are consistent.
- [x] Rental command handoff points to one pushed commit.

## Success Criteria

- [x] Implementation can be reviewed without starting training.
- [x] No unrelated dirty-worktree changes enter the implementation commit.
- [x] Rental setup is deterministic and fails on non-3090 hardware by default.

## CPU Verification Record

Verified on 2026-08-28 without CUDA, model loading, GPU training, official
full-validation prediction, test access, or paid API calls:

- Shell syntax: passed.
- Focused WDC–Qwen suite: 21/21 passed.
- Full repository suite: 132/132 passed.
- Labeler-screening suite: 12/12 passed.
- `git diff --check`: passed.
- Recovery regressions cover partial evaluation temporary files and mismatch
  between the training summary and persisted checkpoint manifest.
- Rental execution used pushed commit
  `bbbb419c074e6e6b4464f14fd44fbcf63175767e`; unrelated local untracked files
  did not enter that commit.

## Risk Assessment

Local mocks may miss GPU behavior. Mitigate with the already reviewed T4 smoke
and mandatory fresh 3090 setup/preflight before full execution.

## Security and Data Integrity

Secret scan must pass. Rental instructions use environment variables only for
non-secret path/runtime overrides; no API keys are required.

## Next Steps

Completed. The rented RTX-3090 checkout used the recorded pushed revision;
Phase 4 then executed both validation arms.
