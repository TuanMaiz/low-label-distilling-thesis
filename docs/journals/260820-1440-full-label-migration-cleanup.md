---
date: 2026-08-20
session: full-label-migration-cleanup
---

# Journal: 2026-08-20 — Full-label migration cleanup

## Context

Started `refactor/full-label-er-migration` to replace the low-label and active-selection experiment with complete gold-label versus LLM-label training across three datasets.

## What Happened

- Retired the low-label sampler, active-pair selector, low-label preparation command, and obsolete Phase 3/4 orchestration scripts.
- Updated direct tests and active documentation while preserving historical artifacts and unrelated working-tree changes.
- Verified the repository with 94 passing tests and no dangling active imports of the removed modules.

## Decisions

- Keep prior plans, outputs, caches, and journals as research history.
- Remove only the obsolete execution path in this slice; defer dataset and runner generalization to the replacement contract.

## Next

- Write the full-label experiment contract, then generalize targets and execution across three datasets and the selected cross-encoder students.
