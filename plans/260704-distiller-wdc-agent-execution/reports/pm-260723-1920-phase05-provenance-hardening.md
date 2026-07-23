---
title: Phase 5 Provenance Hardening
status: completed
date: 2026-07-23
plan: 260704-distiller-wdc-agent-execution
phase: 5
---

# Phase 5 Provenance Hardening

## Summary

| Item | Result |
|---|---|
| Scope | Colab screening tooling only |
| Experiment status | Phase 5 remains in progress |
| Unit tests | 87 passed |
| GPU results | None produced |
| Teacher/test artifacts | Untouched |

## Completed

- Corrected Qwen's system prefix to the exact pretrained reranker template.
- Pinned one immutable Hugging Face model/tokenizer commit per run snapshot.
- Added `runtime_provenance.json` for exact Python and ML package versions.
- Prevented failed reconnect checks from overwriting recorded provenance.
- Added size and SHA-256 coverage for every adapter and merged-model file.
- Verified checkpoint contents before reuse, evaluation, and packaging.
- Archived all active arms when a run-level identity is forcibly replaced.
- Filtered partial summaries through current training/evaluation contracts so
  shared-override changes cannot mix stale arms.

## Plan Sync

| Phase | Done | Open | Status |
|---|---:|---:|---|
| 1 | 4 | 0 | Completed |
| 2 | 3 | 0 | Completed |
| 3 | 10 | 1 | Completed; remaining item optional |
| 4 | 5 | 2 | Completed; remaining items conditional |
| 5 | 12 | 4 | In Progress |
| 6 | 0 | 6 | Pending |
| 7 | 0 | 9 | Pending |
| 8 | 0 | 5 | Pending |

No phase status changed. The four Phase 5 open criteria still require returned
Colab diagnostics and a student-selection decision.

## Verification

- Independent code review: clean.
- Focused recovery/debug pass: clean.
- `.venv/bin/python -m unittest discover -s tests`: 87 passed.
- Shell syntax, Python compilation, and `git diff --check`: passed.

## Next

1. Commit and push the reviewed tooling.
2. Run Qwen3-Reranker-0.6B screening on Colab with a fresh output root.
3. Return the compact archive; do not inspect test.
