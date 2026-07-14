# Phase 5 Colab Readiness

Date: 2026-07-10
Plan: `plans/260704-distiller-wdc-agent-execution/plan.md`
Tags: `wdc-products`, `phase-05`, `colab`, `reproducibility`

## Context

Phase 5 needs three FLAN-T5-base students trained on a GPU, but the run will be
performed in Colab rather than this local workspace. The handoff therefore had
to work from a fresh clone of the fixed branch, resume safely after a Colab
disconnect, validate only the fixed validation split, and return enough
evidence for the Phase 5 decision without requiring large model downloads.

## What Changed

- Added `requirements-colab.txt` without PyTorch so Colab keeps its
  CUDA-compatible runtime build.
- Added `scripts/run_phase05_colab.sh` with setup, preflight, resumable
  per-variant training/evaluation, aggregation, and packaging commands.
- Added `experiments/aggregate_phase05_results.py` and focused tests to combine
  the three student validation results with the fixed direct-LLM quality and
  cost baseline.
- Added a Colab runbook covering branch cloning, Google Drive persistence,
  recovery, and artifact handoff.
- Updated the active plan and repository guidance to record that execution
  tooling is ready while Phase 5 metrics and success criteria remain pending.

## Fresh-Clone Reproducibility

The three canonical 128-row training targets and the canonical GPT-5.4-mini
direct-validation cost and prediction files are explicitly allow-listed for
version control. This lets a fresh clone pass preflight without rerunning the
Phase 3 teacher or rebuilding Phase 4 targets. Only those direct-LLM files are
unignored; stale or unrelated output files remain excluded.

Preflight enforces the expected branch, required files and row counts, and a
CUDA device. CPU execution is rejected unless explicitly allowed for a smoke
check. All three training targets contain 128 unique training pairs and have
zero pair overlap with the 2,500-row validation target. The runner contains no
teacher call and no test-target path.

## Verification

- Phase 5 aggregation tests: 2/2 passed.
- Full regression suite: 23/23 passed.
- Bash syntax check: clean.
- Branch, missing-input, and CPU guards: behaved as intended.
- Direct cost recomputation: `$0.7964985` across 2,500 validation rows.
- Git diff whitespace check: clean.

No Colab GPU training has run, so there are no new student metrics or Phase 5
research conclusions yet.

## Decisions

The default handoff is the compact
`phase05_train_128_results.tar.gz`, containing summaries, logs, validation
predictions and metrics, the aggregated table, fixed inputs, direct baseline,
and provenance. Large checkpoint archives are optional, generated separately
per student only if later model reuse or inspection requires them.

The Colab wrapper uses `MAX_TARGET_LENGTH=8` for the binary target strings
instead of the legacy rationale-oriented length of 192. The generic Python
training CLI keeps its historical default for backward compatibility.

This keeps the normal transfer small while preserving all artifacts needed to
compare `gold_random`, `llm_random`, `llm_active_bucketed_v1`, and the direct
LLM baseline. The validation decision must precede any test evaluation.

## Next Step

Commit and push the complete Phase 5 change set to
`codex/distiller-wdc-implementation`. Then clone that branch in a Colab GPU
runtime, run `scripts/run_phase05_colab.sh setup` followed by
`scripts/run_phase05_colab.sh all`, and return the compact results archive for
the Phase 5 continue/revise/stop decision.
