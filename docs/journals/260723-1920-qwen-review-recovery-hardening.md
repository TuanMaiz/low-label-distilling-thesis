---
date: 2026-07-23
session: qwen-review-recovery-hardening
---

# Journal: 2026-07-23 — Qwen Review and Recovery Hardening

## Context

Review of the Qwen3 reranker implementation found three reproducibility gaps,
then validation exposed two related recovery hazards. The fixes harden the
predeclared screening workflow without changing its fixed training targets,
validation data, or model-selection question.

## What Happened

- Corrected the Qwen prompt prefix to match the official reranker token
  sequence exactly, with no extra whitespace before `Judge`.
- Made Hugging Face resolution immutable: preflight resolves the model
  repository commit into `student_config.json` and records Python, Torch,
  Transformers, PEFT, Accelerate, and Hugging Face Hub versions in
  `runtime_provenance.json`. The run contract hashes both artifacts.
- Extended the checkpoint manifest to record and verify the size and SHA-256 of
  every adapter and merged-model file before reuse, evaluation, or packaging.
- Prevented a failed reconnect from overwriting the provenance needed to
  diagnose a contract mismatch. With `FORCE=1`, the prior provenance is
  archived before replacement.
- When forced run identity replacement is necessary, every existing experiment
  arm is archived so later aggregation cannot mix environments.
- Made partial aggregation contract-aware: only arms matching the current
  shared runtime and training overrides are included; stale arms are reported
  as missing.

## Reflection

The original completion markers proved that files existed, but not that they
were produced from identical dependencies or remained byte-for-byte intact.
That distinction matters for a multi-arm Colab run that may span reconnects.
Run identity, model identity, and checkpoint contents now fail closed instead
of allowing a convenient but scientifically ambiguous resume.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Treat prompt bytes as part of the pretrained interface | A whitespace change alters tokenization at the scoring boundary | The screening uses Qwen's intended reranking prompt |
| Pin the resolved model revision and package environment | Floating revisions or dependency ranges could silently differ after reconnect | All arms share a verifiable runtime identity |
| Hash complete adapter and merged checkpoints | Path and presence checks cannot detect replacement or corruption | Reuse and packaging validate actual checkpoint content |
| Archive all arms when replacing run identity | Keeping apparently completed arms would mix incompatible environments | Forced recovery restarts the comparison cleanly |
| Filter partial aggregation by contracts | Shared override changes can invalidate only some existing outputs | Interim reports omit stale results rather than combining them |

## Verification

The repository test suite passed: **87 tests passed**. No GPU training or
validation result was produced. Teacher-label artifacts and the fixed test
split were untouched.

## Next Steps

- Run the predeclared Qwen3 reranker screening on an A100 in Colab.
- Return the compact archive for comparison across `gold_random`,
  `llm_random`, and `llm_active_bucketed_v1`.
