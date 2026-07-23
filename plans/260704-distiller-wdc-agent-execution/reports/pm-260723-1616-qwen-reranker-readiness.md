---
title: Qwen3 reranker LoRA tooling readiness
status: ready-for-colab
date: 2026-07-23
scope: phase-05-student-screening
---

# Qwen3 Reranker LoRA Tooling Readiness

## Summary

| Gate | Status | Evidence |
|---|---|---|
| Config-driven backend | Ready | `generative_reranker` config and dispatch |
| Full-input protection | Ready | exact-prompt preflight audit; truncation disabled |
| Low-label adaptation | Ready | rank-8 LoRA; best-adapter preservation |
| Validation decision | Ready | macro-F1 checkpointing and persisted threshold |
| Standalone handoff | Ready | safe adapter merge into `best_model/` |
| Recovery/provenance | Ready | audit, manifests, stage contracts, packaging |
| Local verification | Pass | 82 tests; compile and shell syntax checks |
| GPU experiment | Pending | no Qwen Colab result inspected yet |

## Delivered

- Added `Qwen/Qwen3-Reranker-0.6B` as a causal-LM reranker student.
- Mapped complete Record A/B pairs to the official instruct/query/document
  prompt and final `no`/`yes` token scores.
- Added dynamic left padding, 4,096-token configurable preflight auditing, and
  explicit overflow failure.
- Added LoRA, gradient checkpointing, microbatch 1, 16-step accumulation, and
  optimizer-step-aware scheduling.
- Preserved the selected adapter, safely merged it into a standalone model,
  and reused the validation-selected threshold during evaluation.
- Extended Colab recovery, contracts, result packaging, checkpoint packaging,
  runbook, and project guidance.

## Constraints Preserved

- Fixed budget-128 training targets and validation rows unchanged.
- No teacher LLM calls.
- Test target untouched.
- No local GPU training or claimed Qwen quality result.
- Existing FLAN-T5 and ModernBERT backends remain supported.

## Remaining Risk

The exact Qwen tokenizer audit and 0.6B model/PEFT loading have unit coverage
but still require the first real A100 Colab preflight and training process.
The run must use a fresh output root and retain all generated contracts.

## Next Step

Run the predeclared Qwen diagnostic on A100 for `gold_random`, `llm_random`,
and `llm_active_bucketed_v1`; return the compact results archive for validation
comparison before touching the test split.
