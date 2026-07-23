---
date: 2026-07-23
session: qwen3-reranker-lora-tooling
---

# Journal: 2026-07-23 — Qwen3 Reranker LoRA Tooling

## Context

The first FLAN-T5 and ModernBERT screenings did not reach the desired student
quality under the fixed 128-label budget. This iteration prepared
`Qwen/Qwen3-Reranker-0.6B` as a third compact-student diagnostic while
preserving the thesis comparison: `gold_random`, `llm_random`, and
`llm_active_bucketed_v1` use the same fixed targets and validation set.

## What Happened

- Added a `generative_reranker` backend that formats complete product pairs
  with Qwen's instruct/query/document contract and maps final `no`/`yes` token
  logits to non-match/match probabilities.
- Added pointwise cross-entropy training over only those two logits, dynamic
  left padding, and a 4,096-token preflight audit with truncation disabled.
  Over-limit pairs now fail explicitly instead of losing either record.
- Added LoRA configuration and training with rank 8, alpha 16, dropout 0.05,
  attention projection targets, gradient checkpointing, microbatch 1, and
  16-step gradient accumulation.
- Extended validation to select checkpoints by macro F1, calibrate a validation
  decision threshold, preserve the selected adapter, and merge it into the
  standalone model used for evaluation.
- Extended Colab recovery, contracts, manifests, and packaging to require and
  retain the input-length audit, adapter, merged model, threshold, and
  checkpoint manifest.
- Kept the existing seq2seq and sequence-classification paths config-driven,
  including architecture-aware input-length defaults and explicit truncation
  behavior.

## Verification

The repository test suite passed: **82 tests passed**. Coverage includes prompt
formatting, no-truncation overflow failure, dynamic padding, answer-token
validation, yes/no logit scoring and loss, label ordering, LoRA setup and
merging, persisted threshold reuse, gradient accumulation, and Phase 5
contracts.

No GPU training or validation experiment was run during this implementation.
Therefore, there is no Qwen quality, runtime, memory, or cost result yet.
Teacher-label artifacts and the fixed test split were neither modified nor
evaluated.

## Reflection

Keeping a separate reranker backend was worthwhile: it reuses shared training,
metrics, threshold, and orchestration code without forcing Qwen's causal-LM
answer-token interface into the FLAN-T5 generation path or ModernBERT
classification-head path. LoRA reduces trainable state and overfitting risk for
128 examples, but long-context activation memory remains an empirical A100
constraint.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Preserve Qwen's `yes`/`no` reranking interface | Retains pretrained reranking behavior instead of adding a new classifier head | Requires a dedicated causal-LM backend |
| Use LoRA for the first screening | Lower optimizer memory and fewer updated parameters suit the low-label setting | Adapter is preserved and the selected checkpoint is merged for standalone inference |
| Require complete inputs under a configurable 4,096-token cap | Avoids silent record truncation and remains adaptable to future datasets | Preflight blocks any row above the declared cap |
| Calibrate on validation macro F1 | A fixed 0.5 threshold may be poor under class imbalance | The selected threshold becomes part of the checkpoint contract |

## Limitations

- The 4,096-token cap has not yet been validated against the real Qwen-tokenized
  fixed inputs on Colab.
- A100 memory use and training duration remain estimates until the diagnostic
  runs.
- With only 128 labels, the reranker may still collapse or overfit despite
  LoRA; the implementation does not establish model quality.

## Next Steps

- Run the predeclared Qwen3 reranker screening on an A100 in Colab using a fresh
  output root.
- Return the compact archive and compare all three validation arms with the
  existing FLAN-T5, ModernBERT, and direct-LLM baselines.
- Keep teacher artifacts and the test split untouched until the validation
  screening decision is made.
