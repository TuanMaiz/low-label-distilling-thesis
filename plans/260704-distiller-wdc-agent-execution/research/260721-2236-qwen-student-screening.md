---
title: Qwen-family student screening for Phase 5
date: 2026-07-21 22:36 +07:00
status: recommendation
supersedes: 260721-2210-compact-student-screening.md model recommendation
---

# Qwen-family student screening for Phase 5

## Summary

Use [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
as the next new student. It is the smallest official Qwen3 reranker, is public
under Apache-2.0, has 0.6B parameters and 32K context, supports more than 100
languages, and is substantially more recognizable than the Granite candidate.
The Hugging Face Qwen collection showed roughly 2.7M downloads and 375 likes at
screening time.

Prefer the reranker checkpoint over generic `Qwen3-0.6B` and
`Qwen3-Embedding-0.6B`. The reranker is already trained to judge two texts
jointly. The generic checkpoint adds no pair-task specialization; the embedding
checkpoint is a bi-encoder intended to encode records separately.

This is still a model-screening diagnostic. It does not change the fixed
budget-128 targets, evaluation split, active/random comparison, or thesis claim.

## Methodology

- Sources: official Qwen Hugging Face cards and QwenLM GitHub repository.
- Currency: checked 2026-07-21.
- Criteria: recognition/adoption, public access, license, parameter count,
  context, pair-task alignment, Transformers compatibility, Colab feasibility.

## Qwen options

| Model | Size | Native task | Fit for this experiment |
|---|---:|---|---|
| `Qwen3-Reranker-0.6B` | 0.6B | Generative pair reranking | Best: smallest Qwen reranker; preserves pair-scoring knowledge |
| `Qwen3-Embedding-0.6B` | 0.6B | Bi-encoder embedding | Useful for retrieval/blocking, but blocking is out of current scope |
| `Qwen3-0.6B` | 0.6B | General causal LM | Could generate match/non-match, but lacks reranker specialization |
| `Qwen3-Reranker-4B` | 4B | Generative pair reranking | More capacity, but no longer a compact first screen and costs much more |

## Model behavior

Qwen3-Reranker formats an instruction, query, and document into one causal-LM
input. It does not generate a long explanation. At the final position, it reads
the logits for the tokens `yes` and `no`, then normalizes those two scores into a
relevance probability. Official Qwen training documentation describes its
pointwise reranker loss as binary cross-entropy, which maps naturally to WDC
`match` and `non-match` labels.

Recommended task mapping:

```text
Instruction: Given two product records, decide whether they refer to the same
real-world product. Answer yes for a match and no for a non-match.
Query:       <complete Record A>
Document:    <complete Record B>
Target:      yes | no
```

Use the normalized `yes` probability for validation threshold calibration. This
keeps the existing persisted-threshold contract while avoiding autoregressive
generation during validation and student inference.

## Integration impact

This is not a config-only addition. The current backends load either
`AutoModelForSeq2SeqLM` or a two-logit `AutoModelForSequenceClassification`.
Qwen3-Reranker uses `AutoModelForCausalLM` and its pretrained `yes`/`no` token
logits. Add a `generative_reranker` architecture that reuses:

- the same target JSONL rows and label mapping;
- complete-pair input enforcement with truncation disabled;
- validation-only macro-F1 threshold selection and persistence;
- checkpoint/run contracts, timing, aggregation, and output layout;
- identical training settings across `gold_random`, `llm_random`, and
  `llm_active_bucketed_v1`.

The repository already pins Transformers 4.57+, exceeding Qwen's documented
minimum of 4.51, so no Transformers upgrade is required.

## Training recommendation

- Start with the 0.6B reranker only.
- Use the model-specific tokenizer to measure the maximum tokenized fixed pair;
  configure a cap above that measured maximum and fail on overflow.
- Use BF16 on A100, gradient checkpointing, microbatch 1 or 2, and gradient
  accumulation for the effective batch.
- Predeclare either full fine-tuning or LoRA and use it unchanged for all three
  arms. LoRA is the safer first A100/Colab configuration for memory and
  low-label regularization; merge the adapter for standalone student inference
  if the runtime pipeline requires one checkpoint.
- Keep answer-only training. Do not add generated rationales.
- Preserve fixed validation and teacher artifacts; do not touch test labels.

## Risks

- Reranking pretraining means relevance, not product identity. It is closer than
  generic language modeling but does not guarantee strong Entity Matching.
- 0.6B decoder inference is slower than the 149M ModernBERT classifier, although
  it should remain far cheaper than repeated GPT-5.4-mini calls.
- With only 128 labels, any 0.6B model can overfit. Validation checkpointing,
  identical arms, and predeclared training settings remain necessary.
- Instructions can affect reranker results. Freeze one Entity Matching
  instruction before inspecting comparative outcomes.

## Next steps

1. Retain the already prepared ModernBERT repair and full-input FLAN diagnostics.
2. Add the `generative_reranker` backend and one Qwen3-Reranker-0.6B student
   config.
3. Unit-test token scoring, labels, no-truncation behavior, threshold reuse, and
   recovery contracts.
4. Run all three fixed budget-128 arms under a fresh output root on A100.
5. Judge it by match F1 and macro F1 first; accuracy remains secondary because
   the validation distribution is imbalanced.

## References

- [Qwen3-Reranker-0.6B model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Official Qwen3 Embedding and Reranker repository](https://github.com/QwenLM/Qwen3-Embedding)
- [Official Qwen reranker training guide](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md)
- [Transformers Qwen3 documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3)

## Unresolved questions

- Full fine-tuning versus LoRA should be declared before implementation. The
  recommended operational default is LoRA, but full fine-tuning is feasible on
  an A100 for this model and may better preserve comparability with earlier
  students.

