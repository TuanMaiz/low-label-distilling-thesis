---
title: Compact student model screening after FLAN-T5 and ModernBERT
date: 2026-07-21
status: recommendation
scope: Phase 5 model screening; fixed WDC budget-128 targets and validation split
---

# Compact student model screening

## Decision

The next new student to screen should be
[`ibm-granite/granite-embedding-reranker-english-r2`](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2).
It is a public, Apache-2.0, 149M-parameter, 8,192-token cross-encoder reranker.
Unlike the first ModernBERT experiment, its pretraining already teaches the model
to score a pair jointly for relevance. That objective is not identical to Entity
Matching, but it is closer to binary pair classification than generic masked
language-model pretraining.

This is a model-screening diagnostic, not a change to the thesis question or to
the fixed experiment inputs. It should use the same three budget-128 arms,
validation rows, seed, and cost reporting as the other Phase 5 students.

## Screening criteria

- Public and ungated, with no Hugging Face access-token requirement.
- Permissive license suitable for reproducible thesis artifacts.
- Compact enough for an A100 Colab run; approximately one billion parameters or
  fewer.
- Native context long enough to test the complete fixed WDC pair inputs.
- Pairwise or ranking-specific pretraining preferred over another generic
  encoder-only language model.
- Practical integration with the existing Transformers training pipeline.

## Ranked shortlist

| Rank | Candidate | Size/context | Why consider it | Main limitation |
|---:|---|---|---|---|
| 1 | [Granite embedding reranker English R2](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2) | 149M / 8,192 | Compact cross-encoder with ranking-specific training; Apache-2.0; native Transformers sequence-classification architecture | Uses one scalar relevance logit, so the current two-logit backend cannot preserve its pretrained head unchanged |
| 2 | [BAAI BGE reranker v2 M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | about 568M / 8,192 | Established multilingual cross-encoder reranker; Apache-2.0 | Larger and needs the same scalar-score support |
| 3 | [GTE multilingual reranker base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) | 306M / 8,192 | Cross-encoder and multilingual, potentially useful for mixed-language product records; Apache-2.0 | Model card requires `trust_remote_code=True`, increasing reproducibility and code-review burden |
| 4 | [Qwen3 reranker 0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) | 0.6B / long context | Recent multilingual generative reranker; Apache-2.0 | Decoder/generative scoring needs substantially more backend work and compute |
| 5 | [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | 395M / 8,192 | Directly compatible and larger than the current ModernBERT-base student | Same generic pretraining family as the failed first diagnostic; capacity alone may not solve the low-label failure |

## Required integration for the recommended model

The current `sequence_classification` backend forces `num_labels=2`, trains
integer class labels, and evaluates a two-logit softmax. Granite R2 declares a
single sequence-classification score and a sigmoid activation. Loading it with
`num_labels=2` would replace or mismatch the pretrained ranking head, removing
the most useful reason to test the model.

Add a narrowly scoped scalar pair-scoring backend that:

1. retains the model's pretrained one-logit head;
2. maps `non-match`/`match` to floating-point 0/1 targets and trains with binary
   cross entropy with logits;
3. calibrates the match threshold only on validation data and persists it in the
   same way as the repaired classifier pipeline;
4. reuses complete pair tokenization, staged unfreezing, macro-F1 checkpoint
   selection, timing, contracts, and cost aggregation;
5. measures the actual maximum tokenized WDC pair length with this tokenizer,
   then disables truncation and sets the configured limit above that maximum.

Do not pad every example to the model's full 8,192-token capability. Use the
measured dataset maximum or dynamic padding to avoid unnecessary attention and
memory cost.

## Proposed screening order

1. Run the already prepared repaired ModernBERT-base and full-input FLAN-T5-base
   diagnostics so their predeclared changes remain interpretable.
2. Implement and run Granite R2 reranker as the next new-model diagnostic.
3. If Granite still collapses or remains far below the useful quality range,
   choose BGE reranker v2 M3 for greater capacity and multilingual coverage.
4. Use GTE only if multilingual coverage becomes the leading hypothesis, and
   use Qwen3 only if the encoder rerankers fail enough to justify a new
   generative scoring backend.

## Interpretation guardrail

A reranker result tests whether pairwise relevance pretraining improves
low-label student learning. It does not implement DistillER's blocking stage and
does not change the thesis into a reranking or Ditto fine-tuning thesis. The
primary comparison remains active versus random LLM labeling at the same 128
teacher calls.

## Sources

- [IBM Granite embedding reranker English R2 model card](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2)
- [IBM Granite reranker configuration](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2/blob/main/config.json)
- [BAAI BGE reranker v2 M3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Alibaba GTE multilingual reranker base model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base)
- [Qwen3 reranker 0.6B model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [ModernBERT-large model card](https://huggingface.co/answerdotai/ModernBERT-large)

