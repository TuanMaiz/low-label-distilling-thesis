---
title: Small model family comparison for WDC student screening
date: 2026-07-21 23:01 +07:00
status: recommendation
scope: Qwen, Llama, Mistral, Gemma, and selected open alternatives
---

# Small model family comparison for WDC student screening

## Contents

1. [Summary](#summary)
2. [Family comparison](#family-comparison)
3. [Qwen family](#qwen-family)
4. [Recommended screening ladder](#recommended-screening-ladder)
5. [Implementation implications](#implementation-implications)
6. [References](#references)
7. [Unresolved questions](#unresolved-questions)

## Summary

Qwen remains the best family for the next diagnostic. Use
`Qwen3-Reranker-0.6B` first because it is small, established, public,
Transformers-compatible, and pretrained for joint pair scoring. If it is too
weak, use `Qwen3-Reranker-4B` as the capacity step. Use `Qwen3-0.6B` only if a
generic causal-LM control is scientifically useful.

For a different family, the best operational candidate is
`mistralai/Ministral-3-3B-Instruct-2512`: current, Apache-2.0, ungated, and long
context. It is roughly a 4B package with multimodal machinery, however, and has
no pair-specific pretraining. Llama 3.2 1B and Gemma 3 1B/270M are technically
reasonable but require accepting custom terms and Hugging Face gated access;
that conflicts with the established no-token/no-terms requirement.

Popularity improves recognizability and reproducibility, but should not outrank
task fit. The Qwen reranker is already popular at roughly 2.6M monthly Hugging
Face downloads at screening time.

## Research methodology

- Checked 2026-07-21.
- Sources: official model cards, vendor documentation, and official collections.
- Criteria: public access, license, adoption, model size, full-pair context,
  pair-task alignment, stable Transformers support, and A100 feasibility.
- No benchmark value below is treated as an Entity Matching result; vendor
  retrieval and general-language benchmarks do not predict WDC match F1.

## Family comparison

| Family candidate | Parameters/context | Access | Fit for WDC student | Decision |
|---|---|---|---|---|
| `Qwen3-Reranker-0.6B` | 0.6B / 32K | Apache-2.0, ungated | Joint pair scoring and pointwise yes/no loss | **First choice** |
| `Qwen3-Reranker-4B` | 4B / 32K | Apache-2.0, ungated | Same task-aligned interface with more capacity | Capacity fallback |
| `Qwen3.5-0.8B` | 0.8B / 262K | Apache-2.0, ungated | Very recent and popular, but generic and currently requires bleeding-edge Transformers | Later, after stable support |
| `Llama-3.2-1B-Instruct` | 1B / 128K | Gated, custom Llama license | Popular and mature generic generator; no pair specialization | Exclude under access constraint |
| `Ministral-3-3B-Instruct-2512` | about 4B package / 256K | Apache-2.0, ungated | Strong non-Qwen family candidate; generic and materially larger | Second-family choice |
| `Gemma-3-270m` | 270M / 32K | Gated, Gemma terms | Very cheap, but likely capacity-limited and requires terms | Exclude under access constraint |
| `Gemma-3-1b-it` | 1B / 32K | Gated, Gemma terms | More plausible than 270M, but still generic and gated | Exclude under access constraint |
| `Phi-4-mini-instruct` | 3.8B / 128K | MIT, public | Mature Transformers path and strong general model | Optional fallback; larger and not pair-specific |

## Qwen family

### Qwen3 rerankers

| Member | Role in this project |
|---|---|
| `Qwen3-Reranker-0.6B` | Best first diagnostic. Smallest task-aligned member. |
| `Qwen3-Reranker-4B` | Best escalation if 0.6B is below the useful range. It tests capacity without changing the pair-scoring objective. |
| `Qwen3-Reranker-8B` | Too large for the initial compact-student screen. Keep only as a possible upper bound. |

The official family uses instruction-aware generative reranking. It scores the
`yes` and `no` tokens for an instruction/query/document input. Official training
support includes pointwise binary cross-entropy, matching the WDC pair-label
shape without implementing a blocking stage.

### Qwen3 embedding models

`Qwen3-Embedding-0.6B`, `4B`, and `8B` encode texts into vectors. The 0.6B model
is extremely popular, but its natural role is retrieval, blocking, clustering,
or a bi-encoder similarity baseline. Encoding Record A and Record B separately
loses some fine-grained token interaction available to a cross-encoder
reranker. Do not choose it as the next pair-classification student while
blocking remains out of scope.

### Generic Qwen3 models

- `Qwen3-0.6B`: stable Transformers support, 32K context, cheapest generic Qwen
  baseline.
- `Qwen3-1.7B`: more capacity while remaining reasonably small.
- `Qwen3-4B`: stronger but overlaps the reranker 4B cost tier without its
  pair-specific post-training.

Generic models can be trained to output match/non-match text, but 128 examples
must teach more of the comparison behavior. They are useful as controls, not as
the highest-probability route to better WDC F1.

### Qwen3.5 models

- `Qwen3.5-0.8B`: latest compact Qwen, 262K context, highly recognizable.
- `Qwen3.5-2B`: more capable generic member at moderate size.
- `Qwen3.5-4B`: stronger again, but Hugging Face reports about 5B stored
  parameters and the experiment becomes less compact.

These models use a newer hybrid architecture and include a vision encoder.
Their official card currently asks for the latest Transformers from source,
whereas the repository intentionally pins stable Transformers 4.57.x. Adopting
one now would combine a model experiment with a dependency/toolchain experiment.
Wait for stable support or isolate it in a separate Colab requirements profile.
There is no official Qwen3.5 reranker in the screened collection, so its recency
does not automatically make it a better WDC student than Qwen3-Reranker.

## Family details

### Llama

`Llama-3.2-1B-Instruct` is popular, has 128K context, and stable Transformers
support. It is a reasonable generic answer-only student. However, Hugging Face
requires sharing contact information and accepting the Llama 3.2 Community
License. Because avoiding access approval was already a project requirement,
Llama is operationally disqualified unless that requirement changes.

### Mistral

`Ministral-3-3B-Instruct-2512` is the best non-Qwen choice. It is current,
Apache-2.0, ungated, multilingual, and supports 256K context. The model card
describes a 3.4B language model plus a 0.4B vision encoder, and the Hugging Face
package is approximately 4B parameters. It should therefore be treated as a
larger, more expensive general-model screen, not a peer of the 0.6B reranker.

### Gemma

Gemma 3 offers excellent size points: 270M and 1B, both with 32K context. The
270M member may be too small for subtle product identity decisions; 1B is the
more plausible student. Both require reviewing and accepting Google's Gemma
usage license on Hugging Face, so neither meets the current friction-free Colab
requirement. Newer Gemma releases do not remove the licensing issue relevant to
this workflow.

### Phi

`Phi-4-mini-instruct` is a credible public fallback: MIT license, 3.8B
parameters, 128K context, and integration from Transformers 4.49 onward. It is
not pair-specialized and is much larger than Qwen3-Reranker-0.6B, so it ranks
behind Ministral as a family-diversity screen rather than ahead of Qwen.

## Recommended screening ladder

1. `Qwen3-Reranker-0.6B`: best probability/cost balance.
2. `Qwen3-Reranker-4B`: only if 0.6B is promising but under target; capacity
   escalation with the same methodology.
3. `Ministral-3-3B-Instruct-2512`: cross-family check if Qwen-specific behavior
   needs to be ruled out.
4. `Qwen3.5-0.8B`: later generic-current-model check after stable Transformers
   support, or under an explicitly isolated dependency profile.

Do not screen every family. With three training arms per model, a wide model
grid quickly consumes Colab time without improving the active-versus-random
thesis evidence. Use validation-stage stopping rules.

## Implementation implications

Qwen3-Reranker needs the planned `generative_reranker` backend: causal-LM
`yes`/`no` token scoring, pointwise training, and validation-calibrated threshold.
That backend can later support generic Qwen and Llama-style answer-only models,
but their chat templates and target-token behavior must remain model-specific.

Ministral 3 and Qwen3.5 are multimodal model packages even for text-only use.
They should not be forced into the same loader until their text-only loading,
checkpoint saving, and merged-adapter behavior are verified. Llama and Gemma
also require authenticated Hugging Face downloads, which would reintroduce the
token setup deliberately removed from the Colab workflow.

## References

- [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Official Qwen3 family](https://qwenlm.github.io/blog/qwen3/)
- [Official Qwen3.5 collection](https://huggingface.co/collections/Qwen/qwen35)
- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Ministral 3 3B Instruct](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)
- [Gemma 3 270M](https://huggingface.co/google/gemma-3-270m)
- [Gemma terms](https://ai.google.dev/gemma/terms)
- [Phi-4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)

## Unresolved questions

- Whether the experiment needs a cross-family control at all; the main thesis
  comparison is selection strategy, not model-family ranking.
- Whether a 4B student remains acceptably compact under the intended inference
  cost argument. This should be decided using measured student inference time,
  not parameter count alone.

