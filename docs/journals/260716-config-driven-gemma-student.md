---
date: 2026-07-16
session: config-driven-gemma-student
---

# Journal: 2026-07-16 — Config-Driven Gemma Student

## Context

Phase 5 FLAN-T5 validation ended with a **REVISE** decision: active selection improved macro F1 and accuracy over random LLM labeling, but not the primary match F1 reliably. This iteration prepared a predeclared Gemma 3 270M diagnostic on the same targets and validation data, while making the student workflow config-driven rather than FLAN-specific.

## What Happened

- Added two explicit student backends: seq2seq for FLAN-T5 and sequence classification for Gemma 3 270M.
- Added config-driven training, evaluation, Colab orchestration, contracts, aggregation, and packaging without changing the fixed teacher labels or selection manifests.
- Kept classifier inference native: Gemma produces logits, which are mapped and serialized as the same `match` / `non-match` prediction text consumed downstream.
- Assigned trained model ownership to `outputs/students/{student_id}/train_{budget}/`; the direct LLM baseline remains under `outputs/distiller_wdc/direct_llm/`.
- Documented that Gemma weights are gated on Hugging Face and require accepted terms plus authentication, while Transformers 4.57 or newer supplies the required model support.

## Reflection

The two-backend boundary keeps architecture-specific behavior explicit while sharing the experiment workflow and artifact contracts. This avoids a broad plugin abstraction and prevents classifier behavior from being forced through seq2seq generation assumptions. The completed work prepares the experiment but does not provide evidence about Gemma quality or runtime until the Colab GPU run is returned.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Maintain explicit seq2seq and classification backends | The architectures differ materially in targets, loss accounting, and inference | Additional compatible students can be selected by config without obscuring backend-specific behavior |
| Serialize classifier logits as `match` / `non-match` | Preserve the existing prediction and metric interface | Downstream evaluation stays comparable while invalid generated text is eliminated for classifiers |
| Store student artifacts under `outputs/students/` | Separate trained students from teacher and direct-LLM evidence | Each student and budget has a clear run-level config, runtime contract, and variant tree |
| Require Transformers 4.57+ and Hugging Face authentication for Gemma | Gemma 3 is library-supported, but its weights are gated | Colab setup can fail early with an actionable prerequisite message |
| Keep validation inputs fixed and leave test untouched | Preserve the anti-cherry-pick experiment contract | Gemma remains an architecture diagnostic directly comparable with the FLAN-T5 pilot |

## Verification

- All 57 repository tests passed.
- `scripts/run_phase05_colab.sh` passed Bash syntax validation.
- The final diff was checked for unintended changes and contract regressions.
- An independent code review found no blocking side effects or public-contract regressions.

## Limitations

- No Gemma weights were downloaded and no local or Colab GPU training was performed in this iteration.
- Validation artifacts therefore remain pending, and the fixed test split remains untouched.

## Next Steps

- Commit and push the config-driven student change set.
- Run Gemma 3 270M on Colab using the unchanged `train_128` inputs.
- Return the compact result archive for comparison with FLAN-T5 and the fixed direct LLM baseline.
