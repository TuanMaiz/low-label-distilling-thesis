---
phase: 5
title: "Pilot Student Runs And Direct LLM Baseline"
status: in_progress
priority: P1
effort: "1-2 weeks"
dependencies: [4]
---

# Phase 5: Pilot Student Runs And Direct LLM Baseline

## Overview

Run the smallest experiment that can decide whether the thesis direction has
signal: gold-label compact student, direct LLM matcher, random LLM-label
distilled student, and one actively selected LLM-label distilled student.

The FLAN-T5-base validation pilot returned on 2026-07-15 and produced a revise
decision: the active student improved macro F1 and accuracy over random LLM
labels but did not improve the primary match F1. The first ModernBERT-base
binary-classifier diagnostic also returned and received `REVISE` after its
three arms collapsed toward single-class predictions. Both result sets are
preserved as validation-only evidence; neither justified touching test.

Execution tooling initially prepared on 2026-07-13 includes the Colab dependency
file, resumable runner, aggregator, and runbook. The runtime now
uses 8-token binary targets and generations, hardware-aware mixed precision and
validation batches, one-time tokenization, token-weighted validation loss, and
stage-boundary recovery with stale-artifact archiving, atomic completion
markers, and commit/configuration/target-hash contracts. Evaluation captures
structured local student timing and throughput, and aggregation computes signed
deltas versus both random controls. A repaired ModernBERT job, a separate
`flan-t5-base-full-input` job, and a `qwen3-reranker-0-6b` LoRA job are prepared
for continued model screening on the same fixed inputs. Full-input FLAN raises
its contract to 2,700 tokens; Qwen preserves its pretrained yes/no reranking
interface under an audited 4,096-token complete-input contract. None of these
screening jobs has returned experiment results. Phase 5 remains in progress
until a compact student is strong enough for a meaningful active-versus-random
study.

## Requirements

- Functional: train and evaluate `gold_random`, `llm_random`, and `llm_active_bucketed_v1` at budget `128`.
- Functional: add `mixed_gold_llm_active` only if pure active LLM labels are noisy but promising.
- Functional: expand to budget `256` only after fixed `256` selection manifests exist.
- Functional: include the fixed direct LLM matcher result and cost from Phase 3 in the same pilot table.
- Non-functional: save checkpoints, predictions, metrics, and run metadata in stable output folders.
- Decision: end this phase with a clear continue/revise/stop recommendation.

## Ready Inputs

As of 2026-07-10, the 128-budget Phase 5 inputs are ready:

- Gold random target:
  `data/cache/wdc_products/targets/train_128.gold_random.targets.jsonl`
- Random LLM-label target:
  `data/cache/wdc_products/targets/train_128.llm_random.openai-gpt-5-4-mini.targets.jsonl`
- Active LLM-label target:
  `data/cache/wdc_products/targets/train_128.llm_active_bucketed_v1.openai-gpt-5-4-mini.targets.jsonl`
- Validation gold target:
  `data/cache/wdc_products/targets/validation.label_only.targets.jsonl`
- Test gold target:
  `data/cache/wdc_products/targets/test.label_only.targets.jsonl`
- Direct LLM validation baseline:
  `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.predictions.jsonl`
- Direct LLM validation cost summary:
  `outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.cost.json`

These inputs are allow-listed for version control so a fresh clone of the
committed branch can pass the Colab preflight without rebuilding Phase 3 or
Phase 4. The Colab runner reads the validation target but never reads the test
target.

## Architecture

```text
targets for 128 random and active selected pairs
  -> train compact student
  -> validation metrics
direct LLM predictions
  -> validation metrics and inference cost
student + direct LLM metrics
  -> random-vs-active same-budget comparison
  -> test metrics for promising variants
  -> pilot decision table
```

## Related Code Files

- Configs: `/mnt/d/Study/Cao-hoc/luan-van/code/configs/students/`
- Reranker backend: `/mnt/d/Study/Cao-hoc/luan-van/code/models/generative_reranker_student.py`
- Generic training: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/train_student.py`
- Legacy FLAN training: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/train_mt5.py`
- Reuse: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/evaluate_student.py`
- Colab runner: `/mnt/d/Study/Cao-hoc/luan-van/code/scripts/run_phase05_colab.sh`
- Aggregator: `/mnt/d/Study/Cao-hoc/luan-van/code/experiments/aggregate_phase05_results.py`
- Colab requirements: `/mnt/d/Study/Cao-hoc/luan-van/code/requirements-colab.txt`
- Runbook: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/reports/phase-05-colab-runbook.md`
- Student outputs: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/students/<student_id>/`
- Direct baseline: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/direct_llm/`

## Implementation Steps

1. Confirm or map the existing `gold_label`/`label_only` baseline as `gold_random` at budget `128`.
2. Load or run the fixed `direct_llm_matcher` baseline on the validation set or predeclared validation sample.
3. Train `llm_random` student at budget `128`.
4. Train `llm_active_bucketed_v1` student at budget `128` after its manifest and teacher cache are fixed.
5. If cost and time allow, train `mixed_gold_llm_active` at budget `128`.
6. Repeat the same student set at budget `256` only if the sampler/selector has been extended.
7. Evaluate all student models on the same validation split used for the direct LLM comparison.
8. Evaluate the best or most informative variants on the test split only after the validation decision.
9. Aggregate metrics into one pilot table:
   - F1.
   - precision.
   - recall.
   - accuracy.
   - invalid-output rate.
   - selection strategy.
   - direct LLM inference cost.
   - teacher-labeling cost.
   - estimated student inference cost.
   - signed match F1, macro F1, and accuracy deltas versus `llm_random` and
     `gold_random`.
10. Package the compact Colab handoff with the student ID in its filename,
    excluding large checkpoint weights.
11. Write a pilot decision note only after the returned validation artifacts
    have been reviewed.

## Fixed Colab Runtime Defaults

- `MAX_INPUT_LENGTH=2400` for ModernBERT classifiers, 512 for the historical
  FLAN configuration, 2,700 for `flan-t5-base-full-input`, and 4,096 for the
  Qwen reranker. The full-input FLAN value preserves its measured 2,649-token
  fixed-input maximum; Qwen preflight measures its exact formatted prompts
  before training.
- `MAX_TARGET_LENGTH=8` and `MAX_NEW_TOKENS=8` apply to seq2seq students only.
- Sequence classifiers map logits to literal `match` / `non-match` prediction
  text and record both probabilities; they do not generate tokens.
- Training batch defaults to 1 for the Qwen reranker, 16 for classifiers, and 4
  for seq2seq students. Qwen accumulates 16 microbatches for an effective batch
  of 16.
- `PRECISION=auto`: BF16 on supporting CUDA GPUs, FP16 with gradient scaling on
  other CUDA GPUs, and FP32 on CPU smoke checks.
- `VALIDATION_BATCH_SIZE=auto`: 4 for long-input seq2seq, otherwise 32 with
  BF16 CUDA, 16 with other CUDA, and the training batch size on CPU.
- Training, validation, and final-prediction inputs are tokenized once per
  process; truncation is disabled for classifiers and the full-input FLAN
  diagnostic, while the historical seq2seq configuration keeps 512 tokens.
  Qwen uses dynamic left padding, truncation disabled, and a persisted
  `input_length_audit.json`.
- Validation loss is weighted by non-padding label-token count so early
  stopping and checkpoint selection are invariant to validation batch grouping.
- Threshold calibration for FLAN is a documented optional follow-up using
  validation sequence-likelihood ratios; it is not part of the full-input run.

## Recovery Contract

- A checkpoint plus atomically replaced `training_summary.json` marks a
  completed training stage.
- Atomically replaced validation predictions plus metrics mark a completed
  evaluation stage.
- Training and evaluation each write an atomic contract containing the Git
  commit, relevant runtime configuration, and SHA-256 hashes of targets and
  upstream contract inputs.
- The first preflight resolves the Hugging Face model/tokenizer repository to
  one immutable commit in the run's `student_config.json` and records exact
  Python/package versions in `runtime_provenance.json`.
- A run-level contract hashes that provenance and fixes the actual device name,
  resolved precision, and resolved validation batch across every variant in one
  `OUTPUT_ROOT`.
- A stage is reusable only when both completion artifacts and its current
  contract match. A missing or mismatched contract blocks reuse; use a new
  `OUTPUT_ROOT` or explicit `FORCE=1` to replace it intentionally.
- An interrupted training process restarts that variant from the beginning; it
  does not claim unsupported mid-epoch resume.
- Before forced retraining or reevaluation, prior contracts, summaries, and
  downstream validation artifacts are renamed with `.stale.<timestamp>`
  suffixes.
- Replacing a run-level identity archives all active variant directories.
  Partial aggregation independently includes only variants whose current
  training and evaluation contracts match shared overrides.
- Compact result packaging includes the run-level contract and both stage
  contracts for every variant.
- Qwen completion additionally requires `best_adapter/`, the merged standalone
  `best_model/`, the validation threshold, and `checkpoint_manifest.json`.
  That manifest verifies the byte size and SHA-256 of every adapter and
  merged-model file before reuse, evaluation, or packaging. Optional checkpoint
  archives retain both the adapter and merged model.

## Student Inference Evidence

- Validation measures student inference locally on the selected Colab device
  and records synchronized generation seconds, total evaluation wall time,
  rows per second, seconds per pair, device name, precision, batch size, and
  sequence limits in `validation.metrics.json`.
- A small OpenRouter model is not a valid price proxy for the local FLAN-T5
  student because it is a different model served on an opaque provider stack.
  Instead, training records synchronized wall seconds and aggregation applies
  all low/base/high rates predeclared in
  `configs/phase05_cost_assumptions.json` to measured training and inference
  time: low = $0.25, base = $1.00, and high = $4.00 per GPU-hour. The rates are
  analytical sensitivity assumptions, not observed charges or provider quotes.
- The aggregate table preserves these timing fields and reports signed match
  F1, macro F1, and accuracy deltas for each student against `llm_random` and
  `gold_random`. Positive means the row outperforms the named reference.
- The cost-scenario table reports one-time teacher plus training cost, per-pair
  student inference cost, savings at the fixed direct-baseline scale, and the
  first whole-number query count where the student reaches cost parity or
  becomes cheaper. Break-even is null when the measured student per-pair cost
  is not below direct matching. The priced training boundary is the synchronized
  trainer loop stated in `training_time_scope`, not model download or tokenization.

## Success Criteria

- [x] FLAN-T5 pilot metrics table exists.
- [x] FLAN-T5 predictions are saved for each run.
- [x] Direct LLM baseline quality and cost are included.
- [x] The `llm_random` and `llm_active_bucketed_v1` gaps from `gold_random` are quantified.
- [x] The active-vs-random gain or loss at the same budget is quantified.
- [x] Cost gap between direct LLM matching and distilled student inference is quantified.
- [x] FLAN-T5 continue/revise/stop decision is `REVISE`.
- [x] Config-driven ModernBERT-base validation tooling is implemented and verified.
- [x] First ModernBERT-base validation archive is returned from Colab.
- [x] First ModernBERT comparison and `REVISE` decision are written.
- [x] Repaired ModernBERT and full-input FLAN diagnostic tooling is implemented and verified.
- [x] Qwen3-Reranker-0.6B LoRA diagnostic tooling is implemented and verified.
- [ ] Repaired ModernBERT validation archive is returned and reviewed.
- [ ] Full-input FLAN validation archive is returned and reviewed.
- [ ] Qwen reranker validation archive is returned and reviewed.
- [ ] A compact student is selected for the Phase 6 full-budget study.

## Risk Assessment

- Risk: FLAN-T5 output parsing introduces invalid outputs.
  Mitigation: keep labels compact and track invalid-output rate.
- Risk: validation result is noisy at low budgets.
  Mitigation: treat pilot as directional; only expand after consistent signal.
- Risk: active selection improves recall but damages precision.
  Mitigation: report precision/recall separately and inspect hard-negative false positives.
- Risk: no positive signal.
  Mitigation: use failure analysis to decide whether mixed labels, simpler selection, or encoder classifier is needed.
- Risk: direct LLM comparison is accused of cherry-picking.
  Mitigation: use the fixed evaluation set or a predeclared sample before inspecting direct LLM results.
