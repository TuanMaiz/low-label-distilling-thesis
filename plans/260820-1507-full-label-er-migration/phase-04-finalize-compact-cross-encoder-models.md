---
phase: 4
title: "Finalize Compact Cross-Encoder Models"
status: pending
priority: P1
effort: "4-6d"
dependencies: [1, 2]
---

# Phase 4: Finalize Compact Cross-Encoder Models

## Overview

Turn the three Phase-1-selected models into uniform, immutable compact
cross-encoder ER configurations and prove each backend can train, score,
threshold, and reload without dataset-specific shortcuts.

## Context Links

- Scientific decisions: `./phase-01-freeze-experiment-contract.md`
- Narrow WDC/Qwen smoke contract: `./research/wdc-qwen-training-vertical-slice-contract.md`
- Config parser: `/mnt/d/study/cao-hoc/luan-van/code/models/student_config.py` (legacy identifier)
- Existing backends: `/mnt/d/study/cao-hoc/luan-van/code/models/classification_student.py`, `/mnt/d/study/cao-hoc/luan-van/code/models/generative_reranker_student.py` (legacy identifiers)

## Narrow WDC/Qwen boundary

Before global Phase 4 is complete, the separate WDC/Qwen contract authorizes
only rented-RTX-3090 setup, preflight, and a tiny balanced LoRA smoke. Retain
the old full-run Qwen config and hyperparameters without tuning. Zero warmup is
a smoke-only plumbing exception; the frozen full-run warmup remains `0.10`.
Full two-arm training and official full-validation/test predictions remain
blocked pending smoke review and explicit approval.

## Requirements

- Exactly three included compact-model configs, each with explicit repository
  identity, `included`, backend, training, and tokenization
  settings and proof that it jointly encodes both records. The loader asserts
  exactly three configs with `included: true`.
- Freeze tokenizer, separator/prompt, maximum length/truncation, label/logit
  mapping, tuning method, optimizer, schedule, batch/accumulation, epochs/early
  stopping, precision, threshold selection, and checkpoint markers before
  result inspection.
- The same model config serves gold- and LLM-supervision arms; only target
  path/source differs. Run each cell once; never choose reruns by outcome.
- Audit input lengths over all datasets. Unsafe truncation fails or follows one
  predeclared pair-aware policy equally across supervision sources.

## Architecture

Keep the existing `StudentConfig` API (legacy identifier) and train/evaluate
interface with architecture-specific adapters behind it. Persist runtime and
input-length provenance, and hash the used config into each run manifest.

## Related Code Files

- Modify: `/mnt/d/study/cao-hoc/luan-van/code/models/student_config.py` (legacy identifier)
- Modify as eligible: `/mnt/d/study/cao-hoc/luan-van/code/models/classification_student.py` (legacy identifier)
- Modify as eligible: `/mnt/d/study/cao-hoc/luan-van/code/models/generative_reranker_student.py` (legacy identifier)
- Retain historical only unless selected: `/mnt/d/study/cao-hoc/luan-van/code/models/seq2seq_student.py` (legacy identifier)
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/train_student.py`, `/mnt/d/study/cao-hoc/luan-van/code/experiments/evaluate_student.py` (legacy identifiers)
- Reuse: `/mnt/d/study/cao-hoc/luan-van/code/utils/classification_threshold.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/runtime_provenance.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/checkpoint_manifest.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/peft_runtime.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/torch_runtime.py`
- Create/replace: `/mnt/d/study/cao-hoc/luan-van/code/configs/students/{three_model_ids}.json` (`students` is a legacy directory name)
- Extend: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_student_backends.py` (legacy identifier), `/mnt/d/study/cao-hoc/luan-van/code/tests/test_generative_reranker.py`, `/mnt/d/study/cao-hoc/luan-van/code/tests/test_classification_threshold.py`

## Tests Before

Write failing config tests for model identity, cross-encoder type, pair-input
contract, match-label mapping, truncation policy, included/exact-three discovery,
and forbidden supervision-source-dependent hyperparameters. Add tiny mocked
forward, save/reload, threshold, length-boundary, precision, and checkpoint tests
for every selected architecture.

## Implementation Steps

1. Add `included` discovery and exact-three validation to the legacy-named
   `StudentConfig` API.
2. Retire active references to unselected screening configs without deleting
   historical results; create exactly three final compact-model configs.
3. Normalize pair splitting/tokenization and match-probability output across
   selected classifier/reranker backends.
4. Run all-dataset input-length audits and freeze equal pair handling for both
   supervision sources.
5. Smoke-train each model on a tiny balanced fixture; evaluate, persist
   threshold/checkpoint manifest, reload, and reproduce probabilities.
6. Document hardware/precision and resource-stop settings in each config;
   never choose a config from final test performance.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Three final configs | Load with model identity and eligibility metadata |
| Same model, gold vs LLM supervision | Identical hyperparameters/config hash |
| Over-length pair | Predeclared behavior; no silent asymmetric truncation |
| Swapped label mapping | Test fails before training |
| Save/reload tiny checkpoint | Same match probabilities/threshold |
| OOM/resource ceiling in smoke | Revise scientific plan; no post-hoc shrink |
| 2 or 4 included configs | Loader fails with actionable IDs |

## Success Criteria

- [ ] Exactly three eligible compact-model configs pass smoke tests.
- [ ] Dataset lengths are audited and both supervision arms share all model
  settings except target provenance.
- [ ] Checkpoints, thresholds, and runtime provenance reload safely.
- [ ] Run-manifest inputs identify each model config unambiguously.

## Risk Assessment

Architecture misclassification, label-map inversion, truncation, and GPU OOM
can invalidate the comparison. Mitigate with eligibility evidence, boundary
tests, resource gates, small smoke runs, and probability round trips.

## Security/Data Integrity

Use public model repositories, record licenses/config hashes, avoid remote code
unless audited, and validate checkpoint manifests before evaluation or reuse.

## Next Steps

Phase 5 combines these three compact models with Phase-3 supervision targets.
