---
phase: 4
title: "Finalize Cross-Encoder Students"
status: pending
priority: P1
effort: "4-6d"
dependencies: [1, 2]
---

# Phase 4: Finalize Cross-Encoder Students

## Overview

Turn the three Phase-1-selected models into uniform, immutable cross-encoder
student configurations and prove each backend can train, score, threshold, and
reload without dataset-specific shortcuts.

## Context Links

- Scientific decisions: `./phase-01-freeze-experiment-contract.md`
- Config parser: `/mnt/d/study/cao-hoc/luan-van/code/models/student_config.py`
- Existing backends: `/mnt/d/study/cao-hoc/luan-van/code/models/classification_student.py`, `/mnt/d/study/cao-hoc/luan-van/code/models/generative_reranker_student.py`

## Requirements

- Exactly three included student configs, each pinned to an immutable model
  revision, explicit `included` flag, backend, training, and tokenization
  settings, and proven to jointly encode both records. The loader asserts
  exactly three configs with `included: true`. No FLAN/ModernBERT/Qwen
  config is automatically promoted merely because it exists historically.
- Freeze per-model tokenizer, separator/prompt, maximum length and truncation
  policy, label/logit mapping, tuning method, optimizer, schedule, batch/
  accumulation, epochs/early stopping, precision, threshold selection, and
  checkpoint markers before result inspection.
- Same config is used for gold and LLM arms; only train target path/source may
  differ. One seed (scientific-plan value) per cell; no rerun-selection by outcome.
- Input-length audits cover all three datasets; unsafe truncation fails or uses
  a predeclared pair-aware policy equally across label sources.

## Architecture

Keep a common `StudentConfig`/train/evaluate interface with architecture-specific
adapters behind it. Resolve source repo to revision once, persist runtime and
input-length provenance, then hash the used config + revision into run manifests.

## Related Code Files

- Modify: `/mnt/d/study/cao-hoc/luan-van/code/models/student_config.py`
- Modify as eligible: `/mnt/d/study/cao-hoc/luan-van/code/models/classification_student.py`
- Modify as eligible: `/mnt/d/study/cao-hoc/luan-van/code/models/generative_reranker_student.py`
- Retain historical only unless selected: `/mnt/d/study/cao-hoc/luan-van/code/models/seq2seq_student.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/experiments/train_student.py`, `/mnt/d/study/cao-hoc/luan-van/code/experiments/evaluate_student.py`
- Reuse: `/mnt/d/study/cao-hoc/luan-van/code/utils/classification_threshold.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/runtime_provenance.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/checkpoint_manifest.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/peft_runtime.py`, `/mnt/d/study/cao-hoc/luan-van/code/utils/torch_runtime.py`
- Create/replace after freeze: `/mnt/d/study/cao-hoc/luan-van/code/configs/students/{three_frozen_student_ids}.json`
- Extend: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_student_backends.py`, `/mnt/d/study/cao-hoc/luan-van/code/tests/test_generative_reranker.py`, `/mnt/d/study/cao-hoc/luan-van/code/tests/test_classification_threshold.py`

## Tests Before

Write failing config tests for required immutable revision, explicit
cross-encoder type, pair-input contract, match-label mapping, truncation policy,
included flag/exact-three discovery, and forbidden label-source-dependent
hyperparameters. Add tiny mocked forward,
save/reload, threshold, length-boundary, precision, and checkpoint-manifest
tests for every selected architecture.

## Implementation Steps

1. Add `included` discovery and exact-three validation to `StudentConfig`.
2. Retire active references to unselected screening configs without deleting
   their historical results; create exactly three frozen configs.
3. Normalize pair splitting/tokenization and match probability output across
   selected classifier/reranker backends.
4. Run all-dataset input-length audits; freeze equal pair-aware handling for
   both label arms before training.
5. Smoke-train each model on a tiny synthetic balanced fixture; evaluate,
   persist threshold/checkpoint manifest, reload, and reproduce probabilities.
6. Document hardware/precision and resource-stop settings in each config;
   never choose a config from final test performance.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Three frozen configs | Load with pinned revision and eligibility metadata |
| Same model, gold vs LLM source | Identical hyperparameters/config hash |
| Over-length pair | Predeclared behavior; never silent asymmetric truncation |
| Swapped label mapping | Test fails before training |
| Save/reload tiny checkpoint | Same match probabilities/threshold |
| OOM/resource ceiling in smoke | Stop/revise scientific plan; no post-hoc shrink |
| 2 or 4 included student configs | Loader fails with actionable IDs |

## Success Criteria

- [ ] Exactly three eligible, pinned configs pass mocked and tiny smoke tests.
- [ ] All dataset lengths are audited and label-source arms share every model
  setting except target provenance.
- [ ] Checkpoints, thresholds, revisions, and runtime provenance reload safely.
- [ ] Loader discovers exactly three included configs and run-manifest inputs can
  hash each config/revision unambiguously.

## Risk Assessment

Architectural misclassification, label-map inversion, truncation, and GPU OOM
can invalidate comparison. Mitigate with Phase-1 proof, boundary tests, frozen
resource gates, small smoke runs, and probability round trips.

## Security/Data Integrity

Use public pinned model revisions, record licenses/hashes, avoid remote code
unless explicitly audited, and validate checkpoint manifests
before evaluation or reuse.

## Next Steps

Phase 5 combines the three validated configs with Phase-3 targets.
