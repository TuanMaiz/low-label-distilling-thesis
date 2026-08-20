---
phase: 3
title: "Build Full Label Targets"
status: pending
priority: P1
effort: "4-7d plus LLM-labeling runtime"
dependencies: [1, 2]
---

# Phase 3: Build Full Label Targets

## Overview

Build two complete training targets per dataset: benchmark gold labels and
strict answer-only LLM hard labels. Machine labeling is resumable, but target
publication requires exact one-to-one coverage of the frozen training split.

## Context Links

- Data phase: `./phase-02-generalize-dataset-pipeline.md`
- Machine-labeling code: `/mnt/d/study/cao-hoc/luan-van/code/supervision/generate_teacher_labels.py` (legacy identifier)
- Target builder: `/mnt/d/study/cao-hoc/luan-van/code/supervision/build_targets.py`

## Requirements

- Gold targets copy train labels only. LLM-labeling prompt payload contains serialized
  records and identifiers but no gold label, entity/cluster truth, or selection
  metadata. Validation/test labels are never machine-labeling inputs.
- One valid LLM-generated label for every frozen train pair; missing, duplicate,
  invalid, wrong-dataset, wrong-model/prompt, or unexpected pair IDs block the
  LLM target. Rejects remain separate and must be resolved/retried under the
  scientific plan's retry policy before publication.
- Cache keys include dataset/version/pair/labeler/prompt; resume only exact
  identity matches. Persist raw response, parsed label, validity, tokens, price,
  timestamps, request identity, and upstream hashes without secrets.
- Gold and LLM target rows differ only in target/provenance fields; pair order
  and input text are byte-identical within a dataset.
- Attempts are capped by the documented policy; exhaustion becomes a terminal
  reject that blocks publication—no silent drop, endless retry, or fallback.
  Paid generation requires manual checklist confirmation and an explicit
  `--confirm-paid-labeling` CLI flag after reviewing the dry-run estimate.

## Architecture

`verified train JSONL -> [gold copier | answer-only LLM-label cache + validator]
-> completeness join -> atomic targets/{gold,llm_hard}.jsonl + manifest`.
Evaluation JSONL remains separately gold-labeled and read-only.

## Related Code Files

- Modify: `/mnt/d/study/cao-hoc/luan-van/code/supervision/config.py`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/supervision/teacher_label_schema.py` (legacy identifier)
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/supervision/generate_teacher_labels.py` (legacy identifier)
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/supervision/validate_teacher_labels.py` (legacy identifier)
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/supervision/build_targets.py`
- Reuse: `/mnt/d/study/cao-hoc/luan-van/code/supervision/llm_providers.py`, `/mnt/d/study/cao-hoc/luan-van/code/supervision/prompts.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_full_label_targets.py`
- Extend: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_teacher_labels.py` (legacy identifier)

## Tests Before

Write failing tests for dataset/version path isolation, prompt redaction,
complete one-to-one joins, unexpected/missing/duplicate/invalid rows, cache
identity mismatch, resumability, rejects, atomic failure, pair/input parity
between label sources, and token/cost totals. Retain strict parser/direct-cache
tests in the legacy-named `test_teacher_labels.py`.

## Implementation Steps

1. Replace budget/selection path inference with explicit dataset/version/split/
   source identities while retaining readers for historical artifacts only.
2. Version the legacy-named label schema and define a composite cache identity.
3. Remove selection metadata from active request/target requirements; ensure
   gold/cluster fields cannot reach prompt construction.
4. Generate deterministic gold targets and manifests for all three train splits.
5. Run a normalized-input LLM-labeling dry run, compare cost with the ceiling,
   manually check the scientific-plan item, then require
   `--confirm-paid-labeling` for the full three-dataset call set.
6. Generate/resume labels, validate rejects and exact coverage after every run,
   and publish LLM targets only when coverage is 100%.
7. Record disagreement matrices between train gold and LLM labels for analysis,
   never for correcting/cherry-picking LLM-generated labels.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Complete exact LLM-label set | Publish deterministic LLM target |
| Missing/duplicate/extra/invalid label row | No target; actionable audit |
| Cache from another dataset/prompt/model | Refuse reuse |
| Pair has gold/cluster truth in source row | Prompt excludes truth fields |
| Interrupted generation | Resume exact valid IDs only |
| Attempts exhausted | Terminal reject; full target remains blocked |
| Paid flag omitted | Dry run allowed; paid generation refuses to start |
| Gold vs LLM target comparison | Same IDs/order/input; provenance/labels may differ |

## Success Criteria

- [ ] Six complete train targets exist with manifests and matching upstream
  hashes; every LLM target has 100% valid unique coverage.
- [ ] No validation/test truth or train gold truth reaches labeling prompts.
- [ ] LLM-labeling tokens/costs and gold–LLM disagreement are reproducibly reported.

## Risk Assessment

Full labeling cost, transient APIs, and invalid outputs are primary risks.
Mitigate with dry-run review, explicit paid CLI confirmation, identity-safe cache, strict parsing,
reject queues, ceilings, and fail-closed publication—not silent dropping.

## Security/Data Integrity

Load API keys from environment only; redact request headers/errors; hash all
inputs/outputs; use atomic append/publish behavior; never overwrite a cache
whose identity differs.

## Next Steps

Phase 5 may use targets only after all six target manifests validate.
