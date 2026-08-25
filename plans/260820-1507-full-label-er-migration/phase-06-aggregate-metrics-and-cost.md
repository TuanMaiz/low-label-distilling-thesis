---
phase: 6
title: "Aggregate Metrics and Cost"
status: pending
priority: P1
effort: "4-6d"
dependencies: [5]
---

# Phase 6: Aggregate Metrics and Cost

## Overview

Implement a new aggregator that validates all cell provenance, reports the
predeclared accuracy metrics, and compares labeling/training/inference economics
without conflating the direct matcher with a training source.

## Context Links

- Runner: `./phase-05-refactor-experiment-runner.md`
- Deleted/retired aggregation behavior, if needed: Git history only.
- Cost helpers: `/mnt/d/study/cao-hoc/luan-van/code/utils/cost_accounting.py`

## Requirements

- Include only artifacts matching one experiment ID, plan/config/input hashes,
  and complete run manifests;
  default requires all 18 model cells and 3 direct baselines. A clearly marked
  diagnostic `--allow-partial` output may not be used as thesis final results.
- Primary: match precision/recall/F1 with match F1 emphasized. Supporting:
  macro F1, accuracy, confusion counts, invalid rate, and gold-minus-LLM deltas
  per same dataset/model. Report per-dataset first; any cross-dataset summary
  must be macro, never pooled rows that let a large dataset dominate.
- Costs: observed LLM-labeling input/output tokens and cost, synchronized
  compact-model training/inference seconds, throughput, configured GPU-hour
  sensitivity scenarios, per-pair cost, direct per-pair cost, and break-even
  queries. Do not count gold labels as free: label their acquisition cost as
  unavailable/benchmark-provided unless a declared human-cost scenario exists.
- Freeze break-even as
  `N* = (C_label + C_train) / (c_direct - c_model)` only for a positive
  denominator; otherwise report no finite break-even. Charge full labeling cost
  in each per-model deployment comparison; optionally add a clearly labeled
  shared-label three-model portfolio view that charges labeling once.
- One run per cell means no repeated-run variance/significance claim. Report descriptive
  results and limitations honestly.

## Architecture

`artifact discovery -> manifest/hash/schema validation -> normalized result rows
-> paired gold-vs-LLM deltas -> cost scenarios/break-even -> JSON + CSV + audit
report`. The aggregator performs no training, LLM calls, or artifact repair.

## Related Code Files

- Create: `/mnt/d/study/cao-hoc/luan-van/code/experiments/aggregate_full_label_results.py`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/configs/full_label_cost_assumptions.json`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/tests/test_full_label_aggregation.py`
- Reuse/extend: `/mnt/d/study/cao-hoc/luan-van/code/utils/metrics.py`
- Reuse/extend: `/mnt/d/study/cao-hoc/luan-van/code/utils/cost_accounting.py`
- Reuse: `/mnt/d/study/cao-hoc/luan-van/code/utils/artifact_contract.py`
- Migrate generic assertions from `/mnt/d/study/cao-hoc/luan-van/code/tests/test_phase05_cost_accounting.py`; remove obsolete Phase-05-only cases/name.

## Tests Before

Write golden fixture tests for all 18 cells/3 baselines, paired metric deltas,
class-imbalanced match F1, macro-over-datasets, missing/stale/duplicate cells,
invalid numeric/negative/nonfinite costs, synchronized timing, direct-vs-model
break-even, non-cheaper null break-even, assumption hashes, and partial watermark.

## Implementation Steps

1. Define normalized result/audit schemas keyed by experiment ID, plan hash,
   dataset, model, label source, split, and exact artifact hashes.
2. Discover expected paths from included configs and run manifests, never by
   parsing scientific Markdown or relying on loose directory glob semantics;
   validate completeness and provenance before reading metrics.
3. Compute/recompute metric rows from predictions as a cross-check against
   stored metrics; fail on differences beyond fixed tolerance.
4. Join gold and LLM cells by dataset/model; compute directional paired deltas and
   per-dataset rankings without selecting a winner post hoc.
5. Add labeling, training, inference, direct, sensitivity, and break-even tables
   using the frozen equation; separate per-model full-labeling and optional
   shared-portfolio views; retain assumption hashes and observed/assumed flags.
6. Emit deterministic JSON/CSV plus a Markdown audit showing omissions,
   limitations, experiment ID, Git commit, and plan hash. Add partial watermark mode.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Complete valid fixture | 18 model rows, 9 paired deltas, 3 direct rows |
| Missing/stale/duplicate cell | Final aggregation fails |
| Stored vs recomputed metric mismatch | Fail with cell/path |
| Highly imbalanced labels | Match F1 uses positive class correctly |
| Model per-pair cost >= direct | Break-even is null |
| Label cache reused by 3 models | Per-model charges full label cost; portfolio once |
| Missing gold acquisition cost | `unavailable`, never zero |
| Partial mode | Diagnostic output visibly non-final |

## Success Criteria

- [ ] Complete aggregate contains exact expected cardinalities and recomputed
  metrics with immutable provenance.
- [ ] Accuracy and economic comparisons follow the committed plan and distinguish
  observations from assumptions.
- [ ] Golden, failure, and full regression suites pass.

## Risk Assessment

Incorrect joins, class orientation, or cost assumptions can reverse conclusions.
Mitigate with keyed joins, recomputation from predictions, golden fixtures,
assumption hashes, directional deltas, and no silent missing-data coercion.

## Security/Data Integrity

Treat artifacts as untrusted input: strict JSON/schema/numeric validation,
bounded reads, no code execution, and no secrets/raw provider headers in output.

## Next Steps

Phase 7 performs clean-clone verification and hands the stable result schema to
the thesis writing plan.
