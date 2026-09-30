---
title: "Phase 1: Freeze the Foundational Scientific and Execution Contract"
status: todo
priority: P1
---

# Phase 1: Freeze the Foundational Scientific and Execution Contract

## Overview

Convert the accepted pipeline inventory into a reviewable experiment contract
before any new production teacher calls, OOF GPU training, expanded compact-model
training, or final-test access. This phase records decisions and prepares later
implementation phases; it does not execute the main experiment.

## Context Links

- [Existing full-label migration plan](../260820-1507-full-label-er-migration/plan.md)
- [Confident Learning roles journal](../journals/2026-09-16-confident-learning-roles-in-uncertainty-aware-label-refinement.md)
- [Post-hoc error-analysis journal](../journals/2026-09-28-post-hoc-error-analysis-for-llm-label-refinement.md)
- [WDC Sol-high vertical-slice contract](../260820-1507-full-label-er-migration/research/wdc-sol-high-vertical-slice-contract.md)
- [WDC-Qwen training vertical-slice contract](../260820-1507-full-label-er-migration/research/wdc-qwen-training-vertical-slice-contract.md)

## Established Decisions

- Use three benchmark datasets and three compact cross-encoder students.
- Use one teacher model, GPT-5.6 Sol-high.
- Obtain three teacher judgments per training pair to measure self-consistency.
- Use one evidence-check prompt only for pairs selected for relabeling.
- Retain prune and relabel as treatments; exclude reweighting.
- Keep the operational method gold-free and open hidden gold only for
  evaluation after rankings and targets are frozen.
- Preserve validation-based method development and one final test release.
- Treat negative and dataset-dependent results as valid findings.

## Current Starting Design — Partially Frozen

- Base LLM label: majority vote over three teacher passes.
- Consistency: `3/3` versus `2/3` for binary labels.
- Designated detector student: Qwen3-Reranker-0.6B generates the main
  three-fold OOF evidence across datasets; refined targets are then reused
  across all three downstream students. A second detector is optional only as
  a WDC robustness ablation if supervisor review requires it.
- Signal roles: Confident Learning estimates error count/direction,
  teacher-student disagreement ranks candidates within each direction, and LLM
  consistency stratifies or prioritizes candidates.
- Main treatment size: derived from a gold-free CL count; fixed percentages are
  sensitivity analysis only.
- Detector ablations and expensive controls: run on the designated detector
  student; only the final selected refinement method generalizes to all three
  students.

## Foundational Pipeline

### 1. Freeze the research framing

- Finalize the main question and five subquestions.
- Define a successful negative result and the condition for skipping refinement.
- Map every planned metric, arm, and analysis to a research question.
- Separate claims about deployable gold-free behavior from post-hoc hidden-gold
  findings.

### 2. Freeze datasets and students

- Confirm WDC and DBLP-ACM.
- Select and integrate Dataset 3.
- Confirm Qwen and select the other two compact cross-encoders.
- Select the designated detector student.
- Record observed training/validation/test sizes, class balance, licenses, and
  expected workload only after inspecting each dataset.

Current reusable work:

- WDC: integrated; pass-1 Sol-high labels and a one-pass WDC-Qwen pilot exist.
- DBLP-ACM: adapter, preparation, fake-label path, and Qwen readiness exist;
  production labels do not.
- Dataset 3 and students 2-3 remain open.

### 3. Freeze the three-pass teacher protocol

- Freeze teacher identity, OpenRouter routing, prompt, response schema,
  reasoning/sampling settings, retry behavior, and per-dataset cost ceilings.
- Preserve each response, call metadata, and cost separately.
- Freeze majority-label construction and `3/3`/`2/3` consistency semantics.
- Freeze invalid-response and interrupted-run recovery behavior.
- Freeze the evidence-check prompt and relabel acceptance rule.
- Define the pilot criterion that would replace two identical repeats with two
  equivalent prompts instead of adding more calls.

Output family:

```text
teacher_passes/
majority_labels/  # llm_unrefined
```

If majority labels become the main source, the existing one-pass WDC-Qwen result
is a pilot rather than a final cell and must not be silently mixed into the new
matrix.

### 4. Freeze the three-fold OOF protocol

For the designated detector student:

```text
fold 1: train on folds 2+3 -> predict fold 1
fold 2: train on folds 1+3 -> predict fold 2
fold 3: train on folds 1+2 -> predict fold 3
```

- Freeze deterministic, class-aware fold construction.
- Use dataset-defined `pair_id` as the OOF example identity and reserve a
  deterministic class-aware 20% of each fold's non-held-out pool for inner
  validation/checkpoint selection.
- Reuse the accepted model training configuration unless the contract records a
  necessary change.
- Save canonical row identity, fold identity, majority given-label ID, and OOF
  probabilities in frozen class order `0 = non_match`, `1 = match`. Later CL
  code consumes the given-label vector and the two-column probability matrix;
  margin, entropy, disagreement, and issue rankings are derived later rather
  than replacing the raw probabilities.
- Require every training pair to be predicted exactly once by a model that did
  not train on it.
- Define checkpoint recovery and artifact-completeness checks.

### 5. Freeze detector definitions

- **LLM consistency:** categorical `3/3` or `2/3`; not assumed to be a full
  continuous ranking.
- **Confident Learning:** define count estimate, error direction, candidate
  evidence, and any retained label-quality score.
- **Teacher-student disagreement:** define the continuous score for both label
  directions from OOF probabilities.
- Freeze the individual-signal baselines and the small set of integration rules
  evaluated in the WDC-Qwen pilot.
- Keep neighborhood/outlier detection outside the main matrix unless later
  evidence justifies one bounded baseline.

### 6. Freeze the selection rule

- Determine the treatment count without gold.
- Prefer direction-specific counts for `teacher match -> latent non-match` and
  the reverse.
- Rank within direction using the frozen detector evidence.
- Predeclare any percentage-based sensitivity points.
- Do not select the final threshold or fusion rule from test results or hidden
  training-gold performance.

### 7. Freeze treatments

- **Prune:** remove selected pairs; preserve all remaining pair identities and
  majority labels.
- **Relabel:** call the evidence-check prompt only for selected pairs; apply a
  predeclared rule for accepting or rejecting the new label.
- Publish separate `llm_unrefined`, `llm_pruned`, and `llm_relabelled` targets.
- Require an offline fake path before any paid relabeling.

### 8. Freeze controls

Gold-free operational arms:

- Unrefined majority labels.
- Detector-based prune and relabel.
- Fixed random selection with the same intervention size.

Evaluation-only controls:

- Oracle prune using hidden training gold.
- Full human-label training as the perfect-relabel upper reference.
- Direct-LLM matching where retained for deployment-cost comparison.

Decide whether random relabel is worth its API cost and which random/oracle
controls run only on the designated detector student. Oracle artifacts must be
marked non-deployable and must not influence method selection.

### 9. Freeze the tiered experiment strategy

#### Tier 1 — WDC-Qwen method pilot

- Add two remaining teacher passes.
- Validate majority/consistency artifacts and three-fold OOF coverage.
- Compare individual signals and the bounded integration candidates.
- Freeze treatment count and relabel acceptance rules using validation only.
- Keep test locked.

Exit: one frozen gold-free detection/refinement method.

#### Tier 2 — Detection study across three datasets

- Run three OOF folds per dataset with the designated detector student.
- Freeze rankings and treatment targets before hidden-gold evaluation.
- Evaluate signal complementarity, random/oracle controls, and whether any
  dataset should skip refinement.

Expected principal OOF workload: `3 datasets x 3 folds = 9 runs`.

#### Tier 3 — Downstream generalization

Run the principal arms:

```text
3 datasets x 3 students x
{human, llm_unrefined, llm_pruned, llm_relabelled}
```

Do not expand every detector ablation or costly control across all students
without a later explicit scope decision.

### 10. Freeze evaluation metrics

Detection:

- Average Precision/AUPRC.
- Precision@k and Recall@k.
- Error-count estimation error.
- Results by error direction and `3/3` versus `2/3` consistency.
- Detector overlap and complementarity.

Downstream:

- Match F1 primary.
- Precision, recall, macro F1, and accuracy supporting.
- Training/inference time and throughput.

Cost:

- Three-pass teacher labeling.
- Evidence-check relabeling.
- OOF GPU training.
- Final student training and compact-model inference.
- Direct-LLM inference and break-even analysis.

### 11. Freeze post-hoc error analysis

Only after operational rankings, pair selections, targets, and student outputs
are frozen:

1. Analyze teacher false matches and false non-matches.
2. Analyze detector true positives, false positives, and false negatives.
3. Analyze which errors materially affect downstream students.
4. Stratify sparse/conflicting attributes, near duplicates, correct hard cases,
   and any observed dataset-specific corner cases.
5. Test whether teacher errors have structure; do not assume that they do.

Inspect all errors when tractable, such as the current 79 WDC disagreements;
otherwise use a fixed, predeclared manual sample.

### 12. Freeze artifacts and engineering workstreams

Expected artifact families:

```text
teacher_passes/
majority_labels/
oof_predictions/
detector_scores/
rankings/
targets/
  llm_unrefined/
  llm_pruned/
  llm_relabelled/
  random_controls/
  oracle_controls/
training_runs/
evaluation/
error_analysis/
cost_analysis/
```

Later implementation phases are expected to cover:

1. Multi-pass teacher runner and majority builder.
2. Three-fold OOF runner and merger.
3. CL, disagreement, and consistency outputs.
4. Bounded integration/ranking and target builders.
5. Evidence-check relabeling and analytical controls.
6. Tiered student training/evaluation.
7. Detection/downstream/error/cost aggregation.
8. Verification, packaging, and thesis-table handoff.

Do not restore unnecessary provenance machinery. Keep exact pair coverage,
identity alignment, raw predictions, gold-path isolation, dry runs, cost logs,
and recovery checks that materially protect the scientific result. Model weights
must be packaged separately and not committed to Git.

## Test and Validation Strategy

Before expensive work, fixture tests must verify:

- Three teacher passes align to exactly the same pair IDs.
- Majority and consistency calculations are deterministic.
- No gold field can reach teacher, OOF detector, selection, or treatment paths.
- OOF outputs cover every training pair exactly once without overlap.
- CL/disagreement/integration outputs are deterministic on fixtures.
- Prune/relabel targets preserve schema and expected identity sets.
- Random and oracle controls are visibly separated.
- Validation/test locks reject unauthorized access.
- Interrupted labeling/training can resume without duplicating paid calls or
  silently accepting partial predictions.
- WDC and DBLP regression suites remain unchanged where behavior is reused.

## Open Decisions

1. Exact Dataset 3.
2. Exact students 2 and 3.
3. Whether majority vote becomes the official base LLM label beyond the WDC
   pilot.
4. Whether exact-repeat Sol-high calls produce enough variation.
5. Exact CL count/candidate outputs and disagreement formula.
6. Integration role of the `2/3` flag: stratification, tie-break, or priority.
7. Treatment-size rule and sensitivity percentages.
8. Evidence-check relabel acceptance rule.
9. Whether random relabel is worth its paid-call cost.
10. Which controls remain designated-student-only beyond the approved primary
    Qwen detector.
11. Maximum manual error-analysis sample.
12. API, GPU, storage, and calendar ceilings plus a reduced-scope fallback.

## Risks and Mitigations

| Risk | Consequence | Mitigation |
|------|-------------|------------|
| Exact teacher repeats are deterministic | No useful LLM-consistency signal | Pilot first; replace, not add, equivalent prompts if the frozen criterion fires |
| OOF/student and CL evidence are dependent | Apparent multi-signal gain may be double-counting | Report individual signals, overlap, and bounded integration ablations |
| Class imbalance marks rare matches as noise | Prune removes valuable positives | Direction-specific counts/metrics and post-hoc false-alarm analysis |
| Main matrix expands combinatorially | Thesis becomes infeasible | One detector student, tiered execution, anchor-only controls/ablations |
| Real LLM errors form learnable patterns | Student confidently agrees with wrong teacher labels | Preserve teacher consistency and complementary detectors; report missed-error categories |
| Gold leaks into detector design | Invalid gold-free claim | Separate modules/artifacts and freeze rankings before evaluation join |
| Refinement gains arise from regularization | Incorrect causal claim | Same-size random controls and oracle headroom |
| API/GPU cost exceeds value | Method is operationally unattractive | Track marginal cost and allow an explicit skip-refinement result |

## Todo

- [ ] Approve the research questions and scientific boundary.
- [ ] Resolve the thirteen open decisions.
- [ ] Estimate final run counts, API spend, GPU hours, storage, and calendar time.
- [ ] Approve the tiered scope and reduced-scope fallback.
- [ ] Freeze the human-readable experiment contract in Git.
- [ ] Add detailed implementation phases through `ak plan add-phase`.

## Success Criteria

Phase 1 is complete only when:

- Three datasets and three students are named and loadable.
- One designated detector student is selected or the extra workload of an
  alternative is explicitly approved.
- Teacher majority/consistency, OOF, detector, selection, treatment, and
  control definitions are frozen.
- Gold-free boundaries are represented in code/API contracts and fixture tests.
- Detection, downstream, cost, and error-analysis outputs are predeclared.
- Workload and budget fit the thesis schedule.
- Test-release conditions are documented.
- No new production paid call, OOF run, expanded main run, or final test begins
  before the relevant gate passes.

## Next Steps

Review this foundation decision by decision. After acceptance, use the plan CLI
to add small executable phases with owned files, commands, tests, budgets, and
per-phase authorization gates. Implementation then proceeds through `ak:cook`
one reviewed phase at a time.
