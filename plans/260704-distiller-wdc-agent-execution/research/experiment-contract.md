---
title: "Experiment Contract: Cost-Aware Distillation of LLM-Generated Labels for Entity Matching"
status: active
created: 2026-07-04
plan: "/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/plan.md"
phase: 1
---

# Experiment Contract: Cost-Aware Distillation of LLM-Generated Labels for Entity Matching

## Working Title

Cost-Aware Distillation of LLM-Generated Labels for Entity Matching.

## Research Question

Can compact Entity Matching students distilled from LLM-generated teacher labels approach gold-label supervised students while being cheaper at inference time than using the LLM directly as the matcher?

## Safe Thesis Claim

This thesis evaluates label-level knowledge distillation for WDC Products: an LLM teacher generates training labels, then a compact student learns from those labels and performs final inference without teacher calls. It compares this against two references: a gold-label supervised compact student for quality, and direct LLM matching for inference cost. It does not claim that LLM-label distillation for Entity Resolution is new. The safe contribution is a controlled WDC-focused study of label efficiency, compact student distillation, teacher-label cost, direct LLM inference cost, and teacher/student failure behavior.

## Novelty Boundary

DistillER already studies knowledge distillation for Entity Resolution with LLM teachers, data selection, teacher labels, student models, and explanation variants. This thesis should cite it as closest related work and position itself as:

- WDC Products pairwise stress test.
- compact student comparison under low-label budgets.
- explicit cost accounting for teacher-generated labels and direct LLM inference.
- failure analysis of LLM teacher labels and student predictions.
- master-thesis-scale reproduction/adaptation, not a new general framework.

## Experiment Arms

There are three experiment arms. They answer different questions and should not be collapsed into one baseline.

| Arm | Name | What It Does | Main Question | Cost Role |
|---|---|---|---|---|
| A | `gold_label_student` | train compact student on gold labels | how good is trusted supervised training? | performance standard; not the cost-saving method |
| B | `direct_llm_matcher` | ask the LLM to classify evaluation pairs directly | how good/expensive is using the LLM at inference time? | inference-cost baseline |
| C | `llm_label_distilled_student` | ask LLM to label training pairs, then train compact student | can one-time teacher labeling produce cheap student inference? | proposed method |

This is not cherry-picking if all three arms, splits, budgets, prompts, and cost fields are fixed before running.

## Dataset Contract

Dataset: `wdc_products`.

Source artifact:

`/mnt/d/Study/Cao-hoc/luan-van/code/data/raw/wdc_products/80pair.zip`

Existing prepared cache:

`/mnt/d/Study/Cao-hoc/luan-van/code/data/cache/wdc_products/`

Configuration:

| Field | Value |
|---|---|
| corner cases | 80 |
| train size | small |
| test unseen | 100 |
| seed | 42 |

Existing splits:

| Split | Pairs | Matches | Non-matches |
|---|---:|---:|---:|
| train | 2500 | 500 | 2000 |
| validation | 2500 | 500 | 2000 |
| test | 4500 | 500 | 4000 |

Existing low-label training budgets:

| Budget | Pairs | Matches | Non-matches | File |
|---:|---:|---:|---:|---|
| 16 | 16 | 8 | 8 | `data/cache/wdc_products/low_label/train_16.jsonl` |
| 32 | 32 | 16 | 16 | `data/cache/wdc_products/low_label/train_32.jsonl` |
| 64 | 64 | 32 | 32 | `data/cache/wdc_products/low_label/train_64.jsonl` |
| 128 | 128 | 64 | 64 | `data/cache/wdc_products/low_label/train_128.jsonl` |
| full | 2500 | 500 | 2000 | `data/cache/wdc_products/low_label/train_full.jsonl` |

Important correction: budget `256` does not currently exist in the cache. If the pilot uses `256`, Phase 3 or an earlier setup task must update `DEFAULT_LOW_LABEL_BUDGETS` or run the sampler with `256`.

## Student Supervision Variants

These variants apply only to compact student training. `direct_llm_matcher` is not a supervision variant because it does not train a student.

| Variant | Training Label Source | Experiment Arm | Required For Pilot | Purpose |
|---|---|---|---:|---|
| `gold_label` | dataset gold labels | A | yes | supervised compact-student performance standard |
| `llm_label` | validated answer-only teacher labels | C | yes | main distillation variant |
| `mixed_gold_llm` | small gold seed plus LLM labels | C | optional | practical fallback if pure LLM labels are noisy |
| `old_structured_rationale` | existing structured-rationale targets | historical only | no | negative-history ablation only |

Validation and test evaluation must always use gold labels.

## Direct LLM Matcher Contract

Direct LLM matching uses the same answer-only matching prompt family, but it is applied to evaluation pairs rather than to training-label creation.

Rules:

1. Use the same teacher model family as the distillation teacher unless a change is declared before running.
2. Evaluate on a fixed validation/test set or fixed sample declared before running.
3. Store prompt version, model slug, input tokens, output tokens, estimated cost, parsed label, gold label, and validity.
4. Report both quality and cost. This arm is the cost baseline because cost grows with every prediction.

Preferred first run:

- full validation split if budget allows: `2500` pairs.
- otherwise a fixed validation sample with seed `42`, declared before results are inspected.

## Teacher Labeling Contract

Provider: OpenRouter, reusing the existing provider pattern.

Default model convention from the current rationale pipeline:

`openrouter:openai/gpt-4o-mini`

Rules:

1. Temperature must be `0.0`.
2. Prompt version must be recorded.
3. Actual model slug must be recorded in every cache row.
4. Teacher output for both direct matching and training-label generation must be answer-only, not rationale text.
5. Valid labels are exactly:
   - `match`
   - `non_match`
6. Any extra text, uncertainty, malformed JSON, or missing label is invalid unless a strict parser can map it safely.

## Student Contract

First student: existing `google/flan-t5-base` pipeline, because this codebase already has train/evaluate entry points and previous Phase 3 results.

Optional second student after pilot:

- RoBERTa-style classifier if thesis needs a more standard compact ER baseline.
- MiniLM only if speed/cost comparison becomes important.

Do not add a second student before the `128` pilot answers whether LLM labels have signal.

## Pilot Matrix

Minimum pilot:

| Arm | Budget / Eval Set | Variant | Required |
|---|---|---|---:|
| A | train 128, validation eval | `gold_label` | yes |
| B | fixed validation eval | `direct_llm_matcher` | yes |
| C | train 128, validation eval | `llm_label` | yes |
| C | train 128, validation eval | `mixed_gold_llm` | optional |

Preferred pilot if sampler is extended:

| Arm | Budget / Eval Set | Variant | Required |
|---|---|---|---:|
| A | train 256, validation eval | `gold_label` | yes |
| B | fixed validation eval | `direct_llm_matcher` | yes |
| C | train 256, validation eval | `llm_label` | yes |
| C | train 256, validation eval | `mixed_gold_llm` | optional |

Full study after pilot:

`16 / 32 / 64 / 128 / 256`, plus `full` reference if compute allows.

## Metrics

Primary metric:

- match-class F1 (`same_f1`), because matching quality is the main ER concern.

Always report:

- match precision.
- match recall.
- macro F1.
- accuracy.
- invalid-output rate.
- confusion matrix counts: TP, FP, TN, FN.

Cost metrics:

- input tokens.
- output tokens.
- estimated teacher cost per pair.
- total teacher cost per budget.
- direct LLM matching cost per evaluated pair.
- projected direct LLM matching cost for validation/test scale.
- valid label count.
- invalid label count.
- cost per valid teacher label.
- student inference cost if measurable or estimated.
- break-even query count: number of future predictions where distillation becomes cheaper than direct LLM inference.

## Cost Comparison Logic

Gold labels are the trusted quality standard, not the main cost baseline. The cost comparison should be:

```text
direct_llm_matcher cost = LLM cost paid for every prediction

llm_label_distilled_student cost =
  one-time LLM teacher-labeling cost
  + compact student training cost
  + cheap compact student inference cost
```

The thesis should report quality and cost together:

| Question | Compare |
|---|---|
| quality gap from trusted labels | `gold_label` student vs `llm_label` distilled student |
| cost advantage over direct LLM use | `direct_llm_matcher` vs `llm_label` distilled student |
| practical stabilization | `llm_label` vs `mixed_gold_llm` |

## Current Baseline Context

Existing Phase 3 validation result at budget `128`:

| Variant | Train Rows | Match Precision | Match Recall | Match F1 | Macro F1 | Accuracy | Invalid Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `label_only` | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 | 0.0000 |
| `structured_rationale` | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 | 0.0004 |

Interpretation: structured rationales increased recall but badly hurt precision and overall F1. This justifies pivoting away from rationale supervision and testing answer-only teacher labels instead.

## Artifact Paths

Teacher labels:

`data/cache/wdc_products/teacher_labels/train_{budget}.openrouter.answer_only_v1.labels.jsonl`

Teacher rejects:

`data/cache/wdc_products/teacher_labels/train_{budget}.openrouter.answer_only_v1.rejects.jsonl`

Direct LLM predictions:

`outputs/distiller_wdc/direct_llm/{split}.openrouter.answer_only_v1.predictions.jsonl`

Direct LLM cost summary:

`outputs/distiller_wdc/direct_llm/{split}.openrouter.answer_only_v1.cost.json`

Targets:

`data/cache/wdc_products/targets/train_{budget}.{variant}.targets.jsonl`

Outputs:

`outputs/distiller_wdc/flan-t5-base/train_{budget}/{variant}/`

Aggregates:

`outputs/distiller_wdc/summary/`

Figures:

`outputs/distiller_wdc/figures/`

Failure analysis:

`outputs/distiller_wdc/analysis/`

## Stop / Continue Gates

After teacher labeling:

| Decision | Condition |
|---|---|
| Continue | valid labels generated, invalid rate <= 10 percent |
| Revise | parser/prompt problems are fixable |
| Stop | teacher cannot produce stable binary labels |

After direct LLM matching:

| Decision | Condition |
|---|---|
| Continue | direct LLM predictions and costs are logged on the fixed evaluation set |
| Revise | direct LLM output parsing is unstable but prompt/parser fixes are clear |
| Stop | direct LLM matching cannot produce valid binary predictions at acceptable invalid rate |

After pilot training:

| Decision | Condition |
|---|---|
| Continue | `llm_label` is close enough to `gold_label` for quality and cheaper than projected direct LLM inference at scale, or `mixed_gold_llm` clearly helps |
| Revise | pure LLM labels are weak but error pattern suggests better prompt, threshold, or mixed supervision |
| Stop | LLM-label student is worse than the old label-only baseline with no useful diagnostic story |

After full budget study:

| Decision | Condition |
|---|---|
| Continue to thesis writing | clear label-efficiency or cost-efficiency pattern exists |
| Add optional student | FLAN-T5 behavior is too seq2seq-specific |
| Add optional dataset | WDC-only result is too narrow for advisor expectations |

## Out Of Scope For First Execution

- Full DistillER reproduction.
- Multi-teacher comparison.
- Routing or active-learning policy.
- Explanation/rationale generation as the main claim.
- More than one dataset before WDC pilot.
- More than one student before WDC pilot.

## Immediate Next Phase

Phase 2 should update project guidance so future agents stop treating structured-rationale distillation as the active thesis and follow this label-level LLM-to-student distillation contract.
