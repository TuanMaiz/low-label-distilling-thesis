---
title: "Experiment Contract: Cost-Aware Active LLM Labeling for Entity Matching"
status: active
created: 2026-07-04
plan: "/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-agent-execution/plan.md"
phase: 1
---

# Experiment Contract: Cost-Aware Active LLM Labeling for Entity Matching

## Working Title

Cost-Aware Active LLM Labeling for Low-Budget Entity Matching.

## Research Question

Under low-label budgets on WDC Products, can active selection of LLM-labeled
training pairs produce compact Entity Matching students that outperform random
LLM-label distillation at the same labeling cost, while becoming cheaper than
using the LLM directly as the matcher at repeated inference time?

## Safe Thesis Claim

This thesis evaluates cost-aware active LLM labeling for WDC Products: a
selection strategy chooses a small set of candidate training pairs, an LLM
teacher labels them once, and a compact student learns from those labels for
cheap final inference without teacher calls. It compares active selection
against random LLM-label distillation, a gold-label supervised compact student
for quality context, and direct LLM matching for repeated inference cost. It
does not claim that LLM-label distillation, active learning, or data selection
for Entity Resolution is new. The safe contribution is a controlled WDC-focused
study of which pairs are worth spending LLM calls on under low budgets, how
selection affects teacher noise, when the resulting student is cost-effective,
and which WDC failure slices benefit or fail.

## Analytical Lens

The core method is close to prior LLM-labeling and DistillER-style work. The
thesis differentiates itself by the questions it asks:

| Lens | Thesis Question | Main Evidence |
|---|---|---|
| Cost | When does one-time selected teacher labeling plus compact inference become cheaper than repeated direct LLM inference? | token/cost logs, per-budget cost, break-even query count |
| Low-label budgets | How do random and active LLM-label distillation behave at small supervision budgets? | `16 / 32 / 64 / 128`, optional `256 / full` curves |
| Data selection | Which pairs should receive scarce LLM labels: random, uncertain, diverse, hard-negative, or hybrid selections? | same-budget strategy comparison, selected-pair composition, label-efficiency curves |
| WDC difficulty slices | Which product-pair conditions cause teacher, direct LLM, or student errors? | hard negatives, missing fields, brand/title/model conflicts, long descriptions, price/currency mismatch |
| Teacher noise | Do active strategies select informative examples or mainly ambiguous examples where the LLM teacher is wrong? | teacher-vs-gold disagreement joined with selected-pair strategy and student predictions |
| External validity | Does the WDC pattern survive on another dataset? | optional Abt-Buy, Walmart-Amazon, or DBLP-style replication after WDC pilot |

## Novelty Boundary

DistillER and Steiner/Bizer-style LLM-labeling work already study knowledge
distillation or machine labeling for Entity Resolution with LLM teachers, data
selection, teacher labels, student models, and cost/performance tradeoffs.
Classic and deep active-learning work for ER also studies label scarcity.
Direct LLM matching on WDC Products exists in prior LLM-for-EM work. This
thesis should cite those papers and position itself as:

- WDC Products pairwise stress test under low-label budgets.
- active selection of scarce LLM teacher-label calls, compared with random
  LLM-label distillation under the same budgets.
- compact student comparison under low-label budgets.
- explicit break-even cost accounting for teacher-generated labels and direct
  LLM inference.
- failure-slice analysis of selected LLM teacher labels, direct LLM predictions,
  and student predictions.
- optional external-validity check on one additional dataset after WDC.
- master-thesis-scale reproduction/adaptation, not a new general framework.

## Experiment Arms

There are four experiment arms. They answer different questions and should not be collapsed into one baseline.

| Arm | Name | What It Does | Main Question | Cost Role |
|---|---|---|---|---|
| A | `gold_random_student` | train compact student on randomly sampled gold labels | how good is trusted supervised training under the same budget? | quality context; not the cost-saving method |
| B | `direct_llm_matcher` | ask the LLM to classify evaluation pairs directly | how good/expensive is using the LLM at inference time? | repeated inference-cost baseline |
| C | `llm_random_student` | randomly sample training pairs, ask the LLM to label them, then train compact student | what does plain random LLM-label distillation buy? | random distillation control |
| D | `llm_active_student` | actively select training pairs, ask the LLM to label them, then train compact student | does choosing better pairs improve quality per LLM dollar? | proposed active labeling method |

This is not cherry-picking if all arms, candidate pools, selection strategies,
splits, budgets, prompts, and cost fields are fixed before running.

## Dataset Contract

Primary dataset: `wdc_products`.

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

Active-selection candidate pool:

- Use only the WDC training split as the candidate pool.
- Do not use validation or test gold labels to select teacher-label training
  pairs.
- Selection strategies must produce a fixed manifest before teacher labels are
  inspected.
- Each selected manifest must record `selection_strategy`, `budget`, `seed`,
  `rank`, any selection score used, and active bucket metadata where applicable.

Optional later datasets:

- Add only after the WDC pilot and analysis pipeline are working.
- Use them as external-validity checks, not as the main thesis claim.
- Prefer one simple additional EM benchmark first, such as Abt-Buy or
  Walmart-Amazon, before adding more domains.

## Student Supervision Variants

These variants apply only to compact student training. `direct_llm_matcher` is not a supervision variant because it does not train a student.

| Variant | Training Label Source | Selection | Experiment Arm | Required For Pilot | Purpose |
|---|---|---|---|---:|---|
| `gold_random` | dataset gold labels | random balanced low-label sample | A | yes | supervised compact-student performance context |
| `llm_random` | validated answer-only teacher labels | random balanced low-label sample | C | yes | random distillation control |
| `llm_active_uncertainty` | validated answer-only teacher labels | uncertain candidate pairs | D | optional for first pilot, yes for full study if feasible | decision-boundary active selection |
| `llm_active_diversity` | validated answer-only teacher labels | diverse representative candidate pairs | D | optional | coverage-oriented active selection |
| `llm_active_bucketed_v1` | validated answer-only teacher labels | equal default coverage of four WDC-motivated candidate buckets | D | recommended first active variant | practical active-selection candidate |
| `llm_active_hybrid` | validated answer-only teacher labels | blended uncertainty/hard-negative/diversity score | D | optional comparison | superseded pilot selector |
| `mixed_gold_llm_active` | small gold seed plus active LLM labels | active selection plus gold seed | D | optional | fallback if pure LLM labels are noisy |
| `old_structured_rationale` | recorded Phase 03 result | historical only | historical only | no | negative-history context only |

Validation and test evaluation must always use gold labels.

## Selection Strategy Contract

The first active-selection study should stay simple enough for a master's thesis:

| Strategy | Candidate Signal | Why Include It | First-Run Status |
|---|---|---|---|
| `random` | existing balanced low-label sampler | control for current plan | required |
| `uncertainty` | compact seed model confidence near decision boundary, or disagreement between cheap heuristics | tests classic active-learning intuition | optional if seed scores are available |
| `diversity` | embedding or lexical coverage across product pairs | avoids spending all labels on near-duplicates | optional |
| `hard_negative` | high title/token overlap but gold/candidate expectation of non-match, using training split only | targets WDC product false positives | optional |
| `llm_active_bucketed_v1` | 25 percent each from easy-match, hard-match, easy-non-match, and hard-negative candidate buckets | defendable first thesis variant if only one active strategy is affordable | recommended |
| `hybrid` | combine diversity with uncertainty or hard-negative candidates | retained as older pilot comparison | optional |

Minimum accepted comparison:

```text
same budget, same teacher, same student, same validation split:
  llm_random
  vs llm_active_bucketed_v1
```

Do not inspect validation/test outcomes before freezing the selected pair
manifest for a strategy.

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

Teacher labels for active strategies should be generated only after the
selection manifest is written. Cache rows should preserve the selection strategy
or be joinable to the manifest by `pair_id`.

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
| A | train 128, validation eval | `gold_random` | yes |
| B | fixed validation eval | `direct_llm_matcher` | yes |
| C | train 128, validation eval | `llm_random` | yes |
| D | train 128, validation eval | `llm_active_bucketed_v1` | yes, if active-selection scores are available |
| D | train 128, validation eval | `mixed_gold_llm_active` | optional |

Preferred pilot if sampler is extended:

| Arm | Budget / Eval Set | Variant | Required |
|---|---|---|---:|
| A | train 256, validation eval | `gold_random` | yes |
| B | fixed validation eval | `direct_llm_matcher` | yes |
| C | train 256, validation eval | `llm_random` | yes |
| D | train 256, validation eval | `llm_active_bucketed_v1` | yes if 128 pilot is promising |
| D | train 256, validation eval | `mixed_gold_llm_active` | optional |

Full study after pilot:

`16 / 32 / 64 / 128 / 256`, plus `full` reference if compute allows. Run
`random` and the best active strategy across all budgets first; add other active
strategies only if the core curve is stable.

Optional external-validity run after WDC:

| Dataset Role | Candidate | Purpose |
|---|---|---|
| first replication | Abt-Buy or Walmart-Amazon | check whether WDC cost/performance pattern holds on a common product benchmark |
| later contrast | DBLP-ACM or DBLP-Scholar | check whether product-specific findings transfer to bibliographic matching |

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
- selected-pair composition by strategy.
- teacher-vs-gold disagreement rate by strategy.
- student gain or loss versus `llm_random` at the same budget.
- student inference cost if measurable or estimated.
- break-even query count: number of future predictions where distillation becomes cheaper than direct LLM inference.

Failure-slice metrics:

- teacher-vs-gold disagreement rate by slice.
- student-vs-gold error rate by slice.
- teacher-wrong/student-wrong, teacher-wrong/student-correct, and
  teacher-correct/student-wrong counts.
- slice-level match precision, match recall, match F1, macro F1, and accuracy
  where sample size is sufficient.

## Cost Comparison Logic

Gold labels are the trusted quality standard, not the main cost baseline. The cost comparison should be:

```text
direct_llm_matcher cost = LLM cost paid for every prediction

llm_selected_distilled_student cost =
  one-time LLM teacher-labeling cost
  + compact student training cost
  + cheap compact student inference cost
```

The thesis should report quality and cost together:

| Question | Compare |
|---|---|
| quality gap from trusted labels | `gold_random` student vs `llm_random` and `llm_active_*` students |
| selection value under same cost | `llm_random` vs `llm_active_*` at the same budget |
| cost advantage over direct LLM use | `direct_llm_matcher` vs selected-label distilled student |
| practical stabilization | `llm_active_*` vs `mixed_gold_llm_active` |

## Failure-Slice Logic

The thesis should not stop at aggregate F1. WDC Products is useful because it
contains product-specific difficulty patterns. Phase 7 should join gold labels,
teacher labels, direct LLM predictions, and student predictions by `pair_id`,
then analyze at least these slices when fields are available:

| Slice | Why It Matters |
|---|---|
| hard negatives | likely false positives from superficially similar products |
| missing brand/title/description | tests robustness to incomplete product records |
| brand conflict | product records may share titles but disagree on brand |
| model-number conflict or overlap | product matching often depends on small identifiers |
| long descriptions | tests prompt truncation and noisy attribute effects |
| price/currency mismatch | catches pairs that are semantically similar but commercially different |

## Current Baseline Context

Existing Phase 3 validation result at budget `128`:

| Variant | Train Rows | Match Precision | Match Recall | Match F1 | Macro F1 | Accuracy | Invalid Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `label_only` | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 | 0.0000 |
| `structured_rationale` | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 | 0.0004 |

Interpretation: structured rationales increased recall but badly hurt precision and overall F1. This justifies pivoting away from rationale supervision and testing answer-only teacher labels instead.

## Artifact Paths

Teacher labels:

`data/cache/wdc_products/teacher_labels/train_{budget}.{selection_strategy}.openrouter.answer_only_v1.labels.jsonl`

Teacher rejects:

`data/cache/wdc_products/teacher_labels/train_{budget}.{selection_strategy}.openrouter.answer_only_v1.rejects.jsonl`

Selection manifests:

`data/cache/wdc_products/selection/train_{budget}.{selection_strategy}.manifest.jsonl`

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
| Continue | `llm_active_*` outperforms or usefully diagnoses `llm_random` at the same budget, and the best selected-label student is cheaper than projected direct LLM inference at scale |
| Revise | active selection mostly selects noisy examples, but error pattern suggests a simpler hybrid, better seed model, or mixed gold+LLM supervision |
| Stop | random and active LLM-label students are worse than the old label-only baseline with no useful diagnostic story |

After full budget study:

| Decision | Condition |
|---|---|
| Continue to thesis writing | clear label-efficiency or cost-efficiency pattern exists |
| Add optional student | FLAN-T5 behavior is too seq2seq-specific |
| Add optional dataset | WDC-only result is too narrow for advisor expectations |

## Out Of Scope For First Execution

- Full DistillER reproduction.
- Multi-teacher comparison.
- Complex iterative active-learning policy with repeated retraining.
- Explanation/rationale generation as the main claim.
- More than one dataset before WDC pilot.
- More than one student before WDC pilot.

## Immediate Next Phase

Phase 3 should run the already planned answer-only teacher and direct-LLM
pipeline, then add fixed selection manifests for `random` and the first active
strategy before generating active teacher-label caches.
