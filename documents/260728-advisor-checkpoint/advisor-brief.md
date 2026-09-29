# Advisor Brief: Cost-Aware Active LLM Labeling for Entity Matching

**Purpose:** Early scope and experiment-design review before the full budget
study  
**Prepared:** 28 July 2026  
**Evidence status:** Preliminary validation results; fixed test split untouched

## Research Problem

Large language models can directly decide whether two product records describe
the same product, but direct use incurs cost and latency for every comparison.
An alternative is to ask the LLM to label a small training set once and
distill those labels into a compact student. When the teacher-call budget is
small, the unresolved question is which candidate pairs deserve those calls.

## Research Question

> Under low-label budgets on WDC Products, can active selection of LLM-labeled
> training pairs produce compact Entity Matching students that outperform
> random LLM-label distillation at the same labeling cost, while becoming
> cheaper than repeated direct LLM matching?

## Thesis Positioning

This is a **new setting and controlled empirical study**, not a claim that
active learning, LLM labeling, or knowledge distillation is new. Prior work
already studies direct LLM Entity Matching and LLM-to-student distillation.
The proposed contribution is the scarce-call comparison:

- fixed budgets of tens to hundreds of teacher-labeled pairs;
- active versus random selection at equal teacher-label cost;
- compact-student quality and measured inference time;
- break-even cost against direct LLM matching;
- teacher-noise and WDC failure-slice analysis.

The recently identified chemical NER paper by Zhang et al. (2025) is adjacent
but does not answer this question. It trains from approximately 87,000–89,000
LLM-annotated paragraphs per teacher and does not compare active and random
label acquisition under tiny budgets.

## Implemented Pipeline

```text
WDC training candidate pool
    |
    +-- random selection ----------------------+
    |                                           |
    +-- four-bucket active selection -----------+
                                                v
                              fixed selection manifests
                                                |
                                                v
                         GPT-5.4-mini labels pairs once
                                                |
                                                v
                         Qwen3-Reranker-0.6B LoRA student
                                                |
                                                v
                               local repeated inference

Comparisons:
  gold-label student | random LLM-label student
  active LLM-label student | direct LLM matcher
```

The active strategy allocates 25% of the budget to easy-match, hard-match,
easy-non-match, and hard-negative candidates using record attributes only. It
does not use validation or test labels.

## Dataset and Evaluation

| Split | Pairs | Matches | Non-matches |
|---|---:|---:|---:|
| Train | 2,500 | 500 | 2,000 |
| Validation | 2,500 | 500 | 2,000 |
| Test | 4,500 | 500 | 4,000 |

Prepared low-label budgets are 16, 32, 64, and 128. Macro F1 is primary because
the evaluation data are imbalanced; match F1 is the main secondary metric.

## Preliminary Validation Evidence

**Budget 128, training seed 42**

| Variant | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| Gold-random student | 0.7104 | 0.8159 | 0.8764 |
| Random LLM-label student | 0.7267 | 0.8283 | 0.8884 |
| Active LLM-label student | **0.7472** | **0.8386** | **0.8904** |
| Direct LLM matcher | 0.8734 | 0.9208 | 0.9492 |

At this seed, active selection improved match F1 by 0.0205 and macro F1 by
0.0103 over random LLM-label distillation. The direct LLM remained stronger in
quality.

This is promising but not conclusive:

- only one training seed has completed;
- the paired-bootstrap interval for the macro-F1 difference is approximately
  `[-0.0045, 0.0247]`, which includes zero;
- one selection manifest per strategy cannot establish selection stability;
- the fixed test split has not been evaluated.

## Preliminary Cost Evidence

The direct LLM validation run produced 2,500 valid predictions at a recorded
estimated cost of USD 0.7965. Under the predeclared base analytical rate of USD
1 per GPU-hour, the active student path was estimated at USD 0.732 for 2,500
comparisons, with break-even near 2,281 queries.

These GPU rates are analytical sensitivity assumptions, not observed Colab
charges. Final results will report low, base, and high scenarios together and
preserve measured training and inference seconds as primary evidence.

## Immediate Robustness Check

While the thesis draft is reviewed, the same fixed targets will be trained with
paired seeds `42, 43, 44, 45, 46`.

The working continuation criterion is:

1. active wins macro F1 in at least four of five seeds;
2. mean active-minus-random macro F1 is positive;
3. mean active-minus-random match F1 is positive;
4. neither variant collapses to one class.

This check requires no additional teacher calls. If it passes, independent
selection seeds will be considered to test whether the selected pairs, rather
than only model training, were lucky.

## Proposed Next Study

After the robustness decision, the selected Qwen student will be evaluated at
budgets 16, 32, 64, and 128 for:

- `gold_random`;
- `llm_random`;
- `llm_active_bucketed_v1`;
- the fixed direct LLM quality/cost reference.

The study will produce label-efficiency curves, active-minus-random
same-budget comparisons, break-even tables, and failure-slice analysis.
A second Entity Matching dataset is proposed only as an external-validity
extension after the WDC protocol is fixed.

## Decisions Requested from the Advisor

1. Is the WDC-focused scarce-teacher-call setting sufficiently scoped for the
   master's thesis?
2. Is the novelty positioning appropriately conservative?
3. Should an additional Entity Matching dataset be mandatory or optional?
4. Are five training seeds at budget 128 sufficient for the first robustness
   gate?
5. Should independent selection seeds be completed before the full budget
   study?
6. Are budgets 16, 32, 64, and 128 sufficient, or is budget 256 required?
7. Is macro F1 an acceptable primary metric with match F1 as the main
   secondary metric?

## Current Interpretation

The current result supports continuing the study, but not yet claiming that
active selection is reliably superior. The thesis remains viable under three
possible outcomes: stable improvement, improvement limited to particular
budgets or failure slices, or a controlled negative result explaining why the
active selector fails under teacher noise and low-data optimization.

