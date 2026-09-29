# Cost-Aware Active LLM Labeling for Low-Budget Entity Matching

**Preliminary thesis draft for advisor review**

**Status:** Method and experimental design draft; Phase 5 validation results
included; robustness and full-budget results pending  
**Prepared:** 28 July 2026  
**Primary dataset:** WDC Products  
**Student selected for the next study:** Qwen3-Reranker-0.6B with LoRA

> **Reading note.** Results marked `[PRELIMINARY]` use the fixed validation
> split and training seed 42. Sections marked `[PENDING]` will be completed
> after the predeclared robustness and full-budget studies. The fixed test split
> has not been evaluated.

## Working Titles

**English:** Cost-Aware Active LLM Labeling for Low-Budget Entity Matching

**Vietnamese, provisional:** Gán nhãn chủ động bằng mô hình ngôn ngữ lớn có xét
đến chi phí cho bài toán đối sánh thực thể trong điều kiện ít nhãn

## Abstract `[DRAFT]`

Entity Matching determines whether two records describe the same real-world
entity. Although large language models can perform this task directly and can
also generate training labels, repeated inference through a large model may be
unnecessarily expensive when many record pairs must be compared. This thesis
studies a low-label alternative on WDC Products. A selection strategy chooses a
small set of candidate product pairs, an LLM teacher labels those pairs once,
and a compact student learns from the generated labels for subsequent local
inference. The central comparison is between random LLM-label distillation and
active LLM-label distillation under the same label budget, teacher, prompt,
student architecture, and evaluation split. A gold-label student provides
supervised quality context, while a direct LLM matcher provides a repeated
inference-cost reference.

The implemented active strategy allocates the labeling budget across four
attribute-based candidate groups: easy match candidates, hard match candidates,
easy non-match candidates, and hard negative candidates. The strategy does not
use validation or test labels and freezes its selection manifest before teacher
labels are inspected. The selected compact model is Qwen3-Reranker-0.6B,
adapted with LoRA through its native yes/no reranking interface.

`[PRELIMINARY]` At a training budget of 128 pairs and seed 42, the actively
selected LLM-label student achieved 0.7472 match F1 and 0.8386 macro F1,
compared with 0.7267 and 0.8283 for random LLM-label distillation. The direct
LLM matcher achieved higher quality but incurred a cost for every evaluated
pair. These observations provide a promising validation signal, but they do
not yet establish a stable active-selection advantage: the result is based on
one training seed and the paired-bootstrap confidence interval for the
macro-F1 difference includes zero. A predeclared multi-seed robustness study
and a full label-budget study will determine whether the observed improvement
is repeatable and cost-effective.

**Keywords:** Entity Matching, Entity Resolution, active learning, large
language model, knowledge distillation, low-label learning, cost efficiency,
WDC Products.

# Mở đầu `[DRAFT]`

## 1. Motivation

Entity Matching, also called Entity Resolution in broader settings, identifies
records that refer to the same real-world object. Product matching is a
particularly difficult form of this problem. Records collected from different
web shops may omit attributes, use different languages or currencies, shorten
model identifiers, or reuse similar marketing descriptions for different
product variants. False matches can merge distinct products, while missed
matches fragment information about the same product.

Modern language models offer two relevant capabilities. First, an LLM can act
as a direct matcher by receiving two records and returning `match` or
`non_match`. Second, an LLM can act as a teacher that labels training examples
for a smaller model. Direct matching is simple but repeatedly pays the cost and
latency of the teacher for every prediction. Teacher-label distillation pays
for labels once and then uses a compact model for repeated inference.

Plain distillation does not answer which examples should receive scarce
teacher calls. Under a small labeling budget, random pairs may be redundant or
too easy. Conversely, choosing only ambiguous pairs may expose teacher errors
and provide noisy supervision. This creates a cost-aware data-selection
problem: which product pairs are worth spending LLM calls on?

## 2. Research Question

> Under low-label budgets on WDC Products, can active selection of LLM-labeled
> training pairs produce compact Entity Matching students that outperform
> random LLM-label distillation at the same labeling cost, while becoming
> cheaper than repeated direct LLM matching?

The supporting questions are:

1. Does active selection improve match F1 or macro F1 relative to random
   LLM-label distillation at the same budget?
2. Is any improvement stable across training seeds and label budgets?
3. Does active selection choose informative examples or disproportionately
   choose pairs on which the LLM teacher is wrong?
4. At what comparison volume does one-time teacher labeling plus compact
   inference become cheaper than repeated direct LLM matching?
5. Which WDC difficulty slices benefit from or resist active selection?

## 3. Scope

The primary dataset is WDC Products. Candidate selection is restricted to the
training split. Validation and test gold labels are never used for pair
selection or teacher-label generation. The primary compact student is
Qwen3-Reranker-0.6B with LoRA. The initial teacher and direct matcher use
`openai/gpt-5.4-mini` through OpenRouter with deterministic answer-only
prompting.

This thesis does not claim to invent active learning, LLM-generated
supervision, or knowledge distillation for Entity Matching. Its intended
contribution is a controlled study of the scarce-call setting: active versus
random teacher-label selection at equal budgets, explicit cost accounting, and
failure analysis on a difficult product-matching benchmark.

An additional Entity Matching dataset is considered an optional
external-validity experiment after the WDC result and protocol are fixed. It
is not used to change the primary method after inspecting WDC test results.

## 4. Expected Contributions

1. A reproducible pipeline in which selection manifests, teacher labels,
   training targets, student checkpoints, predictions, and cost evidence are
   linked through persisted contracts.
2. A same-budget comparison of random and actively selected LLM-generated
   supervision for compact Entity Matching students.
3. A quality-and-cost comparison against both a gold-label compact student and
   repeated direct LLM matching.
4. A label-efficiency, break-even, and failure-slice analysis that identifies
   when active LLM labeling helps, fails, or amplifies teacher noise.

# Chương 1. Tổng quan `[SKELETON]`

## 1.1 Entity Matching and Entity Resolution

This section will define pairwise Entity Matching, distinguish it from
clustering-level Entity Resolution, and summarize traditional similarity,
feature-engineering, and neural matching approaches.

## 1.2 Large Language Models for Entity Matching

Prior work shows that generative language models can directly classify record
pairs through prompting and can be adapted for Entity Matching. This literature
supports the direct-LLM baseline but also motivates concern about repeated
inference cost, latency, reproducibility, and access to closed providers.

## 1.3 LLM-Generated Labels and Knowledge Distillation

Distillation work already establishes that an LLM can provide supervision to a
smaller Entity Matching model. DistillER is the closest novelty boundary for
the current work. The thesis therefore does not present the teacher-to-student
pipeline itself as novel.

Zhang et al. also demonstrate, in chemical named entity recognition, that large
LLM-annotated corpora can train lightweight domain models whose performance is
close to models trained from human annotations. Their study strengthens the
general motivation for local student deployment, but differs from the current
work in task and supervision scale: it uses tens of thousands of annotated
paragraphs and does not compare active and random teacher-call allocation under
tiny fixed budgets.

## 1.4 Active Learning and Data Selection

Active learning studies how to choose informative unlabeled examples for
annotation. Common principles include uncertainty, diversity, representative
coverage, and hard-negative sampling. Applying these ideas to an LLM teacher
introduces an additional concern: examples that are difficult for the student
may also be difficult for the teacher. An active strategy can therefore
increase information or increase label noise.

## 1.5 Research Gap and Positioning

Existing work provides evidence for direct LLM matching, LLM-generated
supervision, compact-student distillation, and active learning. The remaining
setting examined here is narrower:

> Given only tens to hundreds of affordable teacher labels for product
> matching, does an attribute-aware selection policy provide more student
> quality per teacher call than random LLM-label distillation?

The thesis is positioned as a new setting and controlled empirical study,
rather than a new general learning architecture.

# Chương 2. Cơ sở lý thuyết và phương pháp nghiên cứu `[DRAFT]`

## 2.1 Binary Entity Matching

For a pair of records \(x=(r_a,r_b)\), the model predicts
\(y\in\{0,1\}\), where \(1\) denotes `match` and \(0\) denotes
`non_match`. A record contains available product attributes such as title,
brand, description, price, and currency. Missing values remain explicit so the
model can distinguish absence from an empty string.

## 2.2 Teacher-Label Distillation

Let \(U\) be the candidate pool from the WDC training split, \(S(U,B)\) a
selection procedure with budget \(B\), and \(T\) an LLM teacher. The teacher
produces labels

\[
\tilde{D}_{B}=\{(x_i,T(x_i))\mid x_i\in S(U,B)\}.
\]

A compact student \(f_\theta\) is trained on \(\tilde{D}_B\). The key comparison
holds the teacher, budget, prompt, student, and evaluation split fixed while
changing \(S\):

\[
S_{\mathrm{random}} \quad \text{versus} \quad
S_{\mathrm{active}}.
\]

This design isolates the value of selection from the already known value of
LLM-generated supervision.

## 2.3 Evaluation Metrics

For the `match` class, precision is the fraction of predicted matches that are
correct, while recall is the fraction of true matches recovered:

\[
P_m=\frac{TP}{TP+FP}, \qquad
R_m=\frac{TP}{TP+FN}.
\]

Match F1 is their harmonic mean:

\[
F1_m=\frac{2P_mR_m}{P_m+R_m}.
\]

Non-match F1 is calculated by treating `non_match` as the target class. Macro
F1 gives both classes equal weight:

\[
F1_{\mathrm{macro}}=\frac{F1_m+F1_{nm}}{2}.
\]

Macro F1 is the primary metric because the validation and test splits contain
many more non-matches than matches. Match F1 is the main secondary metric
because recognizing the minority match class is central to Entity Matching.
Accuracy is reported but is not used alone for decisions.

## 2.4 Cost Model

Direct LLM cost grows with every evaluated pair:

\[
C_{\mathrm{direct}}(N)=N c_T,
\]

where \(c_T\) is the observed average direct-teacher cost per pair.

The distilled-student path includes one-time labeling, training, and local
student inference:

\[
C_{\mathrm{student}}(N,B)=B c_L+c_{\mathrm{train}}+N c_S.
\]

Here, \(c_L\) is teacher-label cost, \(c_{\mathrm{train}}\) is student training
cost, and \(c_S\) is student inference cost per pair. The break-even query count
is the smallest \(N\) for which
\(C_{\mathrm{student}}(N,B)\leq C_{\mathrm{direct}}(N)\).

Measured teacher token usage and measured synchronized GPU time are the
primary evidence. GPU costs are reported under all predeclared low, base, and
high sensitivity scenarios rather than selecting one favorable rate after
results are known.

# Chương 3. Xây dựng hệ thống

## 3.1 System Overview

The implemented system separates data selection, LLM interaction, student
training, and evaluation:

```text
WDC Products raw data
  -> serialized train/validation/test pairs
  -> random or active training-pair selection
  -> immutable selection manifest
  -> answer-only LLM teacher labeling
  -> validation and label cache
  -> supervision-specific target files
  -> compact-student training
  -> validation/test prediction
  -> metrics, timing, cost, and failure analysis
```

Teacher calls occur only during direct-baseline measurement or training-label
creation. Final distilled-student inference does not call the teacher.

## 3.2 Data Preparation

Each WDC pair is serialized with:

- a unique pair identifier;
- split identifier;
- record identifiers;
- available title, brand, description, price, and currency fields;
- gold label, retained only where allowed for supervised targets or
  evaluation;
- metadata required for reproducibility and later failure analysis.

The split boundaries are preserved throughout the pipeline. Only training
pairs can enter a teacher-label candidate pool. Validation and test pairs are
used exclusively for evaluation against gold labels.

## 3.3 Random Low-Label Sampling

The trusted `gold_random` context uses deterministic balanced samples. For
budget \(B\), the sampler selects \(B/2\) matches and \(B/2\) non-matches from
the training split. The current prepared budgets are 16, 32, 64, and 128. A
full 2,500-pair training target also exists for diagnostic context.

`llm_random` uses the fixed random selection manifest at the same total budget
as the active strategy. Its labels come from the LLM teacher rather than the
dataset gold labels.

## 3.4 Attribute-Aware Active Selection

The first active strategy is `llm_active_bucketed_v1`. It assigns attribute-only
scores without consulting the training gold label and allocates 25% of the
budget to each candidate group:

| Candidate group | Intended information |
|---|---|
| Easy match candidate | Strong compatible title, brand, model, or attribute evidence |
| Hard match candidate | Positive evidence mixed with missing fields or apparent conflicts |
| Easy non-match candidate | Low similarity and little positive correspondence |
| Hard negative candidate | High surface similarity but conflicting model, brand, price, or currency evidence |

For budget 128, each group receives a quota of 32 pairs. Every manifest records
the strategy, rank, score, selection seed, budget, bucket, bucket rank, quota,
and the attribute features used for selection. The manifest is written before
teacher responses or student results are inspected.

The strategy is best understood as structured coverage rather than a learned
active policy. This simplicity keeps the first thesis comparison auditable and
reduces the risk of tuning a complex selector on validation outcomes.

## 3.5 Teacher Labeling

The teacher configuration is:

| Field | Value |
|---|---|
| Provider | OpenRouter |
| Model | `openai/gpt-5.4-mini` |
| Prompt version | `answer_only_v1` |
| Temperature | 0.0 |
| Maximum output tokens | 16 |
| Valid outputs | `match`, `non_match` |

Each cache row preserves the pair identifier, prompt version, actual model
slug, input and output token counts, estimated cost, parsed label, validity,
and selection metadata. Malformed outputs are rejected rather than silently
treated as labels.

## 3.6 Training-Target Construction

The pipeline creates three required compact-student targets:

| Target | Pair selection | Label source | Role |
|---|---|---|---|
| `gold_random` | Random balanced sample | Dataset gold | Trusted quality context |
| `llm_random` | Fixed random manifest | LLM teacher | Random distillation control |
| `llm_active_bucketed_v1` | Four-bucket active manifest | LLM teacher | Proposed active arm |

Validation and test targets always use dataset gold labels. The direct LLM
matcher is not a training target because it predicts evaluation pairs directly.

## 3.7 Compact Student

After FLAN-T5 and an initial ModernBERT diagnostic produced `REVISE`
decisions, Qwen3-Reranker-0.6B was screened on the unchanged budget-128 targets
and validation split. It was selected because its pretrained reranking
interface produced both classes and gave a usable active-versus-random signal.

The student receives the pair and a fixed instruction:

> Determine whether Record A and Record B describe the same real-world product.
> Answer yes only when they refer to the same product.

The final `no` and `yes` token logits are mapped to non-match and match
probabilities. Adaptation uses LoRA rather than a randomly initialized
classification head.

| Configuration | Value |
|---|---|
| Base model | `Qwen/Qwen3-Reranker-0.6B` |
| Maximum input length | 4,096 tokens |
| Input truncation | Disabled |
| Fine-tuning | LoRA |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Microbatch | 1 |
| Gradient accumulation | 16 |
| Effective batch | 16 |
| Learning rate | \(2\times10^{-4}\) |
| Maximum epochs | 10 |
| Early-stopping patience | 3 |
| Gradient checkpointing | Enabled |

The selected adapter is merged into a standalone checkpoint. Validation-only
threshold selection is persisted and reused for evaluation; test labels never
influence the threshold.

## 3.8 Reproducibility and Recovery

Run contracts record the Git revision, student configuration, immutable model
revision, dependency versions, device, precision, batch sizes, seed, and hashes
of training and validation targets. Stage outputs are reused only when their
contracts match. Mismatched forced reruns archive stale artifacts instead of
mixing them into a new summary.

For the planned robustness study, each training seed uses a separate output
root. This keeps complete artifacts for paired comparisons and prevents a
seed-specific run from replacing another.

# Chương 4. Thực nghiệm

## 4.1 Dataset and Splits

The study uses the WDC Products configuration with 80 corner cases, the small
training setting, 100 unseen test entities, and preparation seed 42.

| Split | Pairs | Matches | Non-matches | Match rate |
|---|---:|---:|---:|---:|
| Train | 2,500 | 500 | 2,000 | 20.0% |
| Validation | 2,500 | 500 | 2,000 | 20.0% |
| Test | 4,500 | 500 | 4,000 | 11.1% |

The test split remains untouched during model screening, threshold selection,
robustness analysis, and Phase 6 design.

## 4.2 Low-Label Budgets

| Budget | Prepared pairs | Matches in balanced gold sample | Non-matches |
|---:|---:|---:|---:|
| 16 | 16 | 8 | 8 |
| 32 | 32 | 16 | 16 |
| 64 | 64 | 32 | 32 |
| 128 | 128 | 64 | 64 |

Budget 128 is the completed screening budget. `[PENDING]` Phase 6 will train the
accepted student across the prepared budgets. Budget 256 may be added only
after its sampler output and protocol are declared; it does not currently
exist in the prepared cache.

## 4.3 Experiment Arms

| Arm | Implementation | Question answered |
|---|---|---|
| Gold-label student | `gold_random` | How well does trusted low-budget supervision train the same student? |
| Direct LLM matcher | Teacher predicts every evaluation pair | What quality and repeated inference cost does direct LLM matching provide? |
| Random LLM-label student | `llm_random` | What does ordinary random LLM-label distillation provide? |
| Active LLM-label student | `llm_active_bucketed_v1` | Does structured selection improve quality per teacher call? |

The primary comparison is active versus random LLM-label students. The
gold-label student is quality context, not the primary cost baseline. The
direct LLM matcher is the repeated inference-cost baseline.

## 4.4 Direct LLM Baseline

The direct baseline applies the same answer-only task family to all 2,500
validation pairs. It uses `openai/gpt-5.4-mini`, temperature 0.0, and records
token-level estimated cost for each prediction.

The completed validation cache contains 2,500 valid predictions and no invalid
outputs. It records 969,364 input tokens, 15,439 output tokens, and a total
estimated provider cost of USD 0.7965 at the recorded call-time prices.

## 4.5 Primary Metrics

The following metrics are reported:

- match precision;
- match recall;
- match F1;
- non-match F1;
- macro F1;
- accuracy;
- \(TP, FP, TN, FN\);
- invalid-output rate for teacher and direct-LLM calls.

Macro F1 is primary. Match F1 is the principal secondary metric. Cost results
report measured seconds and all predeclared monetary sensitivity scenarios.

## 4.6 Preliminary Model-Screening Protocol

All three supervision targets contain 128 training pairs and use the same 2,500
gold-labeled validation pairs. The test split is excluded. The Qwen input-length
preflight verifies that complete prompts fit within 4,096 tokens with
truncation disabled. Seed 42 controls Python, NumPy, PyTorch, CUDA, and training
data order.

## 4.7 Robustness Protocol `[PENDING EXECUTION]`

The immediate extension tests whether the active improvement is caused by
training randomness.

| Field | Predeclared value |
|---|---|
| Variants | `llm_random`, `llm_active_bucketed_v1` |
| Budget | 128 |
| Training seeds | 42, 43, 44, 45, 46 |
| Targets | Fixed existing target files |
| Teacher calls | None |
| Primary metric | Macro F1 |
| Secondary metric | Match F1 |
| Evaluation | Fixed validation split |

The analysis will report the per-seed active-minus-random difference, mean,
standard deviation, paired confidence interval, and number of active wins. The
working continuation criterion is:

1. active wins macro F1 in at least four of five paired seeds;
2. mean macro-F1 difference is positive;
3. mean match-F1 difference is positive;
4. neither arm collapses to one predicted class.

This criterion determines whether the current signal is sufficiently stable to
justify the full budget study. It is not a final hypothesis test.

## 4.8 Selection-Seed Robustness `[PROPOSED]`

If training-seed robustness passes, a stronger extension will create three
independent random and active selection manifests at budget 128. Newly selected
pairs require new teacher labels. Selection variance will be reported
separately from training variance. This experiment is optional before the
advisor checkpoint and may instead become part of Phase 6.

## 4.9 Full Budget Study `[PENDING]`

The accepted student will compare `gold_random`, `llm_random`, and
`llm_active_bucketed_v1` across budgets 16, 32, 64, and 128. The output will
include:

- a full metric table;
- label-efficiency curves;
- active-minus-random same-budget differences;
- training and inference time;
- total teacher-labeling cost;
- break-even comparison with direct LLM matching;
- selected-pair and teacher-noise failure slices.

## 4.10 Threats to Validity

**Optimization variance.** The completed active advantage uses one training
seed. The multi-seed robustness study directly addresses this risk.

**Selection variance.** One active and one random manifest cannot show that the
selection procedure is stable. Independent selection seeds are needed for that
claim.

**Teacher noise.** Active selection may favor ambiguous pairs that the teacher
labels incorrectly. Teacher-versus-gold disagreement is used only for
post-selection audit and never to construct the manifest.

**Dataset scope.** WDC Products supports a focused product-matching claim but
does not establish transfer to other Entity Matching domains. A second dataset
will be considered after the primary protocol is fixed.

**Cost assumptions.** GPU-hour rates are analytical replacement-cost
scenarios, not observed Colab charges. Measured seconds remain the primary
provider-independent evidence.

**Researcher degrees of freedom.** Student architecture, targets, splits,
prompt, teacher, and decision metrics are frozen before robustness results are
inspected. The test split is evaluated only after the final protocol and
decision rule are fixed.

# Chương 5. Kết quả và thảo luận `[PARTIAL]`

## 5.1 Preliminary Budget-128 Validation Results

**Table 5.1. Preliminary validation results at budget 128, training seed 42**

| Variant | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|
| `gold_random` | 0.7104 | 0.8159 | 0.8764 |
| `llm_random` | 0.7267 | 0.8283 | 0.8884 |
| `llm_active_bucketed_v1` | **0.7472** | **0.8386** | **0.8904** |
| Direct LLM matcher | 0.8734 | 0.9208 | 0.9492 |

> `[PRELIMINARY]` Validation only; seed 42; fixed test split not evaluated.

At this seed, active selection improved match F1 by 0.0205, macro F1 by 0.0103,
and accuracy by 0.0020 relative to random LLM-label distillation. The larger
change in match F1 suggests that the selected training pairs primarily helped
the more difficult minority match class. The small accuracy change illustrates
why accuracy alone is inadequate for the imbalanced validation set.

The LLM-labeled students both exceeded the gold-random student in this run.
This does not imply that noisy labels are generally better than gold labels:
the selected pairs and label sources are not identical interventions, and the
result currently has only one training seed.

The direct LLM matcher remained substantially stronger in quality, with 0.8734
match F1 and 0.9208 macro F1. The student is therefore evaluated as a
quality-cost trade-off rather than a strict quality replacement for direct
matching.

## 5.2 Uncertainty of the Preliminary Difference

The paired bootstrap over validation examples gave an approximate 95%
confidence interval of \([-0.0045, 0.0247]\) for the active-minus-random macro-F1
difference. Because this interval includes zero, the current validation
predictions do not establish a statistically clear advantage. Moreover, an
example-level bootstrap does not capture training-seed or selection-seed
variance. The planned paired-seed experiment is therefore necessary before
interpreting the active result as repeatable.

## 5.3 Preliminary Cost Observation

The direct validation run cost approximately USD 0.7965 for 2,500 pairs under
the recorded provider usage. Under the predeclared base analytical GPU rate of
USD 1 per hour, the active budget-128 student path was estimated at
approximately USD 0.732 at 2,500 comparisons and reached break-even near 2,281
queries.

These values are preliminary sensitivity evidence, not observed Colab charges
or current market quotes. Final reporting will show the low, base, and high GPU
scenarios together:

| Scenario | Analytical GPU-hour rate |
|---|---:|
| Low | USD 0.25 |
| Base | USD 1.00 |
| High | USD 4.00 |

## 5.4 Model-Screening Interpretation

The earlier label-only FLAN-T5 pilot was usable but weak. The first ModernBERT
diagnostic collapsed toward one class under the 128-row budget and was retained
as negative model-screening evidence. Qwen's pretrained yes/no reranking
interface, LoRA adaptation, complete-input contract, and validation threshold
avoided that collapse. This supports selecting Qwen for robustness and budget
studies without claiming that Qwen is universally the best compact Entity
Matching architecture.

## 5.5 Multi-Seed Robustness Results `[PENDING]`

**Table 5.2. Paired training-seed results at budget 128**

| Seed | Random match F1 | Active match F1 | Difference | Random macro F1 | Active macro F1 | Difference |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7267 | 0.7472 | +0.0205 | 0.8283 | 0.8386 | +0.0103 |
| 43 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 44 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 45 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 46 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| Mean ± SD | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

**Required interpretation after filling the table:**

- active macro-F1 wins: `[PENDING]/5`;
- paired mean macro-F1 difference: `[PENDING]`;
- paired 95% confidence interval: `[PENDING]`;
- active match-F1 wins: `[PENDING]/5`;
- robustness-gate decision: `[PASS / REVISE / STOP]`.

## 5.6 Label-Efficiency Results `[PENDING]`

**Table 5.3. Validation performance across label budgets**

| Budget | Gold-random macro F1 | LLM-random macro F1 | Active macro F1 | Active − random |
|---:|---:|---:|---:|---:|
| 16 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 32 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 64 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 128 | `[PENDING MULTI-SEED SUMMARY]` | `[PENDING MULTI-SEED SUMMARY]` | `[PENDING MULTI-SEED SUMMARY]` | `[PENDING]` |

`[PENDING FIGURE 5.1: Label-efficiency curves with uncertainty bars.]`

## 5.7 Teacher Noise and Failure Slices `[PENDING]`

The final analysis will join selection metadata, teacher-versus-gold
disagreement, and student predictions. Planned slices include:

- high-overlap hard negatives;
- missing brand, title, model, price, or currency;
- brand conflicts;
- model-number conflicts;
- long descriptions;
- price and currency inconsistencies;
- active-selection bucket;
- teacher-correct versus teacher-wrong selected pairs.

## 5.8 Final Test Results `[PENDING — DO NOT FILL EARLY]`

The fixed test split will be evaluated once after the robustness gate, Phase 6
protocol, student, thresholds, variants, and reporting rules are frozen.

# Chương 6. Kết luận và khuyến nghị `[PLACEHOLDER]`

The conclusion will answer the research question only at the strength supported
by the completed robustness, budget, cost, and test evidence. Possible outcomes
are:

1. **Stable positive result:** active selection consistently improves
   label-efficiency and yields a defensible break-even point.
2. **Conditional result:** active selection helps particular budgets or WDC
   slices but is unstable overall.
3. **Negative result:** the proposed selector does not reliably outperform
   random LLM-label distillation; teacher noise or optimization variance
   explains the initial improvement.

All three outcomes can support a valid master's thesis if the experiment is
controlled and the limitations and failure mechanisms are analyzed honestly.

# Preliminary References

1. Peeters, Steiner, and Bizer. *Entity Matching Using Large Language Models*.
2. Steiner, Peeters, and Bizer. *Fine-Tuning Large Language Models for Entity
   Matching*.
3. Zeakis et al. *DistillER: Knowledge Distillation in Entity Resolution with
   Large Language Models*.
4. Wadhwa et al. *Learning from Natural Language Explanations for Generalizable
   Entity Matching*.
5. Zhang, Vlachos, Liu, and Fang. “Rapid Adaptation of Chemical Named Entity
   Recognition Using Few-Shot Learning and LLM Distillation.” *Journal of
   Chemical Information and Modeling*, 65(9), 4334–4345, 2025.

`[PENDING]` Replace this working list with verified BibTeX entries and the
school-required citation style.

