---
title: Distillation Reading Guide for Active LLM Labeling on WDC Products
date: 2026-07-13T17:51:00+07:00
status: complete
scope: active execution plan
---

# Distillation Reading Guide for Active LLM Labeling on WDC Products

## Summary

The thesis is closest to **hard-label, black-box, task-specific LLM
distillation with active data acquisition**. It is not classical logit
distillation because the OpenRouter teacher returns labels rather than a full
probability distribution. Use classic knowledge distillation as conceptual
background, but use LLM-generated-label, active-learning, and low-resource
entity-matching papers to justify the actual method.

Read the first six papers below before changing the post-pilot experiment.
They give the closest precedents and the most applicable design ideas. Keep the
fixed Phase 5 pilot unchanged; apply new ideas only to predeclared follow-up
experiments after inspecting the pilot.

## Priority Reading

| Priority | Paper | Why it matters here | Apply after the fixed pilot |
|---:|---|---|---|
| 1 | [Evolving Knowledge Distillation with Large Language Models and Active Learning](https://aclanthology.org/2024.lrec-main.593/) (Liu et al., 2024) | Closest conceptual match: an LLM teacher, a small student, and active feedback based on student weaknesses. EvoKD generates new samples, whereas this thesis selects real WDC pairs; that difference is a defensible research gap. | Add an iterative round where the current student identifies weak slices, then spend the next fixed teacher budget on real pool examples from those slices. Compare with one-shot selection at equal cumulative cost. |
| 2 | [Knowledge Distillation in Automated Annotation: Supervised Text Classification with LLM-Generated Training Labels](https://aclanthology.org/2024.nlpcss-1.9/) (Pangakis and Wolken, 2024) | Direct precedent for training a supervised classifier from LLM-generated hard labels. Useful for terminology, teacher-label validation, and studying when label quality transfers to the student. | Report teacher–gold agreement beside student metrics; stratify both by class and difficulty. Do not treat teacher labels as ground truth. |
| 3 | [ActiveLLM: Large Language Model-Based Active Learning for Textual Few-Shot Scenarios](https://aclanthology.org/2026.tacl-1.1/) (Bayer et al., 2026) | Highly relevant to the 128-label cold start: an LLM helps select examples for a compact classifier. It separates using an LLM for **selection** from using it as the **label oracle**, which clarifies this thesis's design space. | Consider a predeclared follow-up with teacher-assisted selection versus cheap heuristic selection, while logging selection-call and labeling-call costs separately. |
| 4 | [Learning from Natural Language Explanations for Generalizable Entity Matching](https://aclanthology.org/2024.emnlp-main.352/) (Wadhwa et al., 2024) | Direct LLM-to-small-model distillation paper for entity matching. It reports gains from natural-language explanations, especially out of domain. The project's opposite 128-row result is valuable negative evidence, not a reason to omit this paper. | Compare task regime, dataset, student objective, explanation quality, and sample size to explain why rationales helped there but hurt precision on WDC here. Keep rationale distillation historical unless a later hypothesis is predeclared. |
| 5 | [The Battleship Approach to the Low Resource Entity Matching Problem](https://arxiv.org/abs/2311.15685) (Genossar et al., 2023) | Entity-matching-specific active selection. It emphasizes latent-space coverage and finding scarce positive pairs, directly relevant to severe class imbalance and the match-recall bottleneck. | Test embedding-space coverage plus positive-candidate quotas as a named selection strategy; compare against the existing bucketed method and random at identical budgets. |
| 6 | [Active Deep Learning on Entity Resolution by Risk Sampling](https://arxiv.org/abs/2012.12960) (Nafa et al., 2020) | Combines estimated error risk with core-set diversity for ER instead of relying on uncertainty alone. It is a strong methodological basis for selecting hard but non-redundant pairs. | Replace hand-built difficulty alone with a student-risk score plus diversity constraint. Avoid selecting 128 near-duplicate hard negatives. |

## Essential Foundations and Design Tools

| Paper | Role in the thesis | Practical lesson |
|---|---|---|
| [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015) | Classical KD foundation. | Soft teacher probabilities carry inter-class information. With a black-box binary teacher returning only `match`/`non-match`, the current method cannot implement this loss; call it hard-label or response/data distillation. |
| [Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://openreview.net/forum?id=ryghZJBKPS) (Ash et al., ICLR 2020) | General batch active-learning baseline, known as BADGE. | Select a batch that is both uncertain and diverse. This is a principled alternative to equal hand-set bucket quotas once an initial student exists. |
| [Cold-start Active Learning through Self-supervised Language Modeling](https://aclanthology.org/2020.emnlp-main.637/) (Yuan et al., EMNLP 2020) | Cold-start selection without a reliable task classifier. | Use pretrained-model surprisal or representation diversity before a student is calibrated; switch to model-aware acquisition in later rounds. |
| [Active Learning for BERT: An Empirical Study](https://aclanthology.org/2020.emnlp-main.638/) (Ein-Dor et al., EMNLP 2020) | Evidence for active learning under tiny budgets, binary labels, and imbalance. | Evaluate multiple seeds and learning curves; active-learning conclusions can depend on initialization and minority-class sampling. |
| [Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation](https://aclanthology.org/2025.acl-long.139/) (Xia et al., ACL 2025) | Addresses noisy single-label LLM annotation by allowing candidate labels under uncertainty. | For binary EM, add an optional `uncertain`/abstain response or repeated/alternative teacher adjudication for ambiguous pairs. Count all extra calls in the budget. |
| [SETEM: Self-ensemble Training with Pre-trained Language Models for Entity Matching](https://doi.org/10.1016/j.knosys.2024.111708) (Ding et al., 2024) | WDC-focused low-resource entity-matching distillation using checkpoint self-ensembles. | Relevant student-side comparator or future enhancement; it does not replace the direct-LLM and random-distillation cost baselines because its teacher is the student training trajectory, not an external LLM. |
| [Distilling Step-by-Step!](https://aclanthology.org/2023.findings-acl.507/) (Hsieh et al., 2023) | Influential rationale-distillation method. | Cite as the positive rationale precedent, then contrast with the project's precision collapse. Rationales are not automatically useful supervision at 128 examples. |

## Recommended Thesis Framing

Use one of these descriptions consistently:

- **Active LLM-label distillation for entity matching**
- **Cost-aware active acquisition of LLM-labeled training pairs**
- **Black-box, hard-label task distillation from an LLM teacher**

Avoid implying access to teacher logits, hidden states, or gradients. If the
teacher returns one parsed class label, the transferred object is a labeled
dataset, not a softened predictive distribution.

The defensible novelty is the intersection and evaluation, not invention of
the ingredients:

1. Real WDC Products pairs rather than LLM-synthesized examples.
2. Fixed, scarce teacher-call budgets and equal-cost random controls.
3. Pair selection tailored to matches, hard negatives, and corner cases.
4. Teacher noise and selection failure analyzed by slice.
5. Break-even comparison against repeated direct LLM inference.

## Concrete Follow-up Experiments

Do not modify the fixed Phase 5 pilot. If it shows a usable signal, predeclare
these in this order:

1. **Budget curve:** 32, 64, 128, 256 labels for random and the same active
   strategy, with multiple selection/training seeds.
2. **Risk plus diversity:** select uncertain/high-risk examples, then enforce
   embedding-space diversity; compare with random and current bucket quotas.
3. **Iterative acquisition:** 64 initial labels, train student, acquire 64 from
   its failure regions; compare with one-shot 128 at equal teacher cost.
4. **Noise-aware acquisition:** teacher abstention or re-query only on
   ambiguous selected pairs; include extra tokens and calls in total cost.
5. **Mixed labels only if needed:** combine a small trusted gold seed with
   actively selected LLM labels when pure LLM labels are too noisy.

For every experiment, report match precision, match recall, match F1, macro
F1, accuracy, confusion counts, teacher–gold agreement, invalid/abstain rate,
class composition, difficulty-slice composition, and total teacher cost.

## Common Pitfalls

- Selecting only maximum-uncertainty cases can give the teacher its noisiest
  examples and produce a worse student.
- Oversampling hard negatives may improve accuracy while starving the student
  of positive match patterns; preserve class- and difficulty-aware reporting.
- Comparing active and random sets by row count is insufficient if prompts,
  retries, or token lengths differ; compare total cost as well.
- A single seed at 128 rows can be dominated by sampling and training variance.
- Teacher–gold disagreement is not always teacher error, but it must be audited
  rather than silently converted into ground truth.
- Explanation-distillation success on other datasets does not override the
  project's negative WDC result.

## Research Method

- Conducted: 2026-07-13 17:51 ICT
- Sources: primary paper pages from ACL Anthology, OpenReview, arXiv, and the
  publisher DOI page
- Coverage: 2015–2026
- Search themes: classical knowledge distillation, LLM-generated labels,
  active LLM distillation, cold-start batch active learning, low-resource
  entity matching, WDC Products
- Selection rule: direct applicability to the active thesis took priority over
  citation count or generic model-compression coverage

## Recommendations

Start with Liu et al. (EvoKD), Pangakis and Wolken, Bayer et al. (ActiveLLM),
Wadhwa et al., Genossar et al. (Battleship), and Nafa et al. (risk sampling).
Use Hinton et al. to define classical KD, not to describe the implemented loss.
Use CanDist, BADGE, and ALPS as sources for carefully predeclared post-pilot
extensions.

## Unresolved Questions

- Does the OpenRouter endpoint expose stable token-level class probabilities
  that could support soft-target KD? The current answer-only contract does not.
- Are teacher errors concentrated in the active hard-negative buckets?
- How much of the active-versus-random difference survives selection and
  training seed variation?
- Is the next teacher call better spent on another hard pair, a diverse pair,
  or adjudicating an uncertain existing label?
