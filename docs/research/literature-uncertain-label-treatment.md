---
title: Treating uncertain labels — what other fields found
date: 2026-11-11
status: research note (literature); no decision taken
context: WDC treatment arms B and C did not beat A; the gold audit (plan 261023) shows why.
---

# Treating uncertain labels — what other fields found

Question: once a training label is flagged as uncertain, what should be done with
it? Sources were read at abstract level (via the pages linked below); details
beyond the abstracts are not verified unless stated.

## Our starting point (WDC, plan 261023 gold audit)

- **Detection works.** Teacher labels are wrong on 17.7% of the 158 detected pairs
  versus 2.2% of the 2,342 others (about 8× enrichment).
- **Treatment fails.** Category rule (arm C) against train gold:
  - flip: 10 pairs, teacher right on all 10, all 10 made wrong;
  - remove (steps 1 and 3): 93 pairs, teacher right on 73;
  - upweight: 52 pairs, 8 wrong labels counted twice.
- Test match F1: C vs A −1.7 (95% CI [−3.6, +0.3], McNemar p = 0.10); B vs A −1.4.

## 1. Correction with a dependent source repeats the error

- **Fan & Li, "Stop Replacing Noise with Noise: Two-Source Reliability Assessment
  for Label Correction and Sample Reweighting in Label-Noise Learning"**
  (Beijing Institute of Technology; arXiv 2608.03432).
  - Problem: refurbishment methods use one cleanliness score for both the observed
    label and the model's pseudo target, so "reducing trust in the observed label
    automatically increases trust in the pseudo target"; the pseudo target comes
    from a model trained on the same noisy data and can repeat its errors.
  - CIFAR-100N analysis: in high-clean bins, pseudo targets followed the noisy label
    99.7% of the time but were correct on 74.7%.
  - Method TRACE: score observed label and pseudo target separately; a replacement
    must pass its own reliability check. Gains on CIFAR-10/100 and Clothing1M
    (DivideMix 74.8% → 77.6%).
- **Relevance:** our flips come from an evidence check by the teacher's own model
  family, not an independent source.

## 2. Removing flagged examples loses clean hard examples

- **Yuan, Feng & Liu, "Late Stopping: Avoiding Confidently Learning from Mislabeled
  Examples"** (arXiv 2308.13862). Small-loss selection also discards clean hard
  examples; Late Stopping removes only "high-probability mislabeled examples" while
  "retaining the majority of clean hard examples". Selection criterion FkL (epochs
  until an example is first learned for k consecutive epochs).
- **Srikanth et al., "An Empirical Study of Automated Mislabel Detection in Real
  World Vision Datasets"** (Stanford; arXiv 2312.02200). Removing detected
  mislabels gave "per-class performance improvements of up to 8%" in small data
  regimes, with "diminishing returns as the size of the data increases"; removal
  hurt one CheXpert class; per-image removal "performs poorly out of the box".
- **Relevance:** our detected pairs are the hard pairs, and most of their labels
  were right.

## 3. Soft labels instead of hard actions

- **Peterson et al., "Human Uncertainty Makes Classification More Robust"**
  (ICCV 2019; CIFAR-10H). Training on full human label distributions improves
  generalisation to shifted test sets and robustness to adversarial attacks.
- **Uma et al., "Learning from Disagreement: A Survey"** (JAIR 72, 2021). With
  large datasets and many good judgments per item, "training directly with soft
  labels achieved better results than training from aggregated or even gold
  labels"; otherwise "leveraging both gold and soft labels generally achieved the
  best results". Strong dataset effect; conclusions depend on the evaluation.
- **Relevance:** we have 3 teacher votes plus an OOF student probability per pair,
  closer to Uma's "combine" case than to full soft labels.

## 4. LLM-generated labels

- **Ye, Shah, Zhang & Chava, "Calibrating Pre-trained Language Classifiers on
  LLM-generated Noisy Labels via Iterative Refinement" (SiDyP)** (KDD 2025;
  arXiv 2505.19675). LLM label noise is more context-dependent than synthetic noise.
  Uncertain items keep candidate labels retrieved from neighbours in embedding space
  (top two when unsure) instead of a hard flip; about 7% accuracy gain for a BERT
  classifier trained on LLM labels.
- **Li et al., "CoAnnotating: Uncertainty-Guided Work Allocation between Human and
  Large Language Models for Data Annotation"** (EMNLP 2023). LLM uncertainty decides
  which items go to humans; up to 21% improvement over random allocation.

## Background references

- **Song et al., "Learning from Noisy Labels with Deep Neural Networks: A Survey"**
  (IEEE TNNLS 2022; arXiv 2007.08199). 62 methods in five groups.
- **Karimi et al., "Deep learning with noisy labels: exploring techniques and
  remedies in medical image analysis"** (arXiv 1912.02911). Experiments on three
  medical imaging datasets; its recommended methods are not listed in the abstract
  (full paper not read).

## Candidate variants for WDC (not decided)

WDC is used as the development dataset; Companies and Geo-ER stay untouched for
confirmation. Each variant is first checked at the label level (free), and trained
only if its labels are closer to gold than arm A's.

1. **No flip without an independent second source** (from 1).
2. **Remove only the most likely wrong pairs** (from 2).
3. **Soft targets for detected pairs** (from 3, 4). Needs trainer support: the
   trainer currently takes hard targets only and upweights by duplicate rows.
4. **Human review of detected pairs** (from 4, CoAnnotating).

## Sources

- https://arxiv.org/pdf/2608.03432
- https://arxiv.org/pdf/2308.13862
- https://arxiv.org/pdf/2312.02200
- https://mlanthology.org/iccv/2019/peterson2019iccv-human/
- https://jair.org/index.php/jair/article/view/12752
- https://arxiv.org/pdf/2505.19675
- https://aclanthology.org/2023.emnlp-main.92
- https://arxiv.org/abs/2007.08199
- https://arxiv.org/abs/1912.02911v4
