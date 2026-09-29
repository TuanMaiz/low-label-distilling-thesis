---
title: Uncertainty-aware LLM label refinement literature map
date: 2026-09-10
status: research-note
---

# Uncertainty-Aware LLM Label Refinement Literature Map

## Summary

The advisor's extension changes the thesis from a controlled replication of
LLM labeling for entity matching into a noisy-label detection and selective
refinement study. The defensible novelty is the post-label combination of LLM
consistency, out-of-sample student uncertainty, and teacher–student
disagreement under a fixed refinement budget. Generic relabeling alone is not
novel because Steiner and Bizer already evaluate global evidence-based
relabeling and relabel-drop.

## Research Methodology

- Researched: 2026-09-10.
- Scope: entity matching, LLM annotation reliability, annotation-error
  detection, training dynamics, noisy-label filtering/reweighting.
- Sources: primary papers and official proceedings from ACL Anthology, PMLR,
  NeurIPS, OpenReview, JAIR/Google Research, and arXiv.
- Recency: 2023–2026 for LLM work; older foundational noisy-label methods kept.

## Advisor-Proposed Direction

```text
LLM labels training pairs
  -> estimate per-label risk
     - LLM consistency
     - student uncertainty
     - teacher-student disagreement
  -> identify unreliable labels
  -> relabel / filter / confidence-weight
  -> train compact entity matcher
  -> evaluate detection, downstream F1, error propagation, and end-to-end cost
```

Gold labels remain hidden from risk scoring and refinement. They are joined
only after the risk ranking is frozen to evaluate label-error detection and
downstream performance.

## Closest Entity-Matching Work

### Steiner and Bizer (2026)

*Labeling Training Data for Entity Matching Using Large Language Models* is the
closest baseline. It studies pair selection, three teachers, post-processing,
and multiple students across five benchmarks. Its post-processing includes a
global conservative LLM review, replacing changed labels, dropping changed
labels, and graph closure filters. It does not center post-label student
uncertainty, combined teacher–student risk, or selective refinement-budget
curves.

### Kamsteeg et al. (2025)

*Confidence Calibration in Large Language Model-Based Entity Matching* compares
temperature scaling, Monte Carlo dropout, and ensembles for RoBERTa entity
matching on datasets including DBLP-ACM. Useful for defining student uncertainty
and calibration. Temperature scaling requires labeled calibration data;
MC-dropout and ensemble dispersion are more compatible with a zero-gold risk
pipeline.

### Peeters et al. (2023/2025)

*Entity Matching using Large Language Models* motivates direct LLM matching,
unseen-entity robustness, and LLM-generated error descriptions. Useful for the
direct-teacher and corner-case analysis background, not a label-refinement
method by itself.

### Chen et al. (NeurIPS 2024)

*Entity Alignment with Noisy Annotations from Large Language Models* is an
adjacent knowledge-graph entity-alignment paper with active selection and an
unsupervised label refiner. It is not tabular pairwise entity matching, but must
be discussed when claiming novelty for unsupervised LLM-label refinement.

## LLM Consistency and Annotation Reliability

### Knowledge Distillation in Automated Annotation (2024)

Labels each sample three times at temperature 0.7 and defines consistency as
the fraction agreeing with the modal label. Fully consistent annotations are
reported as substantially more accurate, but filtering inconsistent labels
does not improve the downstream BERT models in that study. This directly
supports measuring teacher consistency and warns that detecting hard cases does
not guarantee better training after dropping them.

### Lee, Hong, and Thorne (COLING 2025)

*Evaluating the Consistency of LLM Evaluators* distinguishes self-consistency
from consistency across scoring scales and shows that strong proprietary models
are not automatically consistent evaluators. Supports prompt/run stability as
a measured signal rather than trusting model reputation.

### Nahum et al. (EMNLP 2025)

*Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their
Effect on Model Performance* uses an LLM ensemble to flag suspected errors in
existing datasets and studies how correcting labels changes evaluation. Useful
for multi-judge disagreement and for the warning that benchmark gold may itself
be wrong.

## Student-Side Label-Error Detection

### Confident Learning (Northcutt, Jiang, and Chuang, 2021)

Estimates dataset label issues from predicted class probabilities, probabilistic
thresholds, and a noisy/latent-label joint distribution. It is model-agnostic
and naturally supports binary entity matching. Its main relevance is using
out-of-sample predicted probabilities to rank likely label errors without
revealing gold to the detector.

### Dataset Cartography (Swayamdipta et al., EMNLP 2020)

Uses per-example confidence and variability across epochs to separate
easy-to-learn, ambiguous, and hard-to-learn regions. Hard-to-learn examples
often include label errors. Important warning: ambiguity and label error are
not identical, so uncertainty alone should not trigger automatic relabeling.

### Area Under the Margin (Pleiss et al., NeurIPS 2020)

Uses margins accumulated during training to identify suspicious samples. It is
an alternative student-side risk score that reuses training dynamics rather
than requiring repeated LLM calls. More implementation work than simple
out-of-fold entropy; valuable as a baseline or later ablation.

### Annotation Error Detection survey (Klie et al., 2023)

Reviews model-based and data-based annotation-error detection. Reports strong
families including classification uncertainty, Confident Learning, data-map
confidence, ensembles, and label aggregation. Useful for organizing the related
work and selecting evaluation metrics.

## Filtering, Disagreement, and Reweighting

### Co-teaching / Co-teaching+ (Han et al., 2018; Yu et al., ICML 2019)

Two networks exchange small-loss samples; Co-teaching+ keeps updates focused on
examples where the peer models disagree. This is evidence that disagreement can
be useful under label corruption. It is not the same as LLM-teacher versus
compact-student disagreement, so cite it as conceptual noisy-label precedent,
not as the exact proposed method.

### Learning to Reweight Examples (Ren et al., ICML 2018)

Learns example weights using a small clean validation set. Strong reference for
confidence weighting, but it violates a strict zero-gold operational setting.
Use only as a clean-calibration baseline or explicitly change the claim to
small-gold refinement.

### DivideMix (Li et al., ICLR 2020)

Models the per-sample loss distribution, divides data into likely clean and
noisy groups, and uses two networks plus semi-supervised learning. Strong noisy-
label baseline, but high engineering/compute complexity and mostly validated in
vision. Not recommended as the primary one-year master's implementation.

## Recommended Thesis Positioning

### Main method

- Teacher instability: agreement across equivalent prompts/repeated decisions.
- Student risk: out-of-fold probability assigned to the supplied LLM label;
  optionally margin/entropy.
- Disagreement: LLM label versus confident out-of-fold student prediction.
- Combination: rank aggregation, avoiding gold-trained risk weights.
- Budget: refine top 5%, 10%, and 20% risk-ranked labels.

### Refinement arms

- Evidence-based relabel.
- Filter suspected labels.
- Monotonic confidence weighting.

### Essential controls

- Raw LLM labels.
- Random selection at the same refinement budget.
- LLM self-reported confidence only.
- Student uncertainty only.
- Global relabeling, matching the closest Steiner–Bizer approach.
- Human/benchmark-label training arm as an evaluation ceiling, never detector
  input.

### Metrics

- Label-error/disagreement detection: AUPRC, precision@budget,
  recall@budget, risk–coverage curve.
- Student: match F1 primary; precision, recall, macro F1, accuracy supporting.
- Cost: initial labeling, extra LLM reviews, GPU training, inference, and
  break-even.

## Google Scholar Queries

```text
"LLM-based data labeling" "entity matching" noisy labels
"uncertainty-aware" "entity matching"
"label refinement" "entity matching" LLM
"confidence calibration" "entity matching"
"LLM annotation" self-consistency repeated sampling
"prompt variation" LLM annotation reliability
"LLM-as-a-judge" label error detection ensemble
"teacher-student disagreement" "label noise"
"out-of-fold" "label error detection"
"confident learning" text classification noisy labels
"dataset cartography" label errors training dynamics
"area under the margin" mislabeled data
"selective relabeling" noisy labels budget
"sample reweighting" noisy labels transformer
"error propagation" teacher student noisy labels
```

For LLM work, filter to 2023–2026. For noisy-label foundations, do not apply a
recent-year filter. Use “Cited by” and “Related articles” from the four anchor
papers: Steiner–Bizer, Confident Learning, Dataset Cartography, and the EM
calibration paper.

## References

- Steiner and Bizer (2026): https://arxiv.org/html/2606.28823v1
- Kamsteeg et al. (2025): https://aclanthology.org/2025.uncertainlp-main.12/
- Peeters et al.: https://arxiv.org/abs/2310.11244
- Chen et al. (2024): https://mlanthology.org/neurips/2024/chen2024neurips-entity/
- Knowledge Distillation in Automated Annotation:
  https://aclanthology.org/anthology-files/pdf/nlpcss/2024.nlpcss-1.9.pdf
- Lee et al. (2025): https://aclanthology.org/2025.coling-main.710/
- Nahum et al. (2025): https://aclanthology.org/2025.emnlp-main.1360/
- Northcutt et al. (2021):
  https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/
- Swayamdipta et al. (2020): https://aclanthology.org/2020.emnlp-main.746/
- Pleiss et al. (2020):
  https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html
- Klie et al.: https://doi.org/10.1162/coli_a_00464
- Han et al. (2018):
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
- Yu et al. (2019): https://proceedings.mlr.press/v97/yu19b.html
- Ren et al. (2018): https://proceedings.mlr.press/v80/ren18a.html
- DivideMix: https://openreview.net/forum?id=HJgExaVtwr

## Unresolved Questions

- Whether LLM consistency should use stochastic repeats, prompt variants, or
  both.
- Whether risk aggregation remains fixed/rank-based or receives a small clean
  calibration arm.
- Whether gold disagreement is the primary large-scale target and a small
  blinded audit handles ambiguous/gold-error cases.
- Exact dataset 3 and compact models 2–3.
