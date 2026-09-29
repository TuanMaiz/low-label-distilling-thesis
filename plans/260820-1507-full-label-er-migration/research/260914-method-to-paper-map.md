---
title: Paper map for uncertainty-aware LLM label refinement methods
date: 2026-09-14
status: research-note
---

# Method-to-Paper Map

This note narrows the broader literature review in
`260910-uncertainty-aware-label-refinement-literature.md` to the exact proposed
pipeline. No single located paper combines all of these components for tabular
entity matching. The closest entity-matching work is Steiner and Bizer (2026);
the remaining components have direct precedents in automated annotation,
annotation-error detection, uncertainty estimation, and noisy-label learning.

## Exact pipeline and supporting papers

| Proposed component | Best supporting paper(s) | Strength of relation | Practical takeaway |
|---|---|---|---|
| LLM labels used to train a compact matcher | Steiner and Bizer (2026); Pangakis and Wolken (2024) | Direct | Establishes the teacher-label/student-train workflow. |
| Repeat the LLM and measure label consistency | Pangakis and Wolken (2024) | Direct outside ER | Three calls per item; modal-label agreement predicts annotation accuracy, but dropping inconsistent examples did not improve downstream F1. |
| Evidence-based LLM re-check | Steiner and Bizer (2026) | Direct in ER | A conservative second prompt relabels or drops unstable pairs; useful as the global-relabel baseline. |
| Student out-of-fold probabilities | Northcutt et al. (2021); Klie et al. (2023) | Direct for label-error detection | Score every training item using a model that did not train on that item. Klie et al. explicitly evaluate cross-validated annotation-error detection. |
| Student entropy/margin | Klie et al. (2023); Pleiss et al. (2020); Swayamdipta et al. (2020) | Direct for suspicious-label ranking | Compare simple classification uncertainty with AUM and training-dynamics alternatives. Do not equate ambiguity with wrong labels. |
| Optional MC Dropout | Gal and Ghahramani (2016); Gal et al. (2017); Kamsteeg et al. (2025) | Foundational; direct EM application in Kamsteeg | Repeated dropout-enabled predictions provide model dispersion. Treat as an ablation, not the core requirement. |
| Teacher–student disagreement | Yu et al. (2019); Zheng et al. (2020) | Conceptual noisy-label precedent | Disagreement or a noisy classifier's opposing prediction can reveal suspicious labels, but neither paper studies LLM-teacher/ER-student disagreement exactly. |
| Select top-k% under a fixed refinement budget | Bernhardt et al. (2022); Kremer et al. (2018) | Direct budgeted label-cleaning precedent | Rank suspected errors and spend a limited re-annotation budget; compare against random selection at the same budget. |
| Filter suspicious labels | Northcutt et al. (2021); Pleiss et al. (2020); Steiner and Bizer (2026) | Direct | Remove only the risk-ranked subset and compare with random removal of equal size. |
| Reweight suspicious labels | Ren et al. (2018) | Direct mechanism, incompatible assumption | Their learned weights require a small clean validation set. For zero-gold operation, use a predeclared monotonic weight from the risk rank instead of claiming to reproduce their method. |
| Relabel suspicious labels | Steiner and Bizer (2026); Bernhardt et al. (2022); Kremer et al. (2018) | Direct mechanism | Re-query only selected cases, rather than globally reviewing every label. |

## Recommended reading order

1. **Steiner and Bizer (2026)** — closest baseline and exact ER setting.
2. **Pangakis and Wolken (2024)** — direct evidence for repeated LLM labeling
   and consistency scores.
3. **Klie, Webber, and Gurevych (2023)** — map of 18 annotation-error detection
   methods and their evaluation.
4. **Northcutt, Jiang, and Chuang (2021)** — the cleanest foundation for
   out-of-fold label-risk ranking.
5. **Bernhardt et al. (2022)** — strongest justification for top-k review under
   a fixed correction budget and a random-budget baseline.
6. **Kamsteeg et al. (2025)** — uncertainty methods already tested in entity
   matching.
7. **Yu et al. (2019)** — conceptual foundation for exploiting disagreement
   under label corruption.
8. **Swayamdipta et al. (2020)** and **Pleiss et al. (2020)** — alternative
   student-side risk signals for later ablations.

## What is and is not supported

- Supported directly: repeated LLM decisions can expose unstable annotations;
  out-of-sample model probabilities are appropriate for label-error detection;
  limited-budget relabeling should be evaluated against random selection.
- Supported indirectly: disagreement helps with noisy labels. Existing papers
  usually compare two peer networks or a noisy label with a classifier, not an
  LLM teacher with a compact entity matcher.
- Not established by the literature found: that a weighted sum of LLM
  consistency, student uncertainty, and teacher–student disagreement is
  universally optimal. That combination must be presented as the proposed
  method and evaluated through ablation.
- Important negative evidence: Pangakis and Wolken find that consistency is a
  useful error signal but filtering inconsistent labels does not improve their
  downstream models; Steiner and Bizer find that post-processing gains vary by
  dataset and that self-reported LLM confidence is too noisy to use as a
  filter. These results motivate comparing filter, reweight, and selective
  relabel rather than assuming one will win.

## Primary sources

1. Steiner, A. and Bizer, C. (2026). *Labeling Training Data for Entity
   Matching Using Large Language Models*. https://arxiv.org/html/2606.28823v1
2. Pangakis, N. and Wolken, S. (2024). *Knowledge Distillation in Automated
   Annotation*. https://aclanthology.org/2024.nlpcss-1.9/
3. Klie, J.-C., Webber, B., and Gurevych, I. (2023). *Annotation Error
   Detection*. https://aclanthology.org/2023.cl-1.4/
4. Northcutt, C., Jiang, L., and Chuang, I. (2021). *Confident Learning*.
   https://research.google/pubs/confident-learning-estimating-uncertainty-in-dataset-labels/
5. Bernhardt, M. et al. (2022). *Active label cleaning for improved dataset
   quality under resource constraints*.
   https://www.nature.com/articles/s41467-022-28818-3
6. Kamsteeg, I. et al. (2025). *Confidence Calibration in Large Language
   Model-Based Entity Matching*.
   https://aclanthology.org/2025.uncertainlp-main.12/
7. Yu, X. et al. (2019). *How does Disagreement Help Generalization against
   Label Corruption?* https://proceedings.mlr.press/v97/yu19b.html
8. Zheng, S. et al. (2020). *Error-Bounded Correction of Noisy Labels*.
   https://proceedings.mlr.press/v119/zheng20c.html
9. Swayamdipta, S. et al. (2020). *Dataset Cartography*.
   https://aclanthology.org/2020.emnlp-main.746/
10. Pleiss, G. et al. (2020). *Identifying Mislabeled Data using the Area Under
    the Margin Ranking*.
    https://proceedings.neurips.cc/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html
11. Gal, Y. and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*.
    https://proceedings.mlr.press/v48/gal16.html
12. Gal, Y., Islam, R., and Ghahramani, Z. (2017). *Deep Bayesian Active
    Learning with Image Data*. https://proceedings.mlr.press/v70/gal17a.html
13. Kremer, J., Sha, F., and Igel, C. (2018). *Robust Active Label Correction*.
    https://proceedings.mlr.press/v84/kremer18a.html
14. Ren, M. et al. (2018). *Learning to Reweight Examples for Robust Deep
    Learning*. https://proceedings.mlr.press/v80/ren18a.html

## Google Scholar queries

```text
"LLM annotation" consistency repeated sampling downstream classifier
"entity matching" LLM label post-processing relabel
"out-of-sample predicted probabilities" label error detection
"teacher student disagreement" noisy labels
"active label cleaning" fixed relabeling budget
"selective relabeling" noisy labels budget
"confidence calibration" entity matching MC dropout
"annotation error detection" text classification cross validation
```
