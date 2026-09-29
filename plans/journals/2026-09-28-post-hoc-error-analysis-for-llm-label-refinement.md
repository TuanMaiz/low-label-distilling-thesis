---
title: Post-hoc error analysis for LLM label refinement
date: 2026-09-28
summary: Add gold-hidden detection evaluation, post-hoc teacher/detector error analysis, and random/oracle controls without exposing gold to refinement.
---

## What happened

The proposed uncertainty-aware label-refinement study was reviewed against the
findings of NoiseBench and the requirement that the operational method must work
without benchmark gold labels. The central distinction is between a gold-free
detection/refinement pipeline and post-hoc research evaluation with hidden gold.

The working research question is no longer required to assume that refinement
always helps. It asks when and how LLM-generated labels should be refined for
compact Entity Matching models, including the valid outcome that refinement is
unnecessary or not cost-effective for a dataset.

## Decision

Add a dedicated post-hoc error-analysis section to the experiment design. The
detector ranking must be produced and frozen using only teacher labels,
three-pass teacher consistency, student out-of-fold evidence, and Confident
Learning outputs. Benchmark training gold must not influence ranking,
selection, pruning, or evidence-check relabeling.

After the ranking, treatments, and student results are fixed, hidden training
gold may be joined for research evaluation. The analysis should cover three
layers:

1. **Teacher errors:** quantify `teacher match -> human non-match` and `teacher
   non-match -> human match`; examine missing or conflicting attributes,
   near-duplicate text, sparse records, and other dataset-specific corner cases;
   and stratify errors by teacher agreement (`3/3` versus `2/3`). Whether LLM
   errors have structure is a hypothesis to test, not an assumption.
2. **Detector errors:** distinguish true errors caught, true errors missed, and
   correct-but-flagged hard examples. Report which error types each signal
   detects or misses.
3. **Downstream impact:** determine whether detected errors materially affect
   the compact student and whether prune or relabel is more appropriate for each
   observed category. A detectable error is not necessarily a consequential
   training error.

Use quantitative analysis over all pairs where possible. WDC currently has only
79 gold/LLM disagreements, so all can be inspected manually. For datasets with
many errors, use a predeclared fixed sample, such as up to 100 true teacher
errors and 100 detector false alarms per dataset, rather than manually coding
every pair.

## Evaluation structure

Evaluate the study at two separate levels:

- **Detection quality:** compute Precision@k, Recall@k, Average Precision/AUPRC,
  and class-direction-specific results only after opening hidden gold. These
  metrics test whether the suspicious-pair ranking identifies actual teacher
  errors.
- **Downstream quality:** train on unrefined, pruned, and
  evidence-check-relabeled targets, then compare student F1 and supporting
  metrics. This tests whether acting on the ranking improves the final ER model.

Keep operational and analytical arms visibly separate:

- Gold-free operational arms: no refinement, fixed random selection,
  detector-based prune, and detector-based relabel.
- Evaluation-only controls: oracle prune and full human-label training. The
  existing gold arm is the upper bound for perfect relabeling; oracle prune
  remains distinct because it removes known teacher errors instead of restoring
  their labels.

A fixed random control with the same intervention size tests whether the
suspicious ranking adds value beyond simply removing or relabeling records.
Oracle prune estimates the remaining headroom under perfect error detection.
These gold-using controls are analytical upper bounds, not deployable methods.

## Implications

This addition strengthens the thesis without increasing the core number of
model architectures or datasets. It adds explanation for why refinement
succeeds or fails and guards against claiming label correction when an apparent
gain is actually caused by regularization, reduced dataset size, or removal of
legitimate hard examples.

The proposed method remains deployable without gold. Gold is opened only after
decisions are frozen, in the same role as an answer key used to evaluate a
completed prediction.

## Next steps

- Incorporate the post-hoc error-analysis subsection and the
  operational-versus-oracle distinction into the next advisor brief revision.
- Predeclare the error taxonomy and manual-sampling rule before inspecting
  DBLP-ACM and the third dataset results.
- Finalize how the three detector signals produce the suspicious ranking and
  how the treatment size is chosen.
- Do not implement paid relabeling or additional training cells until the
  revised scope is reviewed.

## Status

This journal records a research-design decision and writing ingredient. It does
not authorize paid labeling, relabeling, new full-training runs, or access to
locked test splits.
