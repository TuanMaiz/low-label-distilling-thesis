---
title: Confident Learning roles in uncertainty-aware label refinement
date: 2026-09-16
summary: Count characterizes dataset-level label noise, Rank prioritizes per-pair risk, and filter/reweight/relabel are separate treatments.
---

## What happened

Reviewed the Confident Learning framework as a candidate component for the
advisor-proposed uncertainty-aware LLM label-refinement pipeline. The useful
separation is `Count -> Rank -> treatment`, rather than treating all three as
one undifferentiated uncertainty method.

## Working interpretation

- **Count** estimates label-noise structure at dataset/class level: estimated
  error volume, error direction (`match -> non-match` versus the reverse), and
  the confident joint between observed LLM labels and latent labels. It can
  flag some confident off-diagonal candidates, but it is not by itself the
  final per-pair uncertainty score.
- **Rank** assigns per-pair risk and prioritizes review. The proposed thesis
  ranking can combine student out-of-fold self-confidence, teacher-student
  disagreement, and selective LLM-consistency evidence.
- **Prune/filter** is a treatment after detection, not the detector itself. It
  should be compared with per-example reweighting and selective LLM relabeling.
- For normalized binary probabilities, Prune-by-Class and Prune-by-Noise-Rate
  induce essentially the same within-class ranking: low confidence in the
  assigned label is monotonic with a large opposing-class margin. They do not
  need to become separate main experimental dimensions without evidence of a
  meaningful implementation difference.

## Experimental implication

Use Confident Learning as a principled baseline and dataset-level diagnostic,
while using Rank to prioritize individual pairs for refinement. Count should
not be treated as the final per-pair uncertainty score. Compare filter,
per-example reweighting, and selective LLM relabeling as separate treatments.
Benchmark gold labels remain hidden from detection and refinement and are
joined only for final error-detection and downstream evaluation.

## Status and next steps

This is a research interpretation and writing ingredient, not yet a frozen
experiment-contract change. Continue the literature review, define the exact
risk aggregation and treatment arms, then review the revised scope before
implementation or paid relabeling.

## Scope refinement — 2026-09-18

The researcher narrowed and clarified the teacher-consistency protocol:

- Use one teacher model only: GPT-5.6 Sol-high.
- Remove prompt variants from the main experiment.
- Run three independent labeling passes over the full training set with the
  same prompt to measure teacher self-consistency.
- Keep one separate evidence-check prompt, called only on suspicious labels for
  the relabel treatment. It is not one of the three consistency passes.
- If a pilot shows that Sol-high is too deterministic for repeated calls to
  yield a useful consistency signal, replace the second and third exact-repeat
  passes with two semantically equivalent prompt variants. Do not add those
  variants on top of the three passes.
- Retain only prune and relabel as label treatments; per-example reweighting is
  removed from the proposed experiment scope to reduce final training runs.

This section supersedes the earlier working assumption that LLM-consistency
evidence would necessarily be selective and that reweighting would remain a
treatment arm. The experiment contract and advisor brief still need to be
revised after the remaining scope decisions are closed.
