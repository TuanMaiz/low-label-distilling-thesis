---
title: Gemma Student Readiness
status: in_progress
created: 2026-07-16
plan: ../plan.md
phase: 5
---

# Gemma Student Readiness

## Summary

Config-driven compact-student execution is ready for the predeclared Gemma 3
270M validation diagnostic. Existing FLAN-T5 evidence remains unchanged. Test
is untouched.

## Completed

| Item | Evidence |
|---|---|
| Student configuration boundary | `configs/students/flan_t5_base.json`, `configs/students/gemma_3_270m.json` |
| Two explicit architectures | `seq2seq`, `sequence_classification` |
| Generic training | `experiments/train_student.py` |
| Classifier evaluation | textual labels, both probabilities, zero malformed outputs |
| Canonical output ownership | `outputs/students/<student_id>/train_<budget>/` |
| Provenance | config snapshot plus runtime/training/evaluation contracts |
| Aggregation integrity | training and evaluation identities must match |
| Verification | 57 unit tests pass; shell syntax and diff checks pass |
| Review | final independent review has no actionable findings |

## Remaining

1. Commit and push the implementation.
2. Run the Gemma config on Colab.
3. Return `phase05_gemma-3-270m_train_128_results.tar.gz`.
4. Compare Gemma active/random/gold results and update the Phase 5 decision.

## Unresolved Questions

- Whether Gemma removes the FLAN-T5 architecture bottleneck at budget 128.
- Whether active selection improves primary match F1 across seeds after the
  second-student diagnostic.
