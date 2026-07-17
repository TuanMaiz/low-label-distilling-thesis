---
title: ModernBERT Student Readiness
status: in_progress
created: 2026-07-17
plan: ../plan.md
phase: 5
---

# ModernBERT Student Readiness

## Summary

Config-driven compact-student execution is ready for the predeclared public,
ungated ModernBERT-base validation diagnostic. It replaces the unrun gated
Gemma choice before any second-student or test result was inspected. Existing
FLAN-T5 evidence remains unchanged, and test is untouched.

## Completed

| Item | Evidence |
|---|---|
| Student configuration boundary | `configs/students/flan_t5_base.json`, `configs/students/modernbert_base.json` |
| Two explicit architectures | `seq2seq`, `sequence_classification` |
| Generic training | `experiments/train_student.py` |
| Classifier evaluation | textual labels, both probabilities, zero malformed outputs |
| Canonical output ownership | `outputs/students/<student_id>/train_<budget>/` |
| Provenance | config snapshot plus runtime/training/evaluation contracts |
| Aggregation integrity | training and evaluation identities must match |
| Access | public ungated weights; no `HF_TOKEN` or approval required |
| Verification | 58 repository tests pass; config, shell syntax, and diff checks pass |
| Review | final independent review has no actionable findings |

## Remaining

1. Commit and push the model substitution.
2. Run the ModernBERT config on Colab.
3. Return `phase05_modernbert-base_train_128_results.tar.gz`.
4. Compare ModernBERT active/random/gold results and update the Phase 5 decision.

## Unresolved Questions

- Whether ModernBERT removes the FLAN-T5 architecture bottleneck at budget 128.
- Whether active selection improves primary match F1 across seeds after the
  second-student diagnostic.
