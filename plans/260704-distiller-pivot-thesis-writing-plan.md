---
title: "DistillER-Style WDC Pivot Thesis Writing Plan"
description: "Writing plan aligned with the official master's thesis structure: front matter, Mo dau, Tong quan, methods, system, experiments, results, and conclusion."
status: pending
priority: P1
effort: "8-10 weeks alongside experiments"
created: 2026-07-04
tags: [thesis, writing, entity-resolution, llm-labeling, wdc]
blockedBy: []
blocks: []
---

# DistillER-Style WDC Pivot Thesis Writing Plan

> Superseded by the detailed ck plan:
> `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`.

## Writing Principle

Write the thesis like a paper, but format it like the school requires.

Order of work:

1. Run pilot experiments.
2. Write method and experiment setup.
3. Write results and discussion.
4. Write related work.
5. Write `Mo dau` and conclusion last.

Do not write a strong introduction before the experiment tells us what claim is safe.

## Official Structure To Follow

The official requirement uses this structure:

1. Front matter.
2. `Mo dau`.
3. `Chuong 1: Tong quan`.
4. Main research chapters.
5. Final chapter: conclusion and recommendations.
6. Publications, references, appendices.

This means the thesis should not use `Chuong 1: Gioi thieu` as the main structure. The introduction-like material belongs in `Mo dau`.

## Formatting Checklist

- Body font: Times New Roman 13.
- Chapter and section titles: Times New Roman 14 or 15.
- Paper: A4, one-sided.
- Line spacing: 1.5.
- Margins: top 2.5 cm, bottom 2.5 cm, left 3.5 cm, right 2 cm.
- Page number: centered at bottom.
- Header: chapter name, Times New Roman 11.
- Table title above table, format `Bang 3.2. ...`.
- Figure title below figure, format `Hinh 4.5. ...`.
- Equation numbering by chapter, e.g. `(3.5)`.
- References in square brackets, e.g. `[5]`, `[9-12]`.
- Reference list grouped by language and sorted according to school rules.

## Front Matter

### Content

- Cover and sub-cover.
- Declaration.
- Table of contents.
- List of abbreviations.
- List of tables.
- List of figures.
- Research-result summary if required by department template.

### Draft Late

Most front matter should be finalized after chapters are stable.

## Mo Dau

### Purpose

Explain why the topic matters and what the thesis studies.

### Sections

1. Reason for choosing the topic.
2. Research purpose.
3. Research object and scope.
4. Scientific meaning.
5. Practical meaning.
6. Thesis structure.

### Write Last

Draft a rough version early if needed, but final `Mo dau` should be written after Chapter 5 results are known.

### Safe Claim Template

This thesis studies whether LLM-generated labels can reduce human labeling effort in Entity Matching by training compact student models on WDC Products, while comparing performance and cost against gold-label supervision.

## Chuong 1: Tong Quan

### Purpose

Show that the topic is grounded in existing work and identify the gap this thesis can safely claim.

### Suggested Sections

1. Entity Resolution and Entity Matching.
2. Product matching and WDC Products.
3. Transformer-based models for Entity Matching.
4. Large Language Models for Entity Matching.
5. Knowledge distillation and LLM-generated supervision.
6. Related work: DistillER and LLM-labeled ER training data.
7. Remaining gap and thesis direction.

### Key Positioning

The thesis should say:

- DistillER and related work already study LLM supervision for ER.
- This thesis does not claim the general idea is new.
- The contribution is a controlled WDC-focused study of label efficiency, cost, compact students, and failure behavior.

## Chuong 2: Co So Ly Thuyet Va Phuong Phap Nghien Cuu

### Purpose

Define the technical concepts and research method.

### Suggested Sections

1. Entity Matching problem formulation.
2. Pair serialization.
3. Label-supervised student training.
4. LLM-generated label supervision.
5. Knowledge distillation framing.
6. Evaluation metrics: precision, recall, F1, accuracy.
7. Cost metrics: teacher labeling cost and student inference cost.
8. Research methodology and experimental controls.

### Write After Pilot Contract

This chapter can be drafted before full results, but only after the experiment matrix is frozen.

## Chuong 3: Xay Dung He Thong

### Purpose

Describe the implemented pipeline.

### Suggested Sections

1. System overview.
2. Dataset preparation pipeline.
3. Low-label budget sampler.
4. LLM label-generation module.
5. Label validation and cache.
6. Training-target builder.
7. Student training module.
8. Evaluation and aggregation module.
9. Cost logging module.

### Figures To Prepare

- Overall pipeline figure.
- Data flow from raw WDC pair to teacher label to student prediction.

### Write During Implementation

This chapter should be written while building the system, because file paths, commands, and artifacts are fresh.

## Chuong 4: Thuc Nghiem

### Purpose

Make the experiment reproducible.

### Suggested Sections

1. Dataset description.
2. Train/validation/test split.
3. Label budgets.
4. Teacher LLM setting.
5. Student model setting.
6. Baselines.
7. Hyperparameters.
8. Hardware and software environment.
9. Evaluation protocol.

### Tables To Prepare

- Dataset statistics.
- Experiment matrix.
- Hyperparameter table.
- Cost assumptions.

### Write Before Final Results

This chapter should be mostly written before Chapter 5, because it describes what was done, not what happened.

## Chuong 5: Ket Qua Va Ban Luan

### Purpose

This is the core proof chapter.

### Suggested Sections

1. Pilot results.
2. Full label-budget results.
3. Gold labels vs LLM-generated labels.
4. Optional mixed-supervision results.
5. Cost-efficiency analysis.
6. Failure analysis.
7. Comparison with related work.
8. Discussion of limitations.

### Required Evidence

- Main metrics table.
- Label-efficiency plot.
- Cost table.
- Failure-analysis table.
- Short comparison with DistillER-style results and prior LLM-labeling work.

### Write First Among Final Chapters

Once experiments finish, write this before polishing the introduction.

## Chuong 6: Ket Luan Va Khuyen Nghi

### Purpose

State what the thesis actually found.

### Sections

1. Summary of completed work.
2. Main findings.
3. Scientific contribution.
4. Practical contribution.
5. Limitations.
6. Future work.

### Write Last

This chapter should mirror Chapter 5 results. Do not overclaim.

## References

### Required Groups

- Vietnamese references.
- English references.
- Other languages if used.

### Must Include

- Entity Resolution / Entity Matching foundations.
- WDC Products or WDC benchmark references.
- Transformer or BERT-style ER references.
- LLMs for Entity Matching.
- DistillER.
- LLM-generated labels for ER / EM.
- Knowledge distillation references.

## Appendices

### Good Appendix Material

- Prompt templates.
- Extra hyperparameters.
- Extra result tables.
- Sample teacher-label outputs.
- Additional failure examples.

### Constraint

Appendices should support the thesis but not become longer than the main content.

## Writing Timeline

| Stage | Writing Output |
|---|---|
| Before pilot | Rough Chapter 3 skeleton and Chapter 4 experiment matrix |
| After pilot | Chapter 4 full setup and first Chapter 5 pilot table |
| After full runs | Chapter 5 results and discussion |
| After results are stable | Chapter 1 related work and Chapter 2 theory cleanup |
| Final month | Mo dau, conclusion, formatting, references, appendices |

## Immediate Next Writing Task

Create a one-page experiment contract. It should become the seed for:

- Chapter 2 research method.
- Chapter 4 experiment setup.
- Chapter 5 result table design.
