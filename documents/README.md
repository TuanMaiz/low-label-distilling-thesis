# Research Papers For Cost-Aware LLM Distillation In ER

This folder contains the core papers referenced by the thesis pivot toward
cost-aware LLM supervision and compact-student distillation for entity
resolution. Some papers were collected during the older structured-rationale
phase and now serve as novelty-boundary or negative-history context.

| File | Paper | Why it matters |
|---|---|---|
| `wadhwa-2024-learning-from-natural-language-explanations-for-generalizable-entity-matching.pdf` | Wadhwa et al. 2024, Learning from Natural Language Explanations for Generalizable Entity Matching | Explanation-distillation boundary; useful for explaining why rationales are not the main claim. |
| `steiner-2024-fine-tuning-large-language-models-for-entity-matching.pdf` | Steiner, Peeters, and Bizer 2024, Fine-tuning Large Language Models for Entity Matching | LLM supervision and EM fine-tuning boundary. |
| `peeters-2023-entity-matching-using-large-language-models.pdf` | Peeters, Steiner, and Bizer 2023/2025, Entity Matching using Large Language Models | Direct LLM matching and prompting background. |
| `zeakis-2026-distiller-knowledge-distillation-in-entity-resolution-with-large-language-models.pdf` | Zeakis et al. 2026, DistillER: Knowledge Distillation in Entity Resolution with Large Language Models | Closest ER knowledge-distillation paper and the most important novelty boundary. |

Original sources are arXiv PDFs.

## Active-Selection Literature To Add

The active thesis extension needs a small related-work set on low-budget active
learning and data selection for Entity Matching. Add these as PDFs before
drafting Chapter 1 if available:

| Topic | Why it matters |
|---|---|
| active learning for deep Entity Resolution | novelty boundary for selecting scarce labels |
| risk or uncertainty sampling for Entity Matching | baseline strategy for `llm_active_uncertainty` |
| diversity or representative sampling for low-resource ER | baseline strategy for `llm_active_diversity` |
| LLM data selection or LLM-as-labeler under budget | connects active learning to the current teacher-label cost lens |
