# AGENTS.md

Read `../AGENTS.md` and this file before changing the experiment workflow.

## Active Project

**Master's thesis:** full-label LLM-to-student distillation for Entity
Resolution across three benchmarks and three cross-encoder students.

Research question:

> Can compact cross-encoder ER students trained on complete LLM-generated hard
> labels provide a practical alternative to the same students trained on
> benchmark gold labels?

Active execution plan:
`plans/260820-1507-full-label-er-migration/plan.md`

Companion writing plan:
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The experiment contract is not frozen yet. Phase 1 of the active plan selects
the exact two non-WDC datasets, three cross-encoder models, teacher, prompt,
evaluation scope, artifact schema, and cost ceiling.

## Scope Guardrail

The fixed high-level design is:

- 3 benchmark datasets.
- 3 cross-encoder students.
- 2 training-label sources: benchmark gold and LLM-generated hard labels.
- 1 fixed reproducible seed/run per cell; no multi-seed experiment dimension.
- 3 direct LLM baselines, one per dataset.
- Match F1 primary; precision, recall, macro F1, accuracy, timing, throughput,
  cost, and break-even supporting.

Out of scope: low-label budgets, active selection, rationale distillation,
adaptive bi-/cross-encoder cascades, and extra datasets/models unless the
supervisor explicitly requires a change. If the researcher worries that the
work is not enough, remind them that this design is already thesis-grade and
favor completing the frozen plan.

## Current Status

- Branch: `refactor/full-label-er-migration`.
- Low-label sampler, active selector, old Phase-03/04 orchestration, old
  Phase-05 runner, and superseded execution-plan directory are removed.
- Historical experiment results and journals remain as research evidence.
- No active production experiment runner exists during migration.
- Start with Phase 1; do not make paid teacher calls or final-test predictions
  before the required contract and approval gates exist.

## Commands

Use the uv-managed environment; never use bare system `python`, `pip`, or
`pytest` for repository work.

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
```

## Planned Architecture

```text
frozen contract
  -> dataset registry and three verified loaders
  -> dataset-namespaced serialized splits
  -> complete gold and LLM-hard-label training targets
  -> three frozen cross-encoder students
  -> 18 train/evaluate cells plus 3 direct LLM baselines
  -> provenance-checked metrics and cost aggregation
  -> thesis tables and figures
```

Gold validation/test labels are evaluation-only. Teacher generation operates on
training pairs only and must publish targets only at 100% valid unique coverage.
Student inference never calls the teacher.

## Reusable Code

- Data: `data/schema.py`, `data/er_dataset_loader.py`,
  `data/serialize_pairs.py`.
- Supervision: `supervision/generate_teacher_labels.py`,
  `supervision/direct_llm_matcher.py`, `supervision/build_targets.py`,
  `supervision/validate_teacher_labels.py`.
- Students: `configs/students/`, `models/student_config.py`,
  `models/classification_student.py`, `models/generative_reranker_student.py`,
  `experiments/train_student.py`, `experiments/evaluate_student.py`.
- Reproducibility: `utils/artifact_contract.py`,
  `utils/runtime_provenance.py`, `utils/checkpoint_manifest.py`,
  `utils/torch_runtime.py`.
- Metrics/cost: `utils/metrics.py`, `utils/cost_accounting.py`,
  `analysis/cost_summary.py`.

## Conventions

- Python files: `snake_case.py`.
- Schemas: Pydantic `BaseModel` with explicit fields.
- Preserve official dataset splits and namespace pair/cache identities by
  dataset and version.
- Freeze manifests, prompts, model revisions, and evaluation IDs before results.
- Store API keys only in environment variables; never commit secrets.
- Preserve unrelated dirty-worktree changes.
- After meaningful workflow changes, update `../AGENTS.md`, this file, and
  `CLAUDE.md` together.
