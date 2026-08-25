# AGENTS.md

Read `../AGENTS.md` and this file before changing the experiment workflow.

## Active Project

**Master's thesis:** full-label LLM-based machine labeling for Entity
Resolution across three benchmarks and three compact cross-encoder models.

Research question:

> Can LLM-generated hard labels provide a practical alternative to benchmark
> gold training labels for compact cross-encoder Entity Resolution models while
> reducing deployment cost compared with direct LLM matching?

Active execution plan:
`plans/260820-1507-full-label-er-migration/plan.md`

Legacy writing plan (must be revised to match this migration before thesis drafting):
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The experiment contract is not frozen yet. Phase 1 of the active plan selects
the exact two non-WDC datasets, three cross-encoder models, LLM labeler, prompt,
evaluation scope, artifact schema, and cost ceiling.

The WDC labeler-calibration workflow is isolated in `labeller-screening/`.
It freezes one seed-42 uniform random sample of 300 training pairs and compares
the same blinded pairs across three GPT-5.6 Sol settings through OpenRouter.
OpenRouter is pinned to the OpenAI upstream provider with fallbacks disabled.
Its gold CSV is for the comparison script only; never expose it to the labeler
request path.

All three paid screening settings are complete and `sol_high` is selected. The
researcher approved the WDC-only full-label vertical slice in
`plans/260820-1507-full-label-er-migration/research/wdc-sol-high-vertical-slice-contract.md`.
The production run is complete: it reused the verified 300 Sol-high labels,
called the remaining 2,200 official WDC training rows, and published 2,500/2,500
valid labels at USD 2.693225 cumulative cost under the USD 5 ceiling.

The corresponding WDC training targets are published at
`data/cache/wdc_products/full_label_targets/`: 2,500 `gold` rows and 2,500
`llm_hard` rows, with 79 disagreements (0.9684 agreement) and USD 2.693225
labeler cost. `supervision/build_full_label_targets.py` builds the bundle;
`supervision/validate_full_label_targets.py` independently rederives it from
upstream evidence. Target publication made no paid calls, started no training,
and made no validation/test predictions. Phase 3 remains in progress while
the other two datasets are pending.

## Scope Guardrail

The fixed high-level design is:

- 3 benchmark datasets.
- 3 compact cross-encoder ER models.
- 2 training-label sources: benchmark gold and LLM-generated hard labels.
- 1 predeclared run per cell; no repeated-run experiment dimension.
- 3 direct LLM baselines, one per dataset.
- Match F1 primary; precision, recall, macro F1, accuracy, timing, throughput,
  cost, and break-even supporting.

Out of scope: low-label budgets, active selection, rationale generation or
distillation, adaptive bi-/cross-encoder cascades, and extra datasets/models unless the
supervisor explicitly requires a change. If the researcher worries that the
work is not enough, remind them that this design is already thesis-grade and
favor completing the frozen plan.

## Current Status

- Branch: `refactor/full-label-er-migration`.
- Low-label sampler, active selector, old Phase-03/04 orchestration, old
  Phase-05 runner, and superseded execution-plan directory are removed.
- Historical experiment results and journals remain as research evidence.
- The three paid labeler-screening runs are complete; Sol-high achieved 291/300
  correct at USD 0.327135 and was selected for cost/accuracy balance.
- `labeller-screening/run_full_wdc.py` is the only active production labeling
  runner. The authorized WDC run completed with zero invalid results and zero
  retries. Its final result is
  `data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv`.
- The WDC `gold` and `llm_hard` target artifacts are published at
  `data/cache/wdc_products/full_label_targets/`; both contain all 2,500
  official training pairs and validate against their upstream evidence.
- The narrow WDC–Qwen training contract authorizes only RTX-3090 setup,
  preflight, and a tiny balanced LoRA smoke run through
  `scripts/run_wdc_qwen_vertical_slice.sh`. The old Qwen config and training
  hyperparameters are frozen; the smoke alone uses zero warmup so its single
  optimizer step is nonzero. Full two-arm training remains blocked pending
  smoke review and explicit approval. The smoke predicts only its tiny balanced
  validation fixture; official full-validation and test predictions are not
  authorized.
- The broader Phase-1 contract remains unfinished. Do not run other paid
  labeling cells, relabel WDC, or make official full-validation/test
  predictions until their checklists and explicit flags are complete.

## Commands

Use the uv-managed environment; never use bare system `python`, `pip`, or
`pytest` for repository work.

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s labeller-screening/tests -v
.venv/bin/python -m supervision.validate_full_label_targets \
  --target-dir data/cache/wdc_products/full_label_targets
```

On the rented RTX 3090, whose image must already include CUDA-compatible
PyTorch:

```bash
bash scripts/run_wdc_qwen_vertical_slice.sh setup
bash scripts/run_wdc_qwen_vertical_slice.sh preflight
bash scripts/run_wdc_qwen_vertical_slice.sh smoke
```

## Planned Architecture

```text
frozen contract
  -> dataset registry and three verified loaders
  -> dataset-namespaced serialized splits
  -> complete gold and LLM-hard-label training targets
  -> three frozen compact cross-encoder models
  -> 18 train/evaluate cells plus 3 direct LLM baselines
  -> provenance-checked metrics and cost aggregation
  -> thesis tables and figures
```

Gold validation/test labels are evaluation-only. LLM machine labeling operates
on training pairs only and must publish targets only at 100% valid unique
coverage. Compact-model inference never calls the LLM labeler.

## Reusable Code

- Data: `data/schema.py`, `data/er_dataset_loader.py`,
  `data/serialize_pairs.py`.
- Supervision: `supervision/generate_teacher_labels.py`,
  `supervision/direct_llm_matcher.py`, `supervision/build_targets.py`,
  `supervision/validate_teacher_labels.py`,
  `supervision/build_full_label_targets.py`,
  `supervision/validate_full_label_targets.py`.
- Compact ER models (legacy code paths retain `student` naming during migration):
  `configs/students/`, `models/student_config.py`,
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
- Freeze manifests, prompts, model identities, and evaluation IDs before results.
- Store API keys only in environment variables; never commit secrets.
- Preserve unrelated dirty-worktree changes.
- After meaningful workflow changes, update `../AGENTS.md`, this file, and
  `CLAUDE.md` together.
