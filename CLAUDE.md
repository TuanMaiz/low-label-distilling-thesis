# CLAUDE.md

Read `../AGENTS.md` and `AGENTS.md` before changing the experiment workflow.
`AGENTS.md` is the canonical repository guidance.

## Active Project

The project is migrating to a full-label Entity Resolution comparison:

`3 datasets × 3 compact cross-encoder models × {gold, LLM hard labels}`

Direct LLM matching is a per-dataset accuracy/cost baseline. Low-label budgets,
active selection, rationale generation or distillation, adaptive cascades, and
repeated-run experiments are outside the frozen scope unless the supervisor
requires them.

Active plan:
`plans/260820-1507-full-label-er-migration/plan.md`

WDC labeler screening:
`labeller-screening/plan.md`

The screening workflow reuses one seed-42 random sample of 300 blinded WDC
training pairs for `sol_high`, `sol_max`, and `sol_pro_max` through OpenRouter,
pinned to the OpenAI upstream provider with fallbacks disabled. Gold labels
remain in a separate comparison-only CSV. Preparation and dry runs are
implemented, and all three paid screening settings are complete. Sol-high was
selected. The frozen WDC-only vertical-slice contract authorized reusing its
300 completed labels and calling the remaining 2,200 WDC training rows under a
USD 5 cumulative ceiling with the dedicated confirmation flag. That run is
complete: 2,500/2,500 valid labels, 300 reused plus 2,200 new, zero invalid
results or retries, and USD 2.693225 cumulative cost. The published result is
`data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv`.

The offline WDC target publication is also complete at
`data/cache/wdc_products/full_label_targets/`: 2,500 `gold` rows and 2,500
`llm_hard` rows, with 79 disagreements (0.9684 agreement) and USD 2.693225
labeler cost. `supervision/build_full_label_targets.py` builds the bundle, and
`supervision/validate_full_label_targets.py` independently rederives it from
the recorded upstream evidence. Publication made no paid calls, started no
training, and made no validation/test predictions. Phase 3 remains in progress
until the other two dataset target pairs are published.

The narrow WDC–Qwen training contract now authorizes only rented-RTX-3090
setup, preflight, and a tiny balanced LoRA smoke. The executable entry point is
`scripts/run_wdc_qwen_vertical_slice.sh`. It binds the old Qwen configuration,
inputs, code, input-length audit, resolved precision, GPU, and package versions;
it makes no LLM calls and never reads the test split. Full gold-versus-LLM
training and official full-validation/test predictions are not yet authorized.
The old full-run Qwen hyperparameters remain frozen at their screened values;
zero warmup applies only to the one-step smoke and does not amend the full-run
`0.10` warmup ratio. Training preflight consumes the committed final targets
directly and does not invoke publication validation;
`supervision.validate_full_label_targets` remains the separate upstream
rederivation command for the publication environment.

Legacy writing plan (revise before thesis drafting):
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The exact datasets, models, LLM labeler, prompt, evaluation scope, artifact
schema, and cost ceiling remain Phase-1 decisions. Do not guess them from
historical configs or results.

## Commands

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s labeller-screening/tests -v
.venv/bin/python -m supervision.validate_full_label_targets \
  --target-dir data/cache/wdc_products/full_label_targets
```

On the rented RTX 3090 with CUDA-compatible PyTorch already installed:

```bash
bash scripts/run_wdc_qwen_vertical_slice.sh setup
bash scripts/run_wdc_qwen_vertical_slice.sh preflight
bash scripts/run_wdc_qwen_vertical_slice.sh smoke
```

Use the uv-managed environment. The only authorized production labeling path was
`labeller-screening/run_full_wdc.py` for the frozen WDC Sol-high vertical slice,
and its full training-label run is complete. Do not relabel WDC or run full
experiment cells; only the WDC–Qwen setup/preflight/smoke exception is active,
and the rest remain blocked by the broader Phase-1 contract.

For complete rules, scope guardrails, reusable files, and conventions, follow
`AGENTS.md`.
