# CLAUDE.md

Read `../AGENTS.md` and `AGENTS.md` before changing the experiment workflow.
`AGENTS.md` is the canonical repository guidance.

## Active Project

The project is migrating to a full-label Entity Resolution comparison:

`3 datasets × 3 cross-encoder students × {gold, LLM hard labels}`

Direct LLM matching is a per-dataset accuracy/cost baseline. Low-label budgets,
active selection, rationale distillation, adaptive cascades, and multi-seed
experiments are outside the frozen scope unless the supervisor requires them.

Active plan:
`plans/260820-1507-full-label-er-migration/plan.md`

Writing plan:
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The exact datasets, models, teacher, prompt, evaluation scope, artifact schema,
and cost ceiling remain Phase-1 decisions. Do not guess them from historical
configs or results.

## Commands

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
```

Use the uv-managed environment. There is no active production experiment runner
until the new plan introduces `scripts/run_full_label_experiments.sh`.

For complete rules, scope guardrails, reusable files, and conventions, follow
`AGENTS.md`.
