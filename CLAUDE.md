# CLAUDE.md

This file provides guidance for Codex/Claude when working in this repository.
Workspace-level agent guidance also lives in `../AGENTS.md`; read both files
when starting a fresh conversation or changing workflow conventions.

## Active Project Overview

**Master's thesis:** Cost-aware active LLM labeling for Entity Resolution /
Entity Matching, centered on WDC Products.

**Current research question:**
> Under low-label budgets on WDC Products, can active selection of
> LLM-labeled training pairs produce compact ER students that outperform random
> LLM-label distillation at the same labeling cost, while becoming cheaper than
> repeated direct LLM matching?

**Active execution plan:**
`plans/260704-distiller-wdc-agent-execution/plan.md`

**Experiment contract:**
`plans/260704-distiller-wdc-agent-execution/research/experiment-contract.md`

**Companion writing plan:**
`plans/260704-distiller-wdc-thesis-writing/plan.md`

## Current Status

The project has pivoted away from the structured-rationale thesis. The old
Phase 03 result is preserved as negative evidence:

| Variant | Train rows | Match precision | Match recall | Match F1 | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `label_only` | 128 | 0.3887 | 0.5900 | 0.4686 | 0.6449 | 0.7324 |
| `structured_rationale` | 122 | 0.2487 | 0.6780 | 0.3639 | 0.4931 | 0.5260 |

Interpretation: structured rationales increased recall but badly hurt
precision, match F1, macro F1, and accuracy. Treat rationale distillation as a
historical/optional negative-history ablation, not the active thesis claim.

The current thesis direction is label-level LLM-to-student distillation with
active data selection under scarce labeling budgets:

- `gold_random_student`: compact student trained on randomly sampled dataset
  labels; quality context, not the cost baseline.
- `direct_llm_matcher`: LLM classifies fixed evaluation pairs directly; repeated
  inference-cost baseline.
- `llm_random_student`: random training pairs are LLM-labeled once, then a
  compact student performs cheap inference; random distillation control.
- `llm_active_student`: actively selected training pairs are LLM-labeled once,
  then distilled into a compact student; main proposed method.
- `mixed_gold_llm_active`: optional fallback if pure active LLM labels are noisy.

The thesis lens is not "we invented LLM-label distillation or active learning
for ER." The safer claim is a cost-aware, low-label, data-selection, and
failure-slice study: which pairs are worth spending LLM teacher calls on, where
selection fails, how much it costs, and whether patterns replicate on one
optional later dataset.

As of 2026-07-13, Phase 3 answer-only LLM pipeline code, fixed `train_128`
selection manifests, live OpenRouter `openai/gpt-5.4-mini` teacher-label
caches, the full validation direct-LLM baseline, and Phase 4 128-budget student
target files exist. The Phase 5 change set prepares the fresh-clone inputs,
Colab dependency file, resumable runner, result aggregator, and runbook needed
to execute the fixed pilot on a GPU after it is committed and pushed. Phase 5
itself remains pending until the Colab validation artifacts are returned and
reviewed.

The Phase 5 Colab defaults are optimized for binary answer-only training
without changing the fixed experiment inputs: target and generation limits are
8 tokens; inputs remain capped and padded at 512 tokens; train, validation, and
prediction rows are tokenized once per process; validation loss is weighted by
non-padding label tokens; and automatic runtime selection uses BF16 plus a
validation batch of 32 on BF16-capable CUDA hardware, FP16 plus 16 on other
CUDA hardware, and FP32 plus the training batch size for CPU smoke checks.
Training batch size remains 4.

Phase 5 cost reporting preserves synchronized training and inference seconds as
the provider-independent primary evidence. Aggregation applies every
predeclared low/base/high GPU-hour scenario in
`configs/phase05_cost_assumptions.json`, records the assumptions SHA-256, and
reports training cost, per-pair student inference cost, comparison-scale
savings, and break-even query count. These rates are analytical sensitivity
assumptions, not observed Colab charges or current provider quotes.

## Build & Run Commands

Always use the project uv-managed virtual environment for Python commands.
Do not run bare system `python`, `pip`, or `pytest` from this repository.
Prefer `.venv/bin/python -m ...` and `uv pip ...` so commands use the pinned
project environment.

```bash
cd /mnt/d/Study/Cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
```

Phase 5 is intended to run on a Colab GPU from a fresh clone of this branch:

```bash
scripts/run_phase05_colab.sh setup
scripts/run_phase05_colab.sh all
```

See
`plans/260704-distiller-wdc-agent-execution/reports/phase-05-colab-runbook.md`
for cloning, Google Drive persistence, recovery, and result handoff. The runner
must not evaluate the test target or call the teacher LLM.

Recovery is stage-boundary based. Atomic `training_summary.json`, validation
prediction, and validation metric writes serve as completion markers alongside
the best checkpoint. Each completed stage also has an atomic contract containing
the Git commit, runtime configuration, and SHA-256 hashes of its target and
upstream contract files. A run-level contract also fixes the actual GPU name,
resolved precision, and resolved validation batch across all variants. A rerun
skips only when the completion markers and current contracts match. A missing
or mismatched contract blocks reuse and
requires a new `OUTPUT_ROOT` or explicit `FORCE=1`; forced reruns archive stale
contracts and downstream artifacts. Interrupted variants restart from the
beginning rather than resuming mid-epoch. Compact result packages include the
contracts so returned results retain their provenance.

Student validation records local FLAN-T5 generation time, wall time, throughput,
seconds per pair, device name, precision, batch size, and generation limits.
These measurements—not an unrelated small OpenRouter model—provide the student
inference evidence; provider pricing may later be applied to the measured GPU
time under a declared pricing assumption. The aggregator reports signed match
F1, macro F1, and accuracy deltas for every student versus both `llm_random` and
`gold_random`.

Use the existing 128-row label-only FLAN-T5-base result as historical baseline
context. New low-budget training targets should map the trusted supervised
baseline to `gold_random`; validation/test targets can continue using
`gold_label` naming.

## Active Architecture

```text
WDC Products raw/cache data
  -> serialized pair JSONL
  -> low-label budget sampler / active pair selector
  -> direct LLM matcher on fixed evaluation pairs
  -> answer-only teacher LLM labeler
  -> teacher-label validator and cache
  -> target builder: gold_random / llm_random / llm_active_* / mixed_gold_llm_active
  -> compact student training
  -> validation and test evaluation
  -> aggregation, cost table, error analysis
  -> thesis tables and figures
```

Teacher LLM calls are used only for direct-baseline measurement or training-data
creation. Final distilled-student inference must use the compact student without
calling the teacher.

## Immediate Next Steps

Follow the active plan phases:

1. Commit and push the Phase 5 Colab change set, then run
   `scripts/run_phase05_colab.sh all` from a fresh Colab clone of this branch.
2. Return `phase05_train_128_results.tar.gz`; only then compare the three
   students with the fixed direct LLM baseline and write the Phase 5
   continue/revise/stop decision.
3. Keep the test split untouched until the validation decision, then proceed to
   Phase 7 failure and cost analysis after student outputs exist.

The anti-cherry-pick rule is important: fixed evaluation split/sample, prompt
version, model slug, budgets, and cost fields must be declared before results
are inspected.

Predeclared Phase 3 defaults: prompt version `answer_only_v1`, provider
`openrouter`, default model `openai/gpt-5.4-mini`, temperature `0.0`, first
teacher budget `train_128`, first selection strategy `random`, first active
strategy `llm_active_bucketed_v1` with default 25 percent quotas for
`easy_match_candidate`, `hard_match_candidate`, `easy_non_match_candidate`, and
`hard_negative_candidate`, and direct-eval output under
`outputs/distiller_wdc/direct_llm/validation.openrouter.openai-gpt-5-4-mini.answer_only_v1.*`.

## Tech Stack

- Language: Python 3.13
- ML: PyTorch, HuggingFace Transformers
- Data: pandas, datasets
- Validation: Pydantic v2
- First student: `google/flan-t5-base`
- Teacher/direct matcher: OpenRouter-backed LLM calls, answer-only labels first
- Metrics: match precision, match recall, match F1, macro F1, accuracy,
  invalid-output rate, confusion matrix counts, token/cost fields

## Reusable Existing Code

Keep and adapt carefully:

- `data/schema.py`, `data/er_dataset_loader.py`, `data/low_label_sampler.py`,
  and `data/serialize_pairs.py`: WDC pair loading, random budgets, active
  selection manifests, and serialization.
- `experiments/train_mt5.py` and `experiments/evaluate_student.py`: reusable
  compact seq2seq student training/evaluation entry points.
- `supervision/build_targets.py`: active gold-label target builder; extend here
  for `llm_random`, `llm_active_*`, and `mixed_gold_llm_active`.
- `supervision/generate_teacher_labels.py`,
  `supervision/direct_llm_matcher.py`, and
  `supervision/validate_teacher_labels.py`: Phase 3 answer-only LLM cache
  generation, direct-baseline prediction, validation, and cost reporting.
- `models/seq2seq_student.py`: reusable compact seq2seq dataset/model helpers.
- `utils/metrics.py`: binary Entity Matching metrics.

## Removed Historical Code

The branch intentionally removes old Wikidata, multilingual-name, mBART, FEBRL,
and structured-rationale code from the active tree. Keep the research record in
the journals/plans, but do not recreate those packages unless a new experiment
requires them.

## Conventions

- File naming: `snake_case.py`
- Schemas: Pydantic `BaseModel` with explicit fields
- Reproducibility: explicit `seed` parameters, default 42
- Selection: fixed manifests before teacher labels/results are inspected
- Evaluation: train/validation/test split, no test leakage into teacher-label
  generation
- Cost logging: store prompt version, model slug, input tokens, output tokens,
  estimated cost, parsed label, gold label, validity, and selection strategy for
  LLM calls
- Keep files focused; split large modules when they grow beyond a single
  responsibility

When workflow, status, commands, or conventions change after a meaningful
implementation iteration, update both `../AGENTS.md` and this file if the
change affects future agents.
