# Phase 5 Colab Runbook

This runbook trains and validates a config-selected 128-budget compact student
on a Google Colab GPU. The completed FLAN-T5-base pilot and collapsed first
ModernBERT run remain historical evidence. The current model-screening jobs
repair the ModernBERT-base low-data training mechanics and separately test
FLAN-T5-base with complete 2,700-token inputs and Qwen3-Reranker-0.6B with
LoRA, without changing any experiment targets or validation rows. A fresh
clone of the committed branch contains the three
training targets, gold validation target, and fixed GPT-5.4-mini direct
validation artifacts required by preflight. The workflow does not call the
teacher LLM and does not read or evaluate the test split.

Before using this runbook, make sure the Phase 5 change set has been committed
and pushed to `codex/distiller-wdc-implementation`; an unpushed local branch
cannot be cloned by Colab.

## 1. Start a GPU runtime

In Colab, select **Runtime > Change runtime type > GPU**.

## 2. Clone the fixed branch

```bash
!git clone --branch codex/distiller-wdc-implementation --single-branch \
  https://github.com/TuanMaiz/low-label-distilling-thesis.git
%cd low-label-distilling-thesis
```

For a private repository, use the HTTPS/authentication method already used for
your GitHub account in Colab. Do not put a personal access token into a notebook
that will be shared.

## 3. Install the Phase 5 runtime dependencies

```bash
!bash scripts/run_phase05_colab.sh setup
```

`setup` keeps Colab's CUDA-compatible PyTorch installation and installs only
the additional version-constrained runtime dependencies, including
Transformers 4.57 or newer (and below 5.0). The exact resolved package versions
are recorded and enforced later through the run-level provenance.

The selected ModernBERT, FLAN-T5, and
[`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
weights are public and ungated. No `HF_TOKEN`, Hugging Face login, or
model-access approval is required. Setup also installs PEFT for LoRA training.

## 4. Run the resumable pilot

```bash
!STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh all
```

For the separate full-input FLAN-T5 screening run, start an A100 runtime when
available and use a fresh root:

```bash
!STUDENT_CONFIG=configs/students/flan_t5_base_full_input.json \
  STUDENT_OUTPUT_ROOT=outputs/students-flan-full-input \
  bash scripts/run_phase05_colab.sh all
```

This resolves to 2,700 input tokens, input truncation disabled, training batch
4, and validation/evaluation batch 4. The fixed FLAN inputs peak at 2,649
tokens. Threshold calibration is not run; a future optional job may compare
the validation likelihoods of the exact `match` and `non-match` target strings.

For the Qwen reranker screening run, use an A100 runtime and the default student
output root:

```bash
!STUDENT_CONFIG=configs/students/qwen3_reranker_0_6b.json \
  STUDENT_OUTPUT_ROOT=outputs/students \
  bash scripts/run_phase05_colab.sh all
```

The Qwen run keeps the model's pretrained causal-LM reranking interface. Record
A is the query, Record B is the document, and the final `no`/`yes` token logits
become non-match/match probabilities. It does not generate free-form text.
Validation macro F1 selects the adapter checkpoint, and the
validation-selected decision threshold is stored and reused by evaluation.

The wrapper trains and validates these variants in order:

1. `gold_random`
2. `llm_random`
3. `llm_active_bucketed_v1`

The selected config is snapshotted once at the run root:
`${STUDENT_OUTPUT_ROOT}/{student_id}/train_{budget}/student_config.json`, alongside
`runtime_contract.json` and the three variant directories. The fixed direct
baseline remains under `outputs/distiller_wdc/direct_llm/` and is only copied
into the compact handoff for comparison.

The classifier maps its two logits to literal `match` / `non-match` prediction
text and also records both class probabilities. It cannot produce malformed
text, so its invalid-output rate is zero. FLAN-T5 still uses
`MAX_TARGET_LENGTH=8` and `MAX_NEW_TOKENS=8`; these seq2seq-only controls do not
govern the ModernBERT classifier or Qwen reranker.

Execution defaults are architecture- and hardware-aware:

- `MAX_NEW_TOKENS=8` for seq2seq binary validation predictions; ignored by
  sequence classifiers.
- `VALIDATION_BATCH_SIZE=auto`: 32 on BF16-capable CUDA GPUs such as A100,
  otherwise 16 on CUDA; CPU smoke checks retain the training batch size.
- `PRECISION=auto`: BF16 only on native Ampere-or-newer BF16 hardware, FP16 on
  T4 and other older CUDA GPUs, and FP32 on CPU.
- `BATCH_SIZE=auto`: 1 for Qwen rerankers, 16 for sequence classifiers, and 4
  for seq2seq students.
- Qwen uses 16-step gradient accumulation for an effective batch of 16,
  validation/evaluation batch 1, learning rate `2e-4`, 10 percent warmup, at
  most 10 epochs, and early-stopping patience 3. Gradient checkpointing is
  enabled.
- Qwen LoRA trains rank-8 adapters with alpha 16, dropout 0.05, and
  `q_proj`/`k_proj`/`v_proj`/`o_proj` targets. The original model parameters
  remain frozen.
- ModernBERT trains its randomly initialized classification head alone for two
  epochs, then unfreezes the final four encoder blocks. Head and encoder
  learning rates are `1e-3` and `1e-5`, with 10 percent linear warmup.
- Classifier checkpoints are selected by validation macro F1 with match F1 as
  the tie-breaker. The macro-F1-selected validation threshold is stored in
  `decision_threshold.json` and automatically reused by evaluation.
- Training, validation, and final prediction inputs are tokenized once per
  process. Classifiers tokenize Record A and Record B as a complete pair with
  `MAX_INPUT_LENGTH=2400`; the fixed targets and validation set peak at 2,334
  ModernBERT tokens. Truncation is disabled, so any future overflow fails
  instead of silently dropping record content. The historical FLAN config
  remains at 512 tokens with truncation, while `flan-t5-base-full-input` uses
  2,700 tokens with truncation disabled and fails on any future overflow.
- Qwen uses dynamic left padding under a configurable 4,096-token limit with
  truncation disabled. Before training, preflight tokenizes the exact frozen
  reranker prompt for all three fixed training targets and all validation rows.
  It writes `input_length_audit.json` at the run root and stops with the
  overflowing pair IDs and required token counts if any prompt exceeds the
  limit.

Rerunning `all` skips a variant only when its completion artifacts and atomic
stage contract both exist and match the current run. Training contracts record
the Git commit and relevant training configuration plus SHA-256 hashes of the
training and validation targets. Evaluation contracts record the evaluation
configuration and hashes of the training contract and validation target.

A missing or mismatched contract blocks reuse instead of silently mixing an old
Drive result with a new checkout. Use a different `STUDENT_OUTPUT_ROOT`, or use
`FORCE=1` only when intentionally replacing that stage. Forced retraining or
reevaluation archives the old contract and affected artifacts with a
`.stale.<timestamp>` suffix. Recovery remains stage-boundary based: an
interrupted training process restarts that variant rather than resuming
mid-epoch.

For Qwen, a training stage is complete only when the selected
`best_adapter/`, merged standalone `best_model/`, validation threshold,
`checkpoint_manifest.json`, summary, and matching contract all exist.
The checkpoint manifest records the relative path, byte size, and SHA-256 of
every file under both checkpoint directories. It is verified before a completed
stage is reused, before evaluation, and before either archive is packaged.
Evaluation always loads the merged `best_model`. An incomplete, corrupted, or
forced rerun archives stale adapter/model directories before restarting the
affected variant.

A run-level `runtime_contract.json` fixes the actual GPU name, resolved
precision, resolved validation batch, exact Python/package versions, and
immutable Hugging Face model/tokenizer commit across all three variants. On the
first preflight, the resolved commit is written into the run's
`student_config.json`; `runtime_provenance.json` records Python, Torch,
Transformers, PEFT, Accelerate, and Hugging Face Hub versions. If Colab
reconnects with a different GPU class or software environment, the runner stops
before mixing runtimes; use a new `STUDENT_OUTPUT_ROOT`, or use `FORCE=1` and
rerun every affected variant. When the run-level identity is forcibly replaced,
the runner archives all active variant directories first, so a selected rerun
cannot be aggregated with stale metrics from the previous environment.
Automatic partial aggregation additionally validates each variant's current
training and evaluation contracts. Variants produced under different shared
overrides (for example input length, seed, epochs, or learning rate) are
reported as missing rather than mixed into the summary.

Validation metrics include synchronized local student inference time,
evaluation wall time, throughput, seconds per pair, device name, precision,
batch size, and sequence limits. These measurements are the student inference
evidence. Do not substitute a small OpenRouter model as a price proxy: it is a
different model and serving stack. Training summaries also record synchronized
`trainer.train` wall seconds, including epoch validation and checkpointing but
excluding model loading and tokenization.

The aggregator applies all predeclared low/base/high rates from
`configs/phase05_cost_assumptions.json`: $0.25, $1.00, and $4.00 per GPU-hour.
They are analytical GPU-hour
sensitivity assumptions—not observed Colab charges or current provider
quotes—and must be reported together. For each student and scenario it computes
one-time teacher plus trainer-loop cost, per-pair inference cost, fixed-scale
savings, and the first whole query count reaching cost parity or better. The
scenario table carries `training_time_scope` so the excluded loading and
tokenization work is not mistaken for priced training time. The assumptions
file and its SHA-256 are
included in the compact handoff, so later sourced price assumptions can be
versioned and reaggregated without rerunning training.

For GPU-hour rate `r`, direct validation cost `D` over `N` pairs, measured
trainer-loop seconds `T`, per-pair student inference seconds `I`, and one-time
teacher-label cost `L`, the table uses:

```text
trainer_loop_cost = (T / 3600) * r
student_inference_cost_per_pair = (I / 3600) * r
student_upfront = L + trainer_loop_cost
direct_cost_per_pair = D / N
student_total(N) = student_upfront + N * student_inference_cost_per_pair
savings(N) = D - student_total(N)
break_even = ceil(student_upfront /
                  (direct_cost_per_pair - student_inference_cost_per_pair))
```

`break_even` is null when the denominator is non-positive.

To persist artifacts across Colab disconnects, mount Google Drive and choose an
output root there before running:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%env STUDENT_OUTPUT_ROOT=/content/drive/MyDrive/low_label_distilling/students-modernbert-repair
!STUDENT_CONFIG=configs/students/modernbert_base.json \
  bash scripts/run_phase05_colab.sh all
```

## 5. Download the compact handoff

After all three validation runs finish, the `all` command creates:

```text
outputs/students-modernbert-repair/modernbert-base/artifacts/phase05_modernbert-base_train_128_results.tar.gz
```

When `STUDENT_OUTPUT_ROOT` points to Drive, the archive is created under that root
instead. This compact archive contains training summaries and logs, validation
predictions, metrics, and classifier thresholds; the pilot CSV/JSON, fixed targets, direct-LLM artifacts,
the cost-scenario CSV and assumptions, and provenance, including each variant's
training and evaluation contracts. It
also includes signed match F1, macro F1, and accuracy deltas for each student
versus `llm_random` and `gold_random`; positive values favor the row variant.
It intentionally excludes the large model weights.

Download it from the Colab Files pane or from Google Drive and provide that one
archive for Phase 5 analysis.

For Qwen, the compact handoff is:

```text
outputs/students/qwen3-reranker-0-6b/artifacts/phase05_qwen3-reranker-0-6b_train_128_results.tar.gz
```

Its run tree is:

```text
outputs/students/qwen3-reranker-0-6b/train_128/
├── student_config.json
├── runtime_contract.json
├── runtime_provenance.json
├── input_length_audit.json
├── gold_random/
│   ├── best_adapter/
│   ├── best_model/
│   └── checkpoint_manifest.json
├── llm_random/
└── llm_active_bucketed_v1/
```

The compact archive contains the audit, runtime provenance, checkpoint
manifests, contracts, logs, predictions, metrics, and summaries, but excludes
model weights. Use
`package-checkpoints` when the LoRA adapters and merged standalone models are
also needed.

Checkpoint archives are optional and can be created separately:

```bash
!STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh package-checkpoints
```

This produces one large archive per student variant under the same `artifacts/`
directory. Each checkpoint archive includes its validated training contract,
the run-level runtime contract, and `runtime_provenance.json`. Packaging reads
the recorded GPU identity, so
it can be repeated after the original GPU is detached while still validating
the current code, configuration, and target hashes. Qwen checkpoint archives
include both `best_adapter/` and the merged `best_model/`, plus the input audit
and checkpoint manifest.

For the full-input FLAN-T5 command above, the compact archive is written to:

```text
outputs/students-flan-full-input/flan-t5-base-full-input/artifacts/phase05_flan-t5-base-full-input_train_128_results.tar.gz
```

It preserves the completed 512-token FLAN archive; it contains validation-only
results and does not include the optional sequence-likelihood calibration job.

## Useful recovery commands

Run or resume one variant:

```bash
!STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh run llm_random
```

For Qwen recovery, use the same stage command with its fixed config and output
root:

```bash
!STUDENT_CONFIG=configs/students/qwen3_reranker_0_6b.json \
  STUDENT_OUTPUT_ROOT=outputs/students \
  bash scripts/run_phase05_colab.sh run llm_random
```

Preflight reruns the full token audit first. A completed Qwen variant is skipped
only when its adapter, merged model, threshold, manifest, summary, and contracts
all match; otherwise the variant restarts from its training boundary.

Regenerate the pilot table after copying outputs back into place:

```bash
!STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh aggregate
```

Package the compact results again:

```bash
!STUDENT_CONFIG=configs/students/modernbert_base.json \
  STUDENT_OUTPUT_ROOT=outputs/students-modernbert-repair \
  bash scripts/run_phase05_colab.sh package-results
```
