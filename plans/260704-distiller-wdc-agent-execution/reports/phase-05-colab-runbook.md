# Phase 5 Colab Runbook

This runbook trains and validates a config-selected 128-budget compact student
on a Google Colab GPU. The completed FLAN-T5-base pilot remains historical
evidence; the next predeclared diagnostic uses the Gemma 3 270M sequence
classifier. A fresh clone of the committed branch contains the three
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
the additional pinned runtime dependencies, including Transformers 4.57 or
newer (and below 5.0).

For Gemma, accept the license on the
[`google/gemma-3-270m`](https://huggingface.co/google/gemma-3-270m) model page
once, add `HF_TOKEN` to Colab Secrets, and authenticate without printing the
token:

```python
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"))
```

Gemma is supported by Transformers; authentication is needed because its
Hugging Face weights are gated by the Gemma license. FLAN-T5 weights are
publicly downloadable and do not require this step.

## 4. Run the resumable pilot

```bash
!STUDENT_CONFIG=configs/students/gemma_3_270m.json \
  bash scripts/run_phase05_colab.sh all
```

The wrapper trains and validates these variants in order:

1. `gold_random`
2. `llm_random`
3. `llm_active_bucketed_v1`

The selected config is snapshotted once at the run root:
`outputs/students/{student_id}/train_{budget}/student_config.json`, alongside
`runtime_contract.json` and the three variant directories. The fixed direct
baseline remains under `outputs/distiller_wdc/direct_llm/` and is only copied
into the compact handoff for comparison.

The classifier maps its two logits to literal `match` / `non-match` prediction
text and also records both class probabilities. It cannot produce malformed
text, so its invalid-output rate is zero. FLAN-T5 still uses
`MAX_TARGET_LENGTH=8` and `MAX_NEW_TOKENS=8`; these seq2seq-only controls do not
govern the Gemma classifier.

Execution defaults are hardware-aware without changing the training batch:

- `MAX_NEW_TOKENS=8` for seq2seq binary validation predictions; ignored by
  sequence classifiers.
- `VALIDATION_BATCH_SIZE=auto`: 32 on BF16-capable CUDA GPUs such as A100,
  otherwise 16 on CUDA; CPU smoke checks retain the training batch size.
- `PRECISION=auto`: BF16 when CUDA reports support, FP16 on other CUDA GPUs,
  and FP32 on CPU.
- Training, validation, and final prediction inputs are tokenized once per
  process. The 512-token truncation limit and fixed padding remain unchanged.

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

A run-level `runtime_contract.json` fixes the actual GPU name, resolved
precision, and resolved validation batch across all three variants. If Colab
reconnects with a different GPU class, the runner stops before mixing runtimes;
use a new `STUDENT_OUTPUT_ROOT`, or use `FORCE=1` and rerun every affected variant.

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
%env STUDENT_OUTPUT_ROOT=/content/drive/MyDrive/low_label_distilling/students
!STUDENT_CONFIG=configs/students/gemma_3_270m.json \
  bash scripts/run_phase05_colab.sh all
```

## 5. Download the compact handoff

After all three validation runs finish, the `all` command creates:

```text
outputs/students/gemma-3-270m/artifacts/phase05_gemma-3-270m_train_128_results.tar.gz
```

When `STUDENT_OUTPUT_ROOT` points to Drive, the archive is created under that root
instead. This compact archive contains training summaries and logs, validation
predictions and metrics, the pilot CSV/JSON, fixed targets, direct-LLM artifacts,
the cost-scenario CSV and assumptions, and provenance, including each variant's
training and evaluation contracts. It
also includes signed match F1, macro F1, and accuracy deltas for each student
versus `llm_random` and `gold_random`; positive values favor the row variant.
It intentionally excludes the large model weights.

Download it from the Colab Files pane or from Google Drive and provide that one
archive for Phase 5 analysis.

Checkpoint archives are optional and can be created separately:

```bash
!STUDENT_CONFIG=configs/students/gemma_3_270m.json \
  bash scripts/run_phase05_colab.sh package-checkpoints
```

This produces one large archive per student variant under the same `artifacts/`
directory. Each checkpoint archive includes its validated training contract and
the run-level runtime contract. Packaging reads the recorded GPU identity, so
it can be repeated after the original GPU is detached while still validating
the current code, configuration, and target hashes.

## Useful recovery commands

Run or resume one variant:

```bash
!STUDENT_CONFIG=configs/students/gemma_3_270m.json \
  bash scripts/run_phase05_colab.sh run llm_random
```

Regenerate the pilot table after copying outputs back into place:

```bash
!STUDENT_CONFIG=configs/students/gemma_3_270m.json \
  bash scripts/run_phase05_colab.sh aggregate
```

Package the compact results again:

```bash
!STUDENT_CONFIG=configs/students/gemma_3_270m.json \
  bash scripts/run_phase05_colab.sh package-results
```
