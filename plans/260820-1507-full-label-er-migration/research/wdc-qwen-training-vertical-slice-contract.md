# WDC–Qwen Gold-vs-Sol-High Training Vertical-Slice Contract

## Status and scope

| Field | Frozen value |
|---|---|
| Status | Frozen and authorized for setup, preflight, smoke, and one full validation-only run per arm |
| Date | 2026-08-27 |
| Dataset | `wdc_products_80cc_small_100un` |
| Compact model | `Qwen/Qwen3-Reranker-0.6B` |
| Training arms | `gold`, `llm_hard` |
| Runs per cell | 1 |
| Excluded dimensions | No training-seed dimension and no model-revision pin |
| Test scope | Locked; this contract authorizes no test predictions |

This narrow contract authorizes the first complete gold-versus-machine-label
experiment for WDC and Qwen only. It does not select the other datasets or
compact models, authorize any paid LLM call, or authorize final-test access.

## Frozen training configuration

Both supervision arms use exactly the same settings. Only the training target
file and its provenance differ.

The training backend may stabilize stochastic framework operations internally,
but that implementation detail is not a varied factor, reported comparison
dimension, output namespace, or aggregation key.

| Setting | Frozen value |
|---|---|
| Model config | `configs/students/qwen3_reranker_0_6b.json` |
| Optimizer | PyTorch `AdamW` |
| Learning rate | `2e-4` |
| Weight decay | `0.01` |
| Schedule | Linear |
| Warmup | `0.10` of optimizer steps |
| Microbatch | 1 pair |
| Gradient accumulation | 16 microbatches |
| Effective batch | 16 pairs |
| Epoch limit | 10 |
| Early-stopping patience | 3 validation epochs |
| Maximum input length | 4,096 tokens |
| Input truncation | Disabled; any overflow fails preflight |
| Validation batch | 1 pair |
| Evaluation batch | 1 pair |
| Precision | `auto`; resolve BF16 when the rented runtime safely supports it, otherwise FP16 |
| Checkpoint metric | Validation macro F1 |
| Threshold selection | Validation macro F1 using the persisted deterministic tie-break implementation |
| LoRA | Rank 8, alpha 16, dropout 0.05; `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Gradient checkpointing | Enabled |
| Answer logits | Final-token `no` = non-match and `yes` = match |
| W&B | Disabled unless explicitly enabled for observability without changing training |

## Inputs and outputs

| Role | Path |
|---|---|
| Gold training target | `data/cache/wdc_products/full_label_targets/gold.jsonl` |
| LLM-hard training target | `data/cache/wdc_products/full_label_targets/llm_hard.jsonl` |
| Validation input/truth | `data/cache/wdc_products/serialized/validation.jsonl` |
| Output root | `outputs/full_label/wdc-qwen-vertical-slice/wdc_products_80cc_small_100un/qwen3-reranker-0-6b/` |

Training preflight consumes the committed gold and LLM-hard JSONL files
directly. It does not invoke target publication validation or require ignored
source-pair and labeler caches. The final target files remain content-hashed in
the preflight contract and included in the no-truncation input audit. The
separate publication validator retains full upstream rederivation. If full
training is separately authorized later, outputs remain separated under
`gold/` and `llm_hard/` and may not be reused when a bound config, target,
validation input, runtime, or code hash changes.

## Validation boundary

Preflight may inspect and tokenize all 2,500 official WDC validation pairs to
verify identity, class counts, train/validation separation, and the 4,096-token
bound. It makes no predictions. The authorized smoke selects a deterministic
balanced fixture of 16 validation rows (8 per class) and predicts only those
rows for its train/evaluate/reload plumbing check.

The two authorized full training arms may score the full official validation
split during checkpoint and threshold selection, then once after reloading the
best merged checkpoint to publish validation metrics. Validation truth remains
evaluation-only and never enters either training target. The official
4,500-pair test split remains untouched and requires a later contract and
explicit approval.

## Rented RTX 3090 setup and preflight

The rented machine must run the active setup and preflight commands. Preflight
must fail unless all of the following hold:

- The rental image already supplies a CUDA-compatible PyTorch build; setup does
  not replace the host's GPU-specific PyTorch package.
- The setup compatibility check removes optional TorchAO only if it demonstrably
  prevents PEFT LoRA injection.
- CUDA is visible and the device name contains `3090` unless the researcher
  explicitly overrides only that hardware-name assertion.
- `torch`, `transformers`, `peft`, and `accelerate` import successfully.
- PEFT injects trainable LoRA parameters into a tiny local module.
- The committed gold and LLM-hard target files are present and readable;
  publication manifests and upstream caches are not training dependencies.
- The validation split contains 2,500 unique official validation pairs.
- Qwen tokenization of both targets and validation has no input over 4,096
  tokens and performs no truncation.
- Runtime identity, exact config/input/code hashes, and resolved precision are
  written to a content-addressed preflight contract.

The tiny smoke run is a plumbing test, not an experimental result. It therefore
sets warmup to zero so its single optimizer step has a nonzero learning rate;
the two full arms retain the old frozen `0.10` warmup ratio.

## Execution gates

- [x] Researcher selected the previously screened Qwen configuration.
- [x] Researcher removed training seed and model-revision pinning from scope.
- [x] Smoke-only compact-model predictions are documented separately from LLM
  labeling and official full validation/testing.
- [x] Setup, preflight, and smoke implementation are authorized.
- [x] The reviewed T4/FP16 smoke completed one finite optimizer step, produced
  16/16 valid fixture predictions, and verified its merged checkpoint manifest;
  its F1 is plumbing evidence, not an experiment result.
- [x] Researcher reviewed the smoke and authorized one full validation-only run
  for `gold`, followed by one for `llm_hard`, with warmup restored to `0.10`.
- [ ] Rented RTX 3090 setup completes.
- [ ] Fresh RTX 3090 preflight passes before full training.
- [ ] Full gold and LLM-hard validation cells complete with matching provenance.

## Stop conditions

Stop without producing experiment results on missing/mismatched targets,
validation leakage into training, input overflow, CUDA/PEFT failure, OOM,
non-finite loss, incomplete checkpoint manifest, failed checkpoint reload, or
artifact-contract mismatch. Never silently truncate, change hyperparameters,
fall back to CPU, call an LLM, or touch the test split.
