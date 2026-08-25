---
date: 2026-08-25
session: wdc-qwen-rtx3090-smoke-workflow
---

# Journal: 2026-08-25 — WDC–Qwen RTX 3090 Smoke Workflow

## Context

The first WDC gold-versus-LLM vertical slice needs a reproducible rented-GPU
entry point before either full training arm is authorized. The narrow contract
covers environment setup, fail-closed preflight, and a tiny balanced Qwen LoRA
smoke run on an RTX 3090; it does not expand the unfinished 3×3 contract.

## What Happened

- Added one `setup` → `preflight` → `smoke` workflow around the previously
  screened `Qwen/Qwen3-Reranker-0.6B` configuration. It reuses the old LoRA
  setup and full-run hyperparameters instead of introducing a new tuning pass.
- Froze the full experiment settings at AdamW, learning rate `2e-4`, weight
  decay `0.01`, effective batch 16, 10 epochs, patience 3, 4,096-token inputs,
  and linear scheduling with `0.10` warmup.
- Kept training seed and model-revision pinning out of the experiment contract.
  The trainer's internal RNG stabilization is an implementation detail, not a
  varied factor, output namespace, aggregation key, or reported dimension.
- Added fail-closed checks for both 2,500-row target bundles, the official
  validation split, train/validation ID separation, Qwen configuration, CUDA
  device identity, precision resolution, input length without truncation, and
  PEFT LoRA injection.
- Bound runtime identity plus exact config, input, contract, runner, trainer,
  model-backend, validation, and utility content hashes into the preflight
  artifact contract. Changed content must produce a mismatch instead of silently
  reusing old outputs.
- Added conditional PEFT/TorchAO sanitation: keep TorchAO when LoRA injection
  works, and remove it only when its incompatibility is the demonstrated cause
  of adapter-injection failure.
- Defined a deterministic balanced smoke fixture with eight rows per class.
  The one-step smoke uses warmup `0.0` so the plumbing check executes a nonzero
  learning-rate optimizer step; the full gold and LLM-hard arms retain the
  frozen `0.10` warmup ratio.

## Reflection

The smoke run is deliberately a plumbing test, not an experimental result. Its
warmup exception prevents a false pass where the only optimizer step has zero
learning rate, while keeping the actual two-arm comparison unchanged. Binding
hardware/software identity and file content is more useful than adding a seed
dimension that the thesis does not intend to analyze.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Reuse the old Qwen config and full-run hyperparameters | Preserve continuity with the screened model | No new tuning degree of freedom enters the comparison |
| Do not pin model revision or add a seed dimension | These are outside the approved experiment factors | One predeclared run per cell remains the design |
| Record internal RNG and runtime provenance without elevating them to dimensions | Stabilization and auditability are implementation concerns | Outputs remain comparable without claiming repeated-run evidence |
| Use `0.0` smoke warmup but `0.10` full-run warmup | Guarantee a meaningful one-step smoke update | Smoke validates training plumbing without changing experiment settings |
| Fail closed on provenance, truncation, CUDA, PEFT, or incomplete outputs | Prevent silent drift and partial-result reuse | Full training cannot start from an unverified runtime |

## Boundaries

The workflow makes no LLM calls and never reads or predicts the official WDC
test split. Validation is used only for compact-model smoke plumbing and later
model selection. Full gold and LLM-hard training remains blocked until the RTX
3090 setup, preflight, and smoke artifacts pass human review and the researcher
explicitly authorizes both arms.

## Next Steps

- Run `setup`, `preflight`, and `smoke` on the rented RTX 3090.
- Review runtime identity, artifact-contract hashes, input audit, checkpoint
  reload, and smoke validation output.
- Request explicit approval before launching either full training arm.
