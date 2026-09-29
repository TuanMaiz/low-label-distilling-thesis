# PEFT/TorchAO Colab Compatibility Fix

## Context

Phase 5 Qwen LoRA screening must run reliably in Colab without changing the predeclared experiment inputs or touching the fixed test split.

## What happened

PEFT adapter injection failed in some Colab environments with `ImportError: Found an incompatible version of torchao`. The failure occurred before the diagnostic could begin training.

## Root cause

Colab could provide an optional TorchAO build incompatible with the installed PyTorch/PEFT stack. Treating TorchAO as universally broken would also remove compatible installations unnecessarily.

## Decision

Setup now performs a tiny LoRA injection smoke test and removes TorchAO only when that test reproduces the exact incompatibility. Compatible TorchAO is preserved, and unrelated PEFT failures remain visible. Qwen preflight rechecks LoRA injection before loading the remote model, while runtime provenance records the installed `torchao` version or its absence.

## Verification

The compatibility, preservation, and failure-routing paths are covered by tests; the full suite passes with 94 tests. Phase 5 remains in progress, with experiment inputs unchanged and the test split untouched.

## Next

Run the predeclared Phase 5 diagnostics in Colab and retain the resulting runtime provenance with each result package.
