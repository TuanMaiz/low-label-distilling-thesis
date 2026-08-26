---
date: 2026-08-26
session: wdc-qwen-portable-preflight
---

# Journal: 2026-08-26 — Portable WDC–Qwen Training Preflight

## Context

The WDC–Qwen preflight failed in a fresh Colab clone because it invoked the
publication validator. That validator intentionally rederives the published
targets from original serialized training pairs and labeler evidence, which are
local publication inputs rather than committed training-runtime dependencies.

## What Happened

- Changed training preflight to consume the committed `gold.jsonl` and
  `llm_hard.jsonl` targets directly.
- Removed both manifests, the target builder, and upstream publication evidence
  from the runner's required files and artifact-contract dependencies. The files
  were not deleted; training simply no longer depends on them.
- Kept training-time checks for target readability, the approved Qwen config,
  official validation shape, train/validation pair-ID separation, runtime
  identity, and the existing no-test/no-LLM boundaries.
- Preserved the strong publication validator unchanged. The bound target builder
  remains byte-identical at SHA-256
  `414c70b5eae7821789a49ce4734bb7bdc4e7bb3e266c64634fc4e1960aa03440`.

## Reflection

Publication verification and training consumption have different portability
requirements. Requiring uncommitted upstream evidence on every GPU host did not
protect training; it coupled a portable consumer to the machine that produced
the data. The final committed targets are now the training interface, while
full provenance rederivation remains an explicit publication audit.

## Decisions Made

| Decision | Impact |
|---|---|
| Load the two committed JSONL arms directly | Fresh clones can run preflight across Colab and rented GPU environments |
| Remove manifests and upstream evidence only as runner dependencies | Publication artifacts remain available without blocking training |
| Leave publication validation separate and unchanged | The original evidence chain and frozen builder binding remain auditable |

## Verification

- The focused WDC-Qwen preflight and full-label-target tests passed, including
  a target-only fixture with no publication cache.
- `.venv/bin/python -m supervision.validate_full_label_targets --target-dir
  data/cache/wdc_products/full_label_targets`: passed independently with 2,500
  rows per arm, 79 disagreements, and 0.9684 agreement.

## Boundaries

This change authorizes no paid labeling, full two-arm training, or official
full-validation/test prediction. The narrow workflow remains limited to RTX
3090 setup, preflight, and the tiny balanced smoke run pending human review and
explicit approval for full training.

## Next

Run `setup`, `preflight`, and `smoke` from a fresh GPU clone, then review the
generated runtime and smoke artifacts before requesting full-run approval.
