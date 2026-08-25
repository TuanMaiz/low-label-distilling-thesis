---
date: 2026-08-24
session: wdc-sol-high-full-labeling
---

# Journal: 2026-08-24 — WDC Sol-High Full Labeling

## Context

The frozen WDC-only vertical slice tests complete LLM-generated hard-label
supervision before the broader 3×3 experiment contract is finalized. It fixes
the official 2,500-row WDC training split and the screened
`openai/gpt-5.6-sol` high-reasoning setting, while leaving validation and test
data outside the labeling path.

## What Happened

- Reused 300 verified Sol-high screening labels and made paid requests only for
  the remaining 2,200 training pairs under the approved cumulative USD 5 cap.
- Published 2,500/2,500 valid unique predictions with zero invalid responses
  and zero retries: 495 `match` and 2,005 `non_match`.
- Recorded USD 2.693225 cumulative cost: USD 0.327135 reused and USD 2.366090
  new. Usage totaled 995,653 input tokens and 63,716 output tokens.
- Compared the frozen predictions with gold only after publication: 2,421/2,500
  agreed (96.84%), with 79 disagreements. This is analysis only; no machine
  label was corrected or replaced with gold.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl` | `e443fc38fe1206ce961c6f71dce28e50b4e148d720dda3b2bdc688abf196e1ea` |
| `data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.manifest.json` | `406584698e6423583be342ce82c0f13ed501de9f4fdececa8f6befc8da6e7b1c` |
| `data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.attempts.jsonl` | `091e29535312299ae8c6d9becabb3c70e0b853d478f65769e01c02c4cf28f0fb` |
| `data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.audit.jsonl` | `32bfcd11e53c5c8d17b0541dad5e2286e2b051b4263c0d9d542b035730be7b35` |
| `data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv` | `bc37f9113fd363b76a042588c46e95195cb2821223c2f580941121239c99e5d8` |
| `data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.run.json` | `704be597e76bb155c4290832a39b9b3612ebda24c791f262cef7c7f3dc6eeca3` |

The run manifest also binds the serialized source, blinded input IDs, request
payloads, settings, runner, shared provider client, both contracts, Git commit,
and dirty-worktree status. Requests were gold-free, pinned to the OpenAI
upstream with fallbacks disabled, constrained by strict answer-only JSON, and
published only after the 100% unique-coverage gate passed. The reused 300-row
inputs and attempts are bound by hashes
`1475442e91331986a74e07af652e1a47eaa4afa4916dff710b2fd73167a2cb75`
and `d8a133e08c7549989dad012849d578e9bc7e658d95700dee581f07172bd4f4ce`.

## Reflection

The full run stayed close to the screening-based cost projection while
preserving the key scientific separation between machine labeling and gold
evaluation. The 79 disagreements are experimental signal: correcting them
would leak gold supervision into the LLM-label condition and invalidate the
comparison.

## Decisions Made

| Decision | Rationale | Impact |
|---|---|---|
| Keep all published Sol-high labels unchanged | Gold is evaluation-only | The LLM-hard-label training condition remains independent |
| Treat the finalized manifest and hashes as the immutable run boundary | Make a paid run auditable despite a dirty worktree | Downstream targets can reject provenance drift |
| Continue with the WDC/Qwen vertical slice | Complete one end-to-end experiment before expanding Phase 1 | Produces the first gold-versus-LLM student comparison |

## Next Steps

- Build the complete WDC LLM-hard-label training targets from the finalized
  prediction artifact.
- Train and evaluate the Qwen reranker against the corresponding gold-label
  baseline, without changing the 79 disagreement labels.
