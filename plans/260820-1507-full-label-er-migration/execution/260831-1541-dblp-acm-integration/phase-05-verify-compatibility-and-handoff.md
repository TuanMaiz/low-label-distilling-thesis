---
phase: 5
title: "Verify Compatibility and Handoff"
status: pending
priority: P1
effort: "0.5-1d"
dependencies: [1, 2, 3, 4]
---

# Phase 5: Verify Compatibility and Handoff

## Overview

Run the complete offline verification suite, prove WDC compatibility, and hand
off a precise DBLP-ACM readiness report. Acquisition occurs only if explicitly
approved in Phase 1; otherwise report fixture-ready/source-blocked and do not
claim source verification or completion. This phase always ends before paid
labeling, GPU execution, or test materialization/evaluation.

## Requirements

- Functional: independently verify profile/contract agreement, deterministic
  preparation, source/normalized hashes, split audits, blinded-input schema,
  actual outbound payload, origin/path/cache/inflight controls, the connected
  fake-runner-to-publisher/validator integration, and DBLP preflight fixtures.
- Functional: compare protected WDC files/artifacts against Git/blob baselines,
  regenerate WDC serialization into temporary output, and run all regressions.
- Functional: document exact commands, created artifact paths, counts/hashes,
  remaining manual gates, and the future authorized sequence.
- Non-functional: reports distinguish “fixture ready,” “source verified,”
  “dataset frozen,” “labels produced,” “targets published,” and “experiment
  completed.”

## Architecture

Verification consumes only fixtures, exact checksum-approved source if Phase 1
authorized acquisition, and existing committed WDC artifacts. A handoff report
is the only result; downstream paid/GPU operations require new explicit approval
and are not chained automatically.

## Related Code Files

- Create: `plans/260820-1507-full-label-er-migration/execution/260831-1541-dblp-acm-integration/reports/verification.md`
- Update: this plan and its phase statuses through `ck plan check`
- Update after meaningful workflow changes: `data/README.md`, `AGENTS.md`, `CLAUDE.md`, `../AGENTS.md`
- Verify, do not modify: `data/cache/wdc_products/**`, `labeller-screening/**`, `experiments/wdc_qwen_preflight.py`, `scripts/run_wdc_qwen_vertical_slice.sh`, `configs/students/qwen3_reranker_0_6b.json`, `outputs/new/**`

## Verification Matrix

| Check | Evidence | Required result |
|---|---|---|
| Source contract | Five files, headers, hashes, counts | Exact match or fixture-ready/source-blocked |
| Adapter | Normalized rows/stats/manifest | Locally frozen train/validation counts; test contract only |
| Missingness | Manifest + fixtures | Exact locally frozen missingness reproduced |
| Pair integrity | Identity/relationship/duplicate audit | Exact locally frozen rules reproduced |
| Record overlap | Manifest | Locally observed overlap values and reviewed policy reproduced |
| Serialization | Fresh temporary generation | DBLP deterministic; WDC bytes unchanged |
| Supervision payload | Actual fake-client payload | Messages contain instruction + input text only; strict JSON Schema |
| Origin/path/cache | Hostile fixtures | Reject before secret/network/mutation |
| Crash accounting | Fake charged-call crash | Durable inflight; unresolved restart fails closed |
| Target integration | Actual fake-run artifacts | Untouched publisher and independent validator succeed |
| Qwen preflight | DBLP fixtures | Observed counts/policies, derived schedule, test/LLM lock |
| Runner lifecycle | State/package fixtures | Gold verify/package/checksum precedes LLM-hard |
| Portability | Relocated-checkout fixture | Repo-relative identity stable; runtime path safely changes |
| WDC regressions | Suites + regeneration + Git/blob hashes | No historical workflow/config/artifact drift |

## Implementation Steps

1. Run the four new focused test modules, WDC-focused regressions, full repository
   suite, and labeler-screening suite using `.venv/bin/python`.
2. Run preparation twice into temporary locations and compare every normalized
   train/validation artifact byte/hash; run `--verify-only`. Confirm test evidence
   is contract-only and no normalized/labeled test JSONL exists.
3. Regenerate WDC serialization into a temporary location, byte-compare it to
   committed outputs, and verify Git/blob hashes of protected WDC preflight,
   runner, config, settings, screening code, targets, and contracts.
4. Run hostile origin/path/cache and charged-call crash fixtures. Confirm origin
   rejection precedes secret resolution/network, unsafe paths mutate nothing,
   and unresolved inflight state never auto-retries.
5. Run the actual neutral fake-client runner, inspect outbound messages, then
   feed all emitted artifacts unchanged into `publish_full_label_targets` and
   `validate_full_label_target_directory`.
6. Run DBLP Qwen preflight, lifecycle, corruption, and relocated-checkout
   fixtures. Confirm locally derived optimizer/warmup steps, canonical-source overlap
   rejection, gold-first verified packaging, stable portable identities, and no
   test/LLM/CUDA/model/checkpoint access.
7. Write `reports/verification.md` with commands, environment, test counts,
   audit values, hashes, limitations, acquisition decision, and remaining gates.
8. Update plan phase status using `ck plan check`; update the parent plan only to
   reflect proven readiness after the human contract gate. Do not mark parent
   Phase 2/3/5 complete.
9. Present the next separately authorized sequence: approve/freeze source and
   license, acquire/prepare exact source, review dry-run cost, explicitly approve
   paid labeling, publish/verify two targets, then authorize a rental-GPU
   validation slice. Never bundle those operations into this handoff.

## Success Criteria

- [ ] New focused tests, existing 21 WDC-focused tests, full repository suite,
  and 12 labeler-screening tests pass; actual totals are recorded.
- [ ] Repeated DBLP preparation is byte-identical and `--verify-only` succeeds
  without rewriting or downloading.
- [ ] Source/count/balance/missingness/overlap audits match the contract; only
  train/validation are materialized and test remains contract-only.
- [ ] Actual fake-run payload/inflight/output artifacts pass security controls,
  untouched publication, and independent rederivation with zero paid calls.
- [ ] DBLP preflight/lifecycle/relocation fixtures freeze the locally derived schedule,
  canonical-source separation, gold-first packaging, and test/LLM/CUDA locks.
- [ ] Temporary WDC regeneration is byte-identical and Git/blob hashes confirm
  protected WDC code/config/settings/contracts/artifacts were not edited.
- [ ] If acquisition lacks approval, handoff says fixture-ready/source-blocked
  and the plan is not marked source-verified or completed.
- [ ] Handoff does not imply DBLP labels, targets, trained weights, metrics, or
  normalized test artifacts exist.

## Risk Assessment

A green focused test can hide cross-workflow regressions or misleading status.
Mitigate with full-suite execution, fresh temporary WDC regeneration, protected
Git/blob comparisons, real connected fake-run integration, independent target
rederivation, hostile security fixtures, and explicit state terminology. If
license/source approval is unresolved, report fixture-ready/source-blocked; do
not acquire or proceed automatically.
