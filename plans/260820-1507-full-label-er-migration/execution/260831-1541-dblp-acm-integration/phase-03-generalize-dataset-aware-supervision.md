---
phase: 3
title: "Generalize Dataset-Aware Supervision"
status: pending
priority: P1
effort: "1-2d"
dependencies: [1, 2]
---

# Phase 3: Generalize Dataset-Aware Supervision

## Overview

Add one canonical neutral JSON-Schema full-training labeling stack for DBLP-ACM
alongside—never through—the completed WDC screening/production stack. This
phase proves behavior with a fake client and makes zero paid calls.

## Requirements

- Functional: DBLP publication instructions and version are separately frozen;
  requests use strict JSON Schema and responses accept exactly one
  `match`/`non_match` label, matching the screened WDC semantics.
- Functional: local blinded rows contain exactly `pair_id,input_text`, but the
  actual outbound `messages` contain only the instruction and `input_text`.
  `pair_id`, labels, entity/source IDs, metadata, validation, and test data never
  enter provider messages.
- Functional: cache/run identity includes dataset/version/input/profile, model,
  reasoning, requested routing, prompt/parser/schema versions and hashes, plus
  hashes of the new runner, request builder, parser, and provider client.
- Functional: accept only the canonical API origin
  `https://openrouter.ai/api/v1` (normalized exact scheme/host/port/path, no
  userinfo/query/fragment); reject alternatives/redirects before resolving the
  API secret or performing network I/O.
- Functional: offline dry-run reports the frozen observed train count, zero API calls, estimated
  request count/cost inputs, and the separate explicit paid-confirmation gate.
- Non-functional: no DBLP gold screening or prompt tuning; reuse the frozen
  Sol model, high reasoning, and OpenAI-only requested routing only after the
  DBLP prompt/protocol contract is approved. Describe upstream as requested,
  not proven, unless the response supplies trustworthy provider evidence.
- Non-functional: `supervision/prompts.py`, historical `load_settings`, WDC
  screening code/settings/results, and WDC targets remain untouched.
- Non-functional: future charged calls durably journal an inflight identity
  before dispatch. Unresolved inflight state fails closed for manual billing/
  response reconciliation; exactly-once billing is not promised.

## Architecture

`verified normalized train JSONL + DBLP labeler profile -> local blinded input
manifest -> neutral JSON-Schema request builder/parser -> fake/OpenRouter client
-> inflight/attempts/predictions/audit/run/settings -> unchanged
publish_full_label_targets -> independent validator`.

The zero-cost synthetic integration exercises this complete graph with the
actual new fake-client runner; no handcrafted disconnected publisher fixture is
accepted. The current WDC runner/screening assets and the old plain-text prompt
module remain historical, hash-stable entrypoints.

### Exact publisher handoff

| `publish_full_label_targets` input | Actual new-stack artifact/identity |
|---|---|
| `pairs_path` | verified normalized DBLP train JSONL (`split=train`, gold label retained locally) |
| `predictions_path` | CSV with exactly `pair_id,result` and complete unique `match`/`non_match` coverage |
| `attempts_path` | JSONL with sequential attempts, setting, requested/returned model, status/result, usage or reserved cost, and local pair correlation |
| `audit_path` | exact attempts JSONL projection with only `result` removed |
| `labeler_run_path` | JSON binding setting/model/prompt version/max attempts/input hash plus dataset/source/input-manifest/settings provenance hashes |
| `blinded_inputs_path` | ordered JSONL with exactly local `pair_id,input_text`; only `input_text` enters the user message |
| `blinded_inputs_manifest_path` | train/count/field list/source hash/input hash manifest |
| `labeler_settings_path` | DBLP JSON-Schema settings with prompt instructions/version, selected model, high reasoning, OpenAI-only requested routing, retry/price limits |
| remaining arguments | safe output root, `dblp_acm`, observed logical version, and frozen expected count |

## Related Code Files

- Create: `supervision/full_label_protocol.py`
- Create: `supervision/openrouter_json_schema_client.py`
- Create: `supervision/prepare_full_label_inputs.py`
- Create: `supervision/run_full_labeling.py`
- Reuse frozen in Phase 1: `configs/labelers/dblp_acm_sol_high.json`
- Reuse: `supervision/build_full_label_targets.py`
- Reuse: `supervision/validate_full_label_targets.py`
- Preserve unchanged: `supervision/prompts.py`, `supervision/llm_providers.py`, `supervision/generate_teacher_labels.py`, `labeller-screening/screening_lib.py`, `labeller-screening/run_full_wdc.py`, `labeller-screening/settings.json`, `labeller-screening/artifacts/**`
- Create: `tests/test_dataset_aware_supervision.py`
- Run unchanged: `tests/test_teacher_labels.py`, `tests/test_full_label_targets.py`, `labeller-screening/tests/test_labeller_screening.py`

## Tests Before

1. Actual request payload uses the DBLP publication prompt, selected model, high
   reasoning, JSON Schema, and OpenAI-only requested routing; every outbound
   message is inspected and contains no `pair_id`, labels, IDs, or metadata.
2. Strict parser rejects extra keys, invalid labels, prose, multiple choices,
   refusals, abnormal finish reasons, and mismatched returned model.
3. Origin validation rejects HTTP, alternate host/port/path, userinfo, query,
   fragment, encoded ambiguity, and redirects before secret resolution/network.
4. Local blinded input has exactly `pair_id,input_text`, the frozen number of unique ordered
   IDs, and hashes bound to normalized train/profile; no test artifact is read.
5. Any dataset/version/input/profile/model/reasoning/routing/prompt/parser/schema/
   code-hash mismatch prevents cache reuse.
6. Output paths using traversal/symlinks/aliases/protected WDC roots fail before
   directory creation.
7. Charged-call simulation persists inflight identity before fake dispatch;
   crash-before-response leaves unresolved state that fails closed and never
   retries automatically. Completed reconciliation is explicit and auditable.
8. Full fake-client run emits real predictions, attempts, audit, run, settings,
   blinded input and manifest; these exact artifacts feed the untouched target
   publisher and then the independent validator.
9. Missing/duplicate/extra/invalid fake results block publication; successful
   integration preserves pair/input parity in both target arms.
10. Legacy prompt/settings loader and all WDC screening/parser/reuse/cost tests
    remain unchanged and green.

## Implementation Steps

1. Define the new DBLP labeler settings and `full_label_protocol.py` with frozen
   publication instructions/version, JSON Schema request builder, strict parser,
   selected model/high reasoning, and OpenAI-only requested routing. Do not
   modify or import the legacy plain-text prompt protocol for this path.
2. Implement a generic blinded-input builder that validates train-only input,
   strips each row to `pair_id,input_text`, and writes deterministic JSONL plus
   a manifest binding dataset/profile/source hashes.
3. Validate the exact OpenRouter origin before resolving environment secrets.
   Implement a new JSON-Schema client that rejects redirects and records routing
   as requested unless response provenance proves the selected upstream. Do not
   import the legacy client, which imports the plain-text prompt module.
4. Build outbound payloads from `input_text` only; keep `pair_id` in a local
   correlation map/journal and verify serialized messages in tests.
5. Bind resume/cache identity to all scientific/protocol/code dimensions. Apply
   the Phase-2 safe-root policy to every labeler output.
6. Before any future charged dispatch, append+fsync an `inflight` row containing
   request identity/payload hash/reserved cost. On restart, unresolved inflight
   fails closed for manual reconciliation; document that provider idempotency is
   required for exactly-once billing. This plan uses only a fake client.
7. Make the fake-client runner produce the exact publisher inputs:
   `pairs_path`, `predictions_path`, `attempts_path`, `audit_path`,
   `labeler_run_path`, `blinded_inputs_path`,
   `blinded_inputs_manifest_path`, and `labeler_settings_path`, plus output root,
   dataset/version, and expected count.
8. Feed those real synthetic artifacts unchanged into
   `publish_full_label_targets`, then run
   `validate_full_label_target_directory`. Do not modify the builder/validator
   unless this connected test proves a genericity defect.
9. Keep paid mode behind positive spend ceiling and
   `--confirm-paid-labeling`; in this plan record only a zero-call full-train
   dry-run/estimate and synthetic integration.

## Success Criteria

- [ ] DBLP train preparation produces the frozen number of deterministic blinded rows with
  only `pair_id,input_text` and no gold/evaluation truth.
- [ ] New DBLP protocol uses publication instructions and strict JSON Schema;
  no DBLP gold rescreen occurs and legacy prompt/settings code is untouched.
- [ ] Actual outbound messages contain only instruction plus `input_text`—never
  `pair_id`, source IDs, truth, metadata, validation, or test content.
- [ ] Exact-origin, safe-path, composite-cache, and inflight-crash tests fail
  closed before secret/network/mutation where applicable.
- [ ] Dry-run and full synthetic integration use zero network/API calls; the
  actual runner artifacts pass the untouched publisher and independent validator.
- [ ] Output inventory maps exactly to every publisher argument and preserves
  gold/LLM-hard pair order and input-text parity.
- [ ] No paid label, production DBLP target, validation/test prediction, or
  direct-LLM artifact is created.

## Risk Assessment

The highest-risk failures are truth/identifier leakage, stale-cache reuse,
host/origin substitution, unsafe output resolution, and duplicate charges after
a crash. Mitigate with real-payload inspection, a separate protocol module,
exact origin/safe-root checks before secrets, composite identities, durable
inflight rows, and manual reconciliation. Exactly-once provider billing is
explicitly not guaranteed. Historical WDC modules/assets are immutable.
