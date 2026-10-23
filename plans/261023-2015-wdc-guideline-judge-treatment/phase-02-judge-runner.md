# Phase 2: Judge runner, offline

## Context Links

- Pattern: `supervision/run_evidence_check.py`, `tests/test_evidence_check.py`
- Guideline: `supervision/judge_guideline.md`

## Overview

Priority P1. Pending. A runner that sends the guideline and two records to the
judge, saves every raw response, and validates outputs. Tested with a fake
client; no paid call in this phase.

## Requirements

- New `supervision/run_identity_judge.py` with actions `prepare`, `run`, `report`.
- Reuse from `run_evidence_check`: `select_pairs`, `parse_fields`, `dump`,
  `cost_of`, request/response markers, the "no blind retry" rule, the file lock.
  Import rather than copy.
- `--selection uncertain|audit`, `--pass-index 1..3`; each pass and the audit
  write to their own output directory under
  `outputs/uncertainty_refinement/wdc-identity-judge/`.
- Payload: system = guideline text; user = records only (`Record A:` onward).
  No pair ID, label, score or selection flag in the payload (test asserts it).
- Response schema (Pydantic, strict): `verdict` ∈ {0, 1, 2}; `step` ∈ 1..7;
  `reasoning` non-empty; `evidence` list of `{side, field, quote}` (≥ 1).
- Validation:
  - every quote is a literal substring of the named field and side;
  - verdict/step consistency: step 2 → verdict 2; step 6 → verdict 1, or 0
    when the block rule fired; steps 1, 3, 4, 5, 7 → verdict 0;
  - step 2 needs quotes from both sides; step 1 needs two quotes from one side.
- `prepare` writes `contract.json` (source and guideline hashes, selection,
  config, schema, `gold_accessed: false`) and refuses to reuse a directory if
  any of them changed.
- `run` requires `--confirm-paid-judge`; `--fake` uses a deterministic offline
  client for tests and dry runs.
- `report` writes `verdicts.csv` (pair_id, verdict, step, valid, cost) and a summary.
- Audit selection: implements contract item 3 (seeded, stratified), excludes the 158.

## Related Code Files

- Create: `supervision/run_identity_judge.py`, `tests/test_identity_judge.py`
- Read only: `supervision/run_evidence_check.py`, `labeller-screening/settings.json`

## Implementation Steps

1. Schema, payload builder, validator.
2. Selection (uncertain reuses `select_pairs`; audit sampler).
3. Prepare/run/report with fake client.
4. Tests: payload has no label/ID/score; quote and consistency checks; schema
   rejects extra fields; audit sample deterministic and disjoint from the 158;
   contract refuses changed inputs; unresolved request blocks re-dispatch.
5. Dry run with `--fake` on all 158 + 100; full test suite passes.

## Todo List

- [ ] Schema and validator
- [ ] Selection and audit sampler
- [ ] Runner with fake client
- [ ] Tests pass (focused + full suite)
- [ ] Dry run on 258 pairs

## Success Criteria

- Dry run produces 258 valid fake verdicts and a report; zero API calls; tests green.

## Risk Assessment

- Judge ignores the step order. Mitigation: consistency check rejects mismatched verdict/step.
- Output cap too low for long records. Mitigation: same 8,192-token cap as evidence check; reject `finish_reason != stop`.

## Security Considerations

- API key from environment only; record text treated as data in the system prompt.

## Next Steps

- Phase 3 after Phase 1 sign-off.
