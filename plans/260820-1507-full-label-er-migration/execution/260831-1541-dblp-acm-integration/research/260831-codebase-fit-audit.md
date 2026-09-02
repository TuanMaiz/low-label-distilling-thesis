# DBLP-ACM codebase fit summary

## Reusable without redesign

- `data/schema.py`: dictionary-shaped `GenericERRecord` attributes provide a
  candidate-neutral target for whatever fields the local observation confirms.
- `supervision/build_full_label_targets.py`: accepts explicit dataset ID,
  version, expected count, paths, and generic pair rows.
- `experiments/train_student.py` and `experiments/evaluate_student.py`: consume
  path-based normalized/target JSONL and do not require WDC record columns.
- Metrics, cost-accounting patterns, artifact contracts, runtime provenance,
  checkpoint manifests, and classification-threshold utilities are reusable.
  The existing OpenRouter client is not imported for DBLP because it imports the
  legacy prompt protocol and does not satisfy the new exact-origin/no-redirect
  boundary.

## Minimal changes required

1. Add a thin `data/loaders/dblp_acm.py` adapter and one small dataset profile.
2. Add an explicit single-dataset preparation command; defer the global
   exact-three included-config assertion until Dataset 3 exists.
3. Thread configured attribute order through new preparation behavior, then
   regenerate WDC into temporary output and require byte parity.
4. Add one new neutral DBLP JSON-Schema protocol/runner. Local blinded inputs
   contain `pair_id,input_text`, but outbound messages contain only instructions
   plus `input_text`. Add a separate exact-origin/no-redirect provider client;
   do not import `supervision/llm_providers.py`, which imports the legacy
   `supervision/prompts.py` plain-text protocol.
5. Add new DBLP/profile Qwen preflight/runner files accepting independently
   observed split counts and a reviewed dataset instruction. Existing WDC preflight,
   runner, config, settings, and screening code are hash-bound and immutable;
   accept limited duplication until a versioned global runner exists.

## Hard-coded areas to isolate

- `data/serialize_pairs.py` defaults to WDC product attribute order.
- `supervision/prompts.py` says “real-world product” and is a legacy plain-text
  protocol, so the DBLP production path must not use or modify it.
- `labeller-screening/run_full_wdc.py`, its settings/assets, the Qwen WDC config,
  `experiments/wdc_qwen_preflight.py`, and
  `scripts/run_wdc_qwen_vertical_slice.sh` encode the completed WDC slice.

The completed WDC paths are artifact-contract-bound reproducibility evidence.
Create new profile-driven entrypoints alongside them; never edit, wrap, or
refactor the frozen WDC preflight, runner, config, settings, screening code,
targets, results, or assets.

## Test emphasis

- Adapter schema/hash/identity/count/order/missingness fixtures derived from the
  locally frozen observation.
- Determinism plus the locally reviewed overlap and duplicate policies.
- WDC serializer/prompt/artifact byte parity.
- Real outbound-payload truth/ID redaction, exact-origin/safe-path checks,
  composite cache identity, durable inflight crash behavior, and zero paid calls.
- Actual fake-runner outputs wired to the untouched target publisher and
  independent validator—not handcrafted disconnected publisher fixtures.
- DBLP Qwen preflight with locally frozen independent split counts, the reviewed
  canonical pair/overlap policy, locally derived scheduling, gold-first
  package lifecycle, relocated-checkout identities, strict test lock, and all
  current WDC runner/recovery regressions.

Planning baseline: 132 repository tests, 12 labeler-screening tests, and 21
focused WDC orchestration/recovery tests passing.
