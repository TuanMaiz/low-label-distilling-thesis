# AGENTS.md

## WDC student instruction rollback (2026-10-15)

The researcher retired the variant-aware prompt and restored the original
student instruction for subsequent work. The existing legacy-named runner
`scripts/run_wdc_qwen_variant_prompt.sh` now selects
`configs/students/qwen3_reranker_0_6b.json` for training and evaluation, with
result identity `quick_refined_original_prompt` and a separate default output
root `outputs/uncertainty_refinement/wdc-qwen-quick-refined-original-prompt`.
The original instruction is: "Determine whether Record A and Record B describe
the same real-world product. Answer yes only when they refer to the same product."
The versioned variant config remains historical evidence; do not select it for
new runs unless specifically requested. Existing variant checkpoints, scores,
thresholds, and downloaded results remain unchanged. A variant-trained checkpoint
must retain its own instruction when reproducing its results.
The rollback changes neither supervision nor threshold selection and launches
no training or inference. Original test match F1 was 0.7581227437 versus
0.7465986395 for the variant; this does not establish why the variant declined
or prove that shorter prompts generally improve generalization.
See `documents/261014-student-variant-instruction.md` for the retired experiment.

Read `../AGENTS.md` and this file before changing the experiment workflow.

## Research Discussion Records

Before continuing research planning or interpreting earlier discussions, consult:

- [Decisions](docs/research/decisions.md): recorded decisions, dates, rationale,
  and rejected or deferred alternatives.
- [Ideas](docs/research/ideas.md): hypotheses and their trial status; a proposal
  or literature recommendation is not an approved experiment.
- [Techniques](docs/research/techniques.md): concepts discussed and their
  project-specific uses and limitations.
- [Open questions](docs/research/open-questions.md): unresolved choices and
  evidence still needed.

These records preserve discussion history; active plans and experiment
contracts govern execution. Distinguish researcher decisions from assistant
suggestions, observed results from causal explanations, and historical runs
from current scope. When updating the records, include only supported discussion
content and retain `[CẦN XÁC NHẬN]` for uncertain dates, reasons, or status.
Do not turn a brainstorm or an open question into execution authorization.

## Active Project

**Master's thesis:** detecting and treating uncertain pairs in LLM-labelled
training data for Entity Resolution (researcher, recorded 2026-10-22; see
`docs/research/decisions.md` D18–D21).

Research question (wording `[CẦN XÁC NHẬN]`):

> How can uncertain pairs in LLM-labelled training data be detected, and how
> should they be treated, so that a compact ER student learns from LLM labels
> without human training labels?

Current direction (WDC first):

- Detection: 149 OOF teacher–student disagreements plus 13 teacher `2/3`
  pairs, 158 unique pairs, all with evidence-check diagnoses. Confident
  Learning is a side analysis only.
- Treatment: a researcher-written guideline applied by an LLM judge (records
  only, no old label), then a fixed rule: not decidable → remove; agrees with
  the teacher → upweight; disagrees → flip. Guideline, judge model and new
  plan are not written yet. No paid judge run is authorized.
- Backlog: `plans/261021-2235-wdc-order-swap-and-hard-pair-upweighting/`
  (swap training, order-flip detector, RHO-style selection). Do not implement it.
- Gold boundary: train gold only in the final analysis; validation gold for
  checkpoint/threshold selection; test once for frozen arms. WDC test already
  influenced earlier decisions (prompt rollback), so it is not a clean
  confirmation set.
- Weekly goals: `plans/weekly-goals/`. Reviewer preparation:
  `reviewers-question-mock/`.

### Previous main question (full-label design)

Whether the full-label 3 × 3 design below remains the thesis backbone or only
background is `[CẦN XÁC NHẬN]`.

> Can LLM-generated hard labels provide a practical alternative to benchmark
> gold training labels for compact cross-encoder Entity Resolution models while
> reducing deployment cost compared with direct LLM matching?

Active execution plan:
`plans/260820-1507-full-label-er-migration/plan.md`

Legacy writing plan (must be revised to match this migration before thesis drafting):
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The experiment contract is not frozen yet. Phase 1 of the active plan selects
the exact two non-WDC datasets, three cross-encoder models, LLM labeler, prompt,
evaluation scope, artifact schema, and cost ceiling.

The WDC labeler-calibration workflow is isolated in `labeller-screening/`.
It freezes one seed-42 uniform random sample of 300 training pairs and compares
the same blinded pairs across three GPT-5.6 Sol settings through OpenRouter.
OpenRouter is pinned to the OpenAI upstream provider with fallbacks disabled.
Its gold CSV is for the comparison script only; never expose it to the labeler
request path.

All three paid screening settings are complete and `sol_high` is selected. The
researcher approved the WDC-only full-label vertical slice in
`plans/260820-1507-full-label-er-migration/research/wdc-sol-high-vertical-slice-contract.md`.
The production run is complete: it reused the verified 300 Sol-high labels,
called the remaining 2,200 official WDC training rows, and published 2,500/2,500
valid labels at USD 2.693225 cumulative cost under the USD 5 ceiling.

The corresponding WDC training targets are published at
`data/cache/wdc_products/full_label_targets/`: 2,500 `gold` rows and 2,500
`llm_hard` rows, with 79 disagreements (0.9684 agreement) and USD 2.693225
labeler cost. `supervision/build_full_label_targets.py` builds the bundle;
`supervision/validate_full_label_targets.py` independently rederives it from
upstream evidence. Target publication made no paid calls, started no training,
and made no validation/test predictions. Phase 3 remains in progress while
the other two datasets are pending.

## Scope Guardrail

For the current question: no new detectors, arms or datasets without a
researcher decision; finish the guideline-judge treatment on WDC first.

The fixed high-level design of the full-label comparison is:

- 3 benchmark datasets.
- 3 compact cross-encoder ER models.
- 2 training-label sources: benchmark gold and LLM-generated hard labels.
- 1 predeclared run per cell; no repeated-run experiment dimension.
- 3 direct LLM baselines, one per dataset.
- Match F1 primary; precision, recall, macro F1, accuracy, timing, throughput,
  cost, and break-even supporting.

Out of scope: low-label budgets, active selection, rationale generation or
distillation, adaptive bi-/cross-encoder cascades, and extra datasets/models unless the
supervisor explicitly requires a change. If the researcher worries that the
work is not enough, remind them that this design is already thesis-grade and
favor completing the frozen plan.

## Current Status

- On 2026-10-06 the researcher authorized an offline quick-refinement pilot in
  `plans/261006-0927-wdc-quick-evidence-refinement/`. The builder
  `supervision.build_quick_refined_dataset` uses three-pass majority labels and
  existing valid diagnoses, with 65 explicit assistant-reviewed decisions.
  It hard-prunes valid case-5 pairs first; reviewed clear identity conflicts
  can become non-match and safe agreements match. Difficult/unanalysed/invalid
  pairs retain majority labels. No semantic keyword classifier or paid call.
  Published separate ignored `data/cache/wdc_products/refinement/quick-tier-v1/`:
  `train.jsonl` and `labels.csv` contain 2,493 retained pairs; `decisions.csv`
  accounts for all 2,500. Seven pruned; one match-to-non-match relabel
  (`88031640#6256141`, listed RAM 8GB versus 16GB); 2,492 labels unchanged.
  Retained classes: 489 match, 2,004 non-match. Real independent audit and
  byte-identical rerun pass; repository suite 224/224. This is an
  assistant-reviewed experimental dataset, not verified corrected truth or
  measured F1 improvement. Originals and validation/test remain untouched;
  no training, further paid calls, commit or push authorized by this step.

- Evidence retry update (2026-10-06): the researcher explicitly authorized
  retrying five invalid diagnoses and three old uncertain dispatches. All eight
  new replies are saved under `retries/remaining-02/`; coverage is now 155/158
  valid diagnoses with three remaining literal-quote failures. The prior
  seven-invalid retry round is retained under `retries/invalid-01/`.
  Known received-response cost across all rounds is USD 3.253112. Original
  uncertain dispatch receipts remain unreconciled for billing only; their pairs
  now have valid new diagnoses. Do not infer missing diagnosis coverage from
  `unresolved_request_ids` alone. All original response bytes are preserved.
  No further paid retry is authorized by this completed round.

- The refinement extension is recorded in
  `plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/`.
  WDC `run_full_wdc.py --no-reuse-existing` now skips screening reuse for
  separate teacher passes. Pass 2 completed with 2,500/2,500 valid fresh
  labels, zero reuse, zero retries, and USD 2.709705 cost. Pass 3 completed
  with 2,500/2,500 valid fresh labels, zero reuse, one transient failed attempt
  followed by a successful retry, and USD 2.693205 measured valid-response
  cost. Existing pass 1 remains in `full_sol_high`.
  `supervision/build_teacher_consensus.py` then combined the three aligned
  passes offline into ignored artifacts under
  `data/cache/wdc_products/majority_labels/`: 2,500 majority labels (495
  `match`, 2,005 `non_match`) with 2,487 `3/3` and 13 `2/3` consistency rows.
  Phase 4 then committed deterministic three-fold WDC-Qwen OOF inputs under
  `data/cache/wdc_products/oof/majority-3fold/` plus preparation, verification,
  execution, recovery, and CL-ready merge tooling. Fold 1 has 1,333/333/834
  train/validation/held-out rows; folds 2 and 3 each have 1,334/333/833. An
  isolated committed-tree `verify-data` run passed. Phase 5 then completed
  three OOF runs on a Tesla V100-SXM2-32GB. The downloaded result package
  passed checksum verification and independent merge rederivation: 2,500
  unique OOF probabilities. Phase 6 uses pinned Cleanlab 2.9.0 through
  `supervision.build_confident_learning`: 120 ranked candidates (4.8%) and
  a calibrated error-count estimate of 124 (4.96%), using no human labels.
  Raw/calibrated confident joints, estimated joint distribution, all-pair
  scores, and candidate IDs are under the ignored
  `outputs/uncertainty_refinement/wdc-qwen-cl/`. Counts are model estimates;
  no gold comparison, fusion, prune, or relabel has been performed. Do not
  mutate the original pair JSONL to attach labels or scores.

- The researcher authorized WDC teacher evidence-check classification under
  `plans/261005-0916-wdc-teacher-evidence-case-classification/`. The union of
  13 teacher `2/3` pairs and 149 stored OOF teacher–student disagreements has
  158 unique pairs (4 overlap). `supervision.run_evidence_check` calls the
  existing Sol-high/OpenAI-only OpenRouter client with the seven definitions
  in `supervision/evidence_case_prompt.md`. Requests contain records only,
  not previous labels, scores, or selection flags. Outputs are separate
  multi-label diagnoses, exact source quotes, CSV/Markdown tables and costs
  under ignored `outputs/uncertainty_refinement/wdc-evidence-check/`.
  These are model-generated observations, not verified errors or whole-dataset
  prevalence. This authorization does not permit relabel/prune, training,
  human-label comparison or official validation/test access. Existing responses
  are reused; unresolved dispatches must not be blindly repeated.
  After the researcher raised the key limit to USD 20 and authorized
  continuation, all 57 never-attempted pairs completed and the one explicitly
  rejected HTTP 403 request was retried, retaining its request/error under
  `retry_attempts/0100.json`. Current coverage: 155 saved responses, 148 valid
  diagnoses, 7 invalid quotes and 3 old unresolved dispatches; no never-attempted
  pairs remain. Known cumulative response cost is USD 2.8317829, including
  USD 1.0764331 for this continuation; total accounting remains incomplete.
  The original 97 responses and source hashes are unchanged; 1,119 literal
  evidence quotes independently verified. No arbitrary/uncertain retry was
  performed. Resolve the seven invalid outputs and reconcile the three old
  dispatches before claiming complete valid coverage. Never change the key
  limit yourself; `--skip-unresolved` only dispatches never-attempted pairs.

- Branch: `refactor/full-label-er-migration`.
- Low-label sampler, active selector, old Phase-03/04 orchestration, old
  Phase-05 runner, and superseded execution-plan directory are removed.
- Historical experiment results and journals remain as research evidence.
- The three paid labeler-screening runs are complete; Sol-high achieved 291/300
  correct at USD 0.327135 and was selected for cost/accuracy balance.
- `labeller-screening/run_full_wdc.py` is the only active production labeling
  runner. The authorized WDC run completed with zero invalid results and zero
  retries. Its final result is
  `data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv`.
- The WDC `gold` and `llm_hard` target artifacts are published at
  `data/cache/wdc_products/full_label_targets/`; both contain all 2,500
  official training pairs and validate against their upstream evidence.
- WDC target alignment check passed: `gold` and `llm_hard` each contain 2,500
  unique training pairs with identical ordered pair IDs and input text. They
  have 79 label disagreements.
- The narrow WDC–Qwen experiment completed one full-validation run per arm on a
  rented RTX 3090 with bf16, in fixed order `gold` then `llm_hard`. Each arm
  produced 2,500 valid validation predictions with zero invalid rows. Gold
  achieved match F1 0.8208409506398537, macro F1 0.8853308695851598, and
  accuracy 0.9216; `llm_hard` achieved match F1 0.8141263940520446, macro F1
  0.881578997229896, and accuracy 0.92. The test split remains locked.
- `scripts/run_wdc_qwen_vertical_slice.sh` implements setup, preflight, smoke,
  both confirmed full-training actions, result verification, and packaging.
  CPU-side orchestration and recovery verification passes 21/21 focused tests;
  the current full repository suite passes 193/193 and labeler-screening passes 13/13.
  Recovery fails closed on partial evaluation files, binds the training summary
  to the persisted checkpoint manifest, revalidates artifact-contract hashes,
  and checks packaged members against current verified results. The combined
  results package was downloaded and its SHA-256 checksum verified. This first
  two-arm vertical slice does not complete the global 3×3 plan; the other 16
  compact-model arms and all final-test work remain pending.
- The multi-environment training preflight consumes the committed final target
  files directly and does not invoke publication/upstream validation; the
  separate publication validator retains full upstream rederivation.
- The broader Phase-1 contract remains unfinished. Do not run other paid
  labeling cells, relabel WDC, or make official full-validation/test
  predictions until their checklists and explicit flags are complete.
- Dataset 2 DBLP-ACM has been freshly acquired from the official
  DeepMatcher archive plus independent HTTPS file downloads. The two copies are
  byte-identical. A reproducible local audit is implemented at
  `scripts/inspect_dblp_acm_source.py`; its profile, observation, normalization,
  identity, and attribution contract are frozen under `configs/datasets/`.
  Phase 2 has a strict adapter and atomic preparation path that generated
  ignored train/validation JSONL for researcher review: 7,417 train and 2,473
  validation pairs. Test remains source-contract-only and no `test.jsonl`, paid
  label, model run, or prediction exists. `--verify-only` independently
  rederives expected bytes and rejects forged manifests and symlink aliases.
  Phase 2 is researcher-approved. Phase 3's separate neutral JSON-Schema stack
  now prepares gold-free train inputs and completed a 7,417-row deterministic
  offline fake run with zero API calls and USD 0 cost. Its connected temporary
  integration passes the unchanged target publisher and independent validator;
  no production DBLP target or test artifact was created. Source verification
  independently rederives train without opening validation/test, response
  reconciliation is durable, and only the concrete fake client may dispatch.
  Phase 3 was researcher-approved on 2026-09-03. Phase 4 now has a separate
  strict DBLP Qwen execution profile, publication-specific student config,
  CPU-only preflight, fixture result/package verifier, and command-rendering
  shell dispatcher. It freezes 464 optimizer steps/epoch, 4,640 planned steps,
  and 464 warmup steps. Official setup/training/evaluation/package actions
  remain locked until real GPT-5.6 Sol targets and later GPU authorization
  exist. Phase 4 is researcher-approved and Phase 5's offline compatibility
  handoff is complete: source inspection and repeated preparation match,
  `--verify-only` passes alongside downstream artifacts, the 7,417-row fake
  bundle independently rederives, and WDC compatibility remains unchanged.
  This does not authorize or imply production DBLP labels, targets, training,
  metrics, or test access.

## Commands

Use the uv-managed environment; never use bare system `python`, `pip`, or
`pytest` for repository work.

```bash
cd /mnt/d/study/cao-hoc/luan-van/code
source .venv/bin/activate
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s labeller-screening/tests -v
.venv/bin/python -m supervision.validate_full_label_targets \
  --target-dir data/cache/wdc_products/full_label_targets
.venv/bin/python -m supervision.build_teacher_consensus \
  --inputs data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl \
  --pass-01 data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv \
  --pass-02 data/cache/wdc_products/teacher_labels/full_sol_high_pass_02/predictions/sol_high.csv \
  --pass-03 data/cache/wdc_products/teacher_labels/full_sol_high_pass_03/predictions/sol_high.csv \
  --output-dir data/cache/wdc_products/majority_labels \
  --expected-count 2500
.venv/bin/python -m tests.test_teacher_consensus real-files \
  --inputs data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl \
  --pass-01 data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv \
  --pass-02 data/cache/wdc_products/teacher_labels/full_sol_high_pass_02/predictions/sol_high.csv \
  --pass-03 data/cache/wdc_products/teacher_labels/full_sol_high_pass_03/predictions/sol_high.csv \
  --expected-count 2500
.venv/bin/python -m supervision.prepare_oof_folds verify \
  --inputs data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl \
  --majority-labels data/cache/wdc_products/majority_labels/llm_unrefined.csv \
  --output-dir data/cache/wdc_products/oof/majority-3fold \
  --dataset-id wdc_products_80cc_small_100un \
  --expected-count 2500 \
  --validation-percent 20
PYTHON=.venv/bin/python bash scripts/run_wdc_qwen_oof.sh verify-data
PYTHON=.venv/bin/python bash scripts/run_wdc_qwen_oof.sh plan
.venv/bin/python -m supervision.build_confident_learning \
  --oof-csv outputs/uncertainty_refinement/wdc-qwen-oof/oof_predictions.csv \
  --oof-summary outputs/uncertainty_refinement/wdc-qwen-oof/oof_summary.json \
  --output-dir outputs/uncertainty_refinement/wdc-qwen-cl \
  --expected-count 2500
.venv/bin/python scripts/inspect_dblp_acm_source.py \
  --archive data/raw/dblp_acm/dblp_acm_exp_data.zip \
  --source-root data/raw/dblp_acm/archive-2026-09-01/exp_data \
  --direct-root data/raw/dblp_acm/acquisition-2026-09-01 \
  --observed-on 2026-09-01
.venv/bin/python -m data.prepare_benchmark \
  --dataset-config configs/datasets/dblp_acm.json \
  --source-root data/raw/dblp_acm/archive-2026-09-01/exp_data \
  --output-root data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f \
  --verify-only
.venv/bin/python -m supervision.run_full_labeling \
  --pairs data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/serialized/train.jsonl \
  --dataset-profile configs/datasets/dblp_acm.json \
  --labeler-config configs/labelers/dblp_acm_sol_high.json \
  --output-dir data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/teacher_labels/fake_sol_high_phase3 \
  --expected-count 7417 \
  --fake
bash scripts/run_dblp_acm_qwen_vertical_slice.sh config
bash scripts/run_dblp_acm_qwen_vertical_slice.sh identity
bash scripts/run_dblp_acm_qwen_vertical_slice.sh plan
```

On the rented RTX 3090, whose image must already include CUDA-compatible
PyTorch:

```bash
bash scripts/run_wdc_qwen_vertical_slice.sh setup
bash scripts/run_wdc_qwen_vertical_slice.sh preflight
bash scripts/run_wdc_qwen_vertical_slice.sh smoke
bash scripts/run_wdc_qwen_vertical_slice.sh train-gold --confirm-full-training
bash scripts/run_wdc_qwen_vertical_slice.sh package-arm gold
bash scripts/run_wdc_qwen_vertical_slice.sh train-llm-hard --confirm-full-training
bash scripts/run_wdc_qwen_vertical_slice.sh verify-results
bash scripts/run_wdc_qwen_vertical_slice.sh package-results
```

## Planned Architecture

```text
frozen contract
  -> dataset registry and three verified loaders
  -> dataset-namespaced serialized splits
  -> complete gold and LLM-hard-label training targets
  -> three frozen compact cross-encoder models
  -> 18 train/evaluate cells plus 3 direct LLM baselines
  -> provenance-checked metrics and cost aggregation
  -> thesis tables and figures
```

Gold validation/test labels are evaluation-only. LLM machine labeling operates
on training pairs only and must publish targets only at 100% valid unique
coverage. Compact-model inference never calls the LLM labeler.

## Reusable Code

- Data: `data/schema.py`, `data/er_dataset_loader.py`,
  `data/serialize_pairs.py`.
- Supervision: `supervision/generate_teacher_labels.py`,
  `supervision/direct_llm_matcher.py`, `supervision/build_targets.py`,
  `supervision/validate_teacher_labels.py`,
  `supervision/full_label_protocol.py`,
  `supervision/openrouter_json_schema_client.py`,
  `supervision/prepare_full_label_inputs.py`,
  `supervision/run_full_labeling.py`,
  `supervision/build_teacher_consensus.py`,
  `supervision/build_confident_learning.py`,
  `supervision/build_full_label_targets.py`,
  `supervision/validate_full_label_targets.py`.
- Compact ER models (legacy code paths retain `student` naming during migration):
  `configs/students/`, `models/student_config.py`,
  `models/classification_student.py`, `models/generative_reranker_student.py`,
  `experiments/train_student.py`, `experiments/evaluate_student.py`.
- Reproducibility: `utils/artifact_contract.py`,
  `utils/runtime_provenance.py`, `utils/checkpoint_manifest.py`,
  `utils/torch_runtime.py`.
- Metrics/cost: `utils/metrics.py`, `utils/cost_accounting.py`,
  `analysis/cost_summary.py`.

## Conventions

- Python files: `snake_case.py`.
- Schemas: Pydantic `BaseModel` with explicit fields.
- Preserve official dataset splits and namespace pair/cache identities by
  dataset and version.
- Freeze manifests, prompts, model identities, and evaluation IDs before results.
- Store API keys only in environment variables; never commit secrets.
- Preserve unrelated dirty-worktree changes.
- After meaningful workflow changes, update `../AGENTS.md`, this file, and
  `CLAUDE.md` together.
