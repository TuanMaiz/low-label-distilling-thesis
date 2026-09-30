# CLAUDE.md

Read `../AGENTS.md` and `AGENTS.md` before changing the experiment workflow.
`AGENTS.md` is the canonical repository guidance.

## Active Project

The project is migrating to a full-label Entity Resolution comparison:

`3 datasets × 3 compact cross-encoder models × {gold, LLM hard labels}`

Direct LLM matching is a per-dataset accuracy/cost baseline. Low-label budgets,
active selection, rationale generation or distillation, adaptive cascades, and
repeated-run experiments are outside the frozen scope unless the supervisor
requires them.

Active plan:
`plans/260820-1507-full-label-er-migration/plan.md`

WDC labeler screening:
`labeller-screening/plan.md`

The screening workflow reuses one seed-42 random sample of 300 blinded WDC
training pairs for `sol_high`, `sol_max`, and `sol_pro_max` through OpenRouter,
pinned to the OpenAI upstream provider with fallbacks disabled. Gold labels
remain in a separate comparison-only CSV. Preparation and dry runs are
implemented, and all three paid screening settings are complete. Sol-high was
selected. The frozen WDC-only vertical-slice contract authorized reusing its
300 completed labels and calling the remaining 2,200 WDC training rows under a
USD 5 cumulative ceiling with the dedicated confirmation flag. That run is
complete: 2,500/2,500 valid labels, 300 reused plus 2,200 new, zero invalid
results or retries, and USD 2.693225 cumulative cost. The published result is
`data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv`.

The offline WDC target publication is also complete at
`data/cache/wdc_products/full_label_targets/`: 2,500 `gold` rows and 2,500
`llm_hard` rows, with 79 disagreements (0.9684 agreement) and USD 2.693225
labeler cost. `supervision/build_full_label_targets.py` builds the bundle, and
`supervision/validate_full_label_targets.py` independently rederives it from
the recorded upstream evidence. Publication made no paid calls, started no
training, and made no validation/test predictions. Phase 3 remains in progress
until the other two dataset target pairs are published.

The WDC target alignment check passed: both arms contain 2,500 unique training
pairs with identical ordered pair IDs and input text, with 79 label
disagreements.

The narrow WDC–Qwen experiment completed one full-validation run per arm on a
rented RTX 3090 with bf16, ordered `gold` then `llm_hard`; final-test access
remains forbidden. Each arm produced 2,500 valid validation predictions with
zero invalid rows. Gold achieved match F1 0.8208409506398537, macro F1
0.8853308695851598, and accuracy 0.9216. `llm_hard` achieved match F1
0.8141263940520446, macro F1 0.881578997229896, and accuracy 0.92. The
executable entry point is `scripts/run_wdc_qwen_vertical_slice.sh`; it makes no
LLM calls and never reads the test split.

CPU-side implementation verification passes 21/21 focused WDC–Qwen tests; the
current repository suite passes 193/193 and labeler-screening passes 13/13. The recovery path
rejects partial evaluation files instead of overwriting them, verifies that the
training summary embeds the persisted checkpoint manifest, rechecks recorded
contract hashes, and compares archive members with live verified results. The
combined results package was downloaded and its SHA-256 checksum verified.
Training preflight consumes the committed final targets directly and does not
invoke publication validation; `supervision.validate_full_label_targets`
remains the separate upstream rederivation command for the publication
environment. This first two-arm vertical slice does not complete the global
3×3 plan; the other 16 compact-model arms and all final-test work remain
pending.

Legacy writing plan (revise before thesis drafting):
`plans/260704-distiller-wdc-thesis-writing/plan.md`

The remaining global datasets/models and experiment-wide prompt, evaluation,
artifact, and cost decisions remain Phase-1 work. DBLP-ACM's Dataset-2 contract
is already frozen. Do not guess unfinished decisions from historical configs or results.

DBLP-ACM is Dataset 2 with a researcher-approved frozen source, normalization,
identity, and attribution contract. The official archive and independent HTTPS
file downloads were acquired into ignored raw storage and confirmed
byte-identical by `scripts/inspect_dblp_acm_source.py`. The executable candidate
profile and observation manifest are under `configs/datasets/`. Phase 2's
adapter atomically prepared only ignored train/validation artifacts (7,417 and
2,473 pairs) and independently verifies their expected bytes. Test remains
locked and non-materialized. Phase 2 is researcher-approved. Phase 3's neutral
JSON-Schema supervision path has completed a 7,417-row deterministic offline
fake run with zero API calls and USD 0 cost; its connected fixture passes the
unchanged target publisher and independent validator. It created no production
DBLP target or test artifact. Train-source verification does not open validation
or test, response reconciliation is durable, and only the exact fake client may
dispatch. Phase 3 was researcher-approved on 2026-09-03. Phase 4's separate
strict DBLP Qwen execution profile, publication instruction, CPU-only preflight,
fixture result/package checks, and command-rendering runner are implemented and
researcher-approved. The derived schedule is 464 optimizer steps per
epoch, 4,640 planned steps, and 464 warmup steps. Official actions remain locked
because production GPT-5.6 Sol targets and GPU approval do not exist. Phase 5's
offline handoff is complete: source/repeated preparation, real-cache
`--verify-only`, fake publisher/independent-validator integration, DBLP Qwen
fixtures, and WDC compatibility all pass. Production labels, targets, weights,
metrics, and test artifacts do not exist.

## Commands

The refinement extension plan is under
`plans/260928-0938-uncertainty-aware-llm-label-refinement-fundamental/`.
`labeller-screening/run_full_wdc.py --no-reuse-existing` supports separate
teacher passes without importing screening labels. Pass 2 completed with
2,500/2,500 valid fresh labels, zero reuse, zero retries, and USD 2.709705
cost. Pass 3 completed with 2,500/2,500 valid fresh labels, zero reuse, one
transient failed attempt followed by a successful retry, and USD 2.693205
measured valid-response cost; the completed original run is pass 1. The three
aligned passes were combined offline by `supervision.build_teacher_consensus`
into 2,500 majority labels: 495 `match`, 2,005 `non_match`, 2,487 at `3/3`
consistency, and 13 at `2/3`. These are separate artifacts under
`data/cache/wdc_products/majority_labels/`; the original pair JSONL remains
unchanged. Phase 4 committed deterministic three-fold WDC-Qwen inputs under
`data/cache/wdc_products/oof/majority-3fold/` and the CPU verification,
execution, recovery, and CL-ready merge tooling. An isolated committed-tree
`verify-data` run passed. No real OOF model training or probabilities exist;
those remain Phase 5 on a separately authorized CUDA machine.

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
.venv/bin/python -m supervision.prepare_oof_folds verify \
  --inputs data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl \
  --majority-labels data/cache/wdc_products/majority_labels/llm_unrefined.csv \
  --output-dir data/cache/wdc_products/oof/majority-3fold \
  --dataset-id wdc_products_80cc_small_100un \
  --expected-count 2500 \
  --validation-percent 20
PYTHON=.venv/bin/python bash scripts/run_wdc_qwen_oof.sh verify-data
PYTHON=.venv/bin/python bash scripts/run_wdc_qwen_oof.sh plan
.venv/bin/python -m tests.test_teacher_consensus real-files \
  --inputs data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl \
  --pass-01 data/cache/wdc_products/teacher_labels/full_sol_high/predictions/sol_high.csv \
  --pass-02 data/cache/wdc_products/teacher_labels/full_sol_high_pass_02/predictions/sol_high.csv \
  --pass-03 data/cache/wdc_products/teacher_labels/full_sol_high_pass_03/predictions/sol_high.csv \
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

On the rented RTX 3090 with CUDA-compatible PyTorch already installed:

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

Use the uv-managed environment. The only authorized production labeling path was
`labeller-screening/run_full_wdc.py` for the frozen WDC Sol-high vertical slice,
and its full training-label run is complete. Do not relabel WDC. The only
authorized full compact-model cells are the two narrow WDC–Qwen validation arms
described above; all other experiment cells remain blocked by the broader
Phase-1 contract.

For complete rules, scope guardrails, reusable files, and conventions, follow
`AGENTS.md`.
