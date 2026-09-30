---
title: "Phase 4: Prepare WDC Qwen OOF Artifacts and Runner"
status: complete
priority: P1
effort: "1-2d implementation and CPU verification"
dependencies: [3]
---

# Phase 4: Prepare WDC Qwen OOF Artifacts and Runner

## Overview

Prepare the complete three-fold out-of-fold (OOF) input contract and its
execution tooling before renting a GPU. This phase is CPU-only: it freezes the
fold membership, writes and independently verifies each fold's three data
files, commits the verified fold inputs for direct use by a clean clone,
renders the future Qwen commands, and proves the verifier/merger with fixture
predictions. It does not train a model or create real OOF predictions.

The existing generic trainer and evaluator already accept dynamic paths through
`--train-targets`, `--validation-targets`, `--output-dir`, `--input`,
`--predictions`, and `--metrics`. Therefore, this phase must reuse those entry
points rather than change model architecture or duplicate training code.

## Approved WDC Pilot Decisions

Researcher-approved on 2026-09-30.

- Dataset: the 2,500 frozen blinded WDC training pairs.
- Labels: `data/cache/wdc_products/majority_labels/llm_unrefined.csv` only.
- Detector student: `configs/students/qwen3_reranker_0_6b.json` for this WDC
  pilot. This is the dataset-level detector, not the only downstream
  student: later evaluation still trains all three thesis students on the same
  refined target set. An optional second-detector ablation on WDC may be added
  only if the supervisor requires detector-architecture robustness.
- Outer protocol: three deterministic, class-aware folds. Each pair is the
  held-out member of exactly one fold.
- OOF unit: `pair_id`, matching the example identity exposed by each dataset
  adapter. No extra entity-level grouping or cross-pair record logic is added.
- Inner validation: for each outer fold, take a deterministic class-aware 20%
  subset from the other two folds. It may select checkpoints and early stopping
  but must never contain the outer held-out pairs.
- Seed-independent assignment: order each majority-label class by a stable
  SHA-256 key derived from `pair_id`, distribute it round-robin across the
  three outer folds, and record the algorithm version in the manifest.
- File names inside every fold are exactly `train.jsonl`, `val.jsonl`, and
  `test-heldout.jsonl`.
- The internal serialized split names are `oof_train`, `oof_val`, and
  `oof_test_heldout`. `test-heldout` always means an OOF subset of the WDC
  training split; it is never the official benchmark test set.

The 20% inner-validation rule and WDC-Qwen detector designation are frozen.
Changing either after artifacts are generated requires a new named protocol
rather than overwriting these artifacts.

## Gold and Split Boundary

The preparer and verifier may read only:

- the frozen blinded WDC training input JSONL used by the teacher;
- the majority-label CSV and its Phase 3 summary;
- the Qwen student config when rendering commands.

They must not open the WDC human-label training target, official validation, or
official test files. All three generated files carry majority teacher labels,
including a numeric `label` field because the existing evaluator requires one.
Any held-out metrics produced by that evaluator measure agreement with the
majority teacher label only and are not thesis gold metrics.

## Artifact Contract

Write ignored generated data under:

```text
data/cache/wdc_products/oof/majority-3fold/
  fold_assignments.csv
  oof_manifest.json
  fold_01/
    train.jsonl
    val.jsonl
    test-heldout.jsonl
    manifest.json
  fold_02/...
  fold_03/...
```

Every JSONL row preserves `pair_id` and `input_text` from the frozen blinded
input and adds only the fields needed by the existing trainer/evaluator:
`dataset_id`, `split`, `label_source`, `target_text`, and numeric `label`.
`label_source` is `llm_majority`; no teacher pass details or human labels are
embedded in these rows.

The root manifest records source hashes, assignment algorithm/version, class
counts, per-file counts and SHA-256 hashes, and the exact pair-ID partitions.
These verified fold inputs and manifests are committed because they are the
frozen inputs to GPU training and are small enough for normal Git storage. Real
predictions, checkpoints, packages, and other GPU outputs remain ignored.

## Files

- Create `supervision/prepare_oof_folds.py` for deterministic preparation and
  independent `--verify-only` rederivation.
- Create `experiments/wdc_qwen_oof.py` for CPU preflight, fold-result
  verification, and final merge logic.
- Create `scripts/run_wdc_qwen_oof.sh` as the small command dispatcher for
  `prepare`, `verify-data`, `plan`, `preflight`, per-fold training, per-fold
  verification, merge, and packaging.
- Create `tests/test_wdc_qwen_oof.py` with fixtures for preparation,
  verification, command rendering, result rejection, and merging.
- Add a narrow `.gitignore` exception for the frozen OOF input directory while
  keeping teacher evidence, model outputs, and checkpoints ignored.
- Update `../AGENTS.md`, `AGENTS.md`, and `CLAUDE.md` together only after the
  workflow exists and the real CPU preparation check passes.

## Implementation Steps

### 1. Implement deterministic fold preparation

- Strictly align blinded pair IDs with `llm_unrefined.csv`.
- Reject duplicates, missing/extra IDs, invalid labels, reordered or altered
  input text, and non-training source rows before publishing anything.
- Create class-aware outer folds using the frozen hash-and-round-robin rule.
- For each outer fold, form `test-heldout.jsonl` from that fold and derive
  `train.jsonl` plus the 20% `val.jsonl` from the remaining two folds.
- Publish through a temporary directory followed by atomic replacement so a
  failed run cannot leave a plausible partial dataset.

### 2. Add independent verification

- Recompute expected partitions and serialized bytes from the two permitted
  source artifacts rather than trusting the published manifest.
- Require pairwise disjoint train/val/held-out sets within each fold.
- Require each pair to appear in exactly one outer `test-heldout.jsonl` and in
  neither train nor val for that fold.
- Require merged held-out coverage of exactly 2,500 unique source pair IDs,
  with preserved class totals of 495 `match` and 2,005 `non_match`.
- Require the three outer held-out partitions to contain 165 matches each and
  669/668/668 non-matches, hence 834/833/833 total pairs in deterministic fold
  order.
- Reject symlink aliases and any resolved path that points at official
  validation, official test, or human-label target artifacts.

### 3. Add the OOF preflight and command renderer

- Reuse `experiments.train_student` and `experiments.evaluate_student` without
  changing their path interfaces.
- Render one train command and one held-out evaluation command per fold using
  the existing WDC-Qwen hyperparameters.
- Set evaluator split metadata to `oof_test_heldout` and use fold-specific
  output directories so results cannot overwrite one another.
- Start each fold from the same configured base model. Never warm-start from
  the completed full-WDC checkpoint or carry a checkpoint/optimizer state from
  one fold into another, because those states have seen another fold's held-out
  examples.
- Calculate and record fold-specific example counts, optimizer steps per epoch,
  planned maximum steps, and warmup steps after preparation; do not copy WDC
  full-data step counts.
- Make all GPU actions require a separate explicit confirmation flag. `plan`
  and CPU preflight must never import CUDA-only dependencies or start training.

### 4. Implement verification and merge contracts with fixtures

- Define one expected prediction file per fold using the current evaluator's
  `pair_id`, `prediction`, `non_match_probability`, and `match_probability`.
- Verify unique expected IDs, finite probabilities in `[0,1]`, probability
  sums within tolerance, valid predictions, fold identity, and complete files.
- Merge only after all three fold outputs pass. Preserve canonical blinded-input
  order and write a zero-based `row_index`, because Confident Learning consumes
  positional label and probability arrays. The merged CSV schema is:

```text
row_index,pair_id,fold_id,label_source,given_label,given_label_id,
predicted_label,predicted_label_id,non_match_probability,
match_probability,assigned_label_probability
```

- Freeze the class mapping as `0 = non_match`, `1 = match`; the probability
  matrix column order is therefore
  `[non_match_probability, match_probability]`. `given_label_id` is the
  majority LLM label consumed later as CL's noisy/given label, while the two
  probability columns are the out-of-sample `pred_probs`.
- `assigned_label_probability` is a transparent convenience field equal to the
  probability assigned to `given_label_id`. The later CL phase may load inputs
  directly as:

```python
labels = rows["given_label_id"]
pred_probs = rows[["non_match_probability", "match_probability"]]
```

  Entropy, disagreement ranking, Confident Learning issue ranking, and label
  treatment remain later phases and are not computed here.

### 5. Test and materialize the CPU artifacts

- Cover deterministic bytes, uneven fold sizes, class preservation, leakage,
  duplicate/missing/extra IDs, tampered text/labels/manifests, partial
  predictions, invalid probabilities, and exact-once merge behavior.
- Run the focused OOF tests and the full repository suite.
- Generate the real WDC fold artifacts, run `--verify-only`, inspect the
  manifest/counts, and render all six future GPU commands.
- Stage the verified fold files and confirm that a clean Git clone receives all
  three directories and passes `verify-data` without regenerating them.
- Stop for researcher review. Do not continue into Phase 5 on the current
  CPU-only machine.

## Todo

- [x] Approve the WDC-Qwen detector; the inner-validation ratio is frozen at
      20% of each fold's non-held-out pool.
- [x] Implement the fold preparer and independent verifier.
- [x] Implement CPU preflight, result verification, and merge contract.
- [x] Implement the thin shell dispatcher with explicit GPU confirmation.
- [x] Add focused tests and run the complete CPU suite.
- [x] Materialize and inspect all three real WDC fold directories.
- [x] Commit the verified fold inputs and prove clean-clone verification.
- [x] Update durable workflow documentation with verified counts/commands.

## Completion Evidence

Completed on 2026-09-30 in commit `ee18f96`.

- The committed fold bundle contains 2,500 unique majority-labeled WDC pairs:
  495 `match` and 2,005 `non_match`.
- Fold 1 contains 1,333 train, 333 validation, and 834 held-out rows.
  Folds 2 and 3 each contain 1,334 train, 333 validation, and 833 held-out
  rows.
- Focused OOF tests pass 7/7, the repository suite passes 193/193, and
  labeler-screening passes 13/13.
- Shell syntax and Python compilation checks pass.
- An isolated archive of the committed tree passed `verify-data` without
  regenerating folds or reading human training labels, official validation,
  or official test data.
- No GPU training or real OOF prediction was run. Those actions remain Phase
  5 and require the explicit `--confirm-oof-training` flag on a CUDA machine.

## Success Criteria

- Three independently verifiable fold directories exist with the requested
  `train.jsonl`, `val.jsonl`, and `test-heldout.jsonl` names.
- Within every fold, the three partitions are disjoint and the held-out IDs are
  absent from training and checkpoint-selection data.
- Across folds, the held-out union is exactly the 2,500 majority-labeled WDC
  training pairs, each exactly once, with no gold/official-evaluation access.
- Re-running preparation produces byte-identical files and manifests.
- A clean clone contains every required fold input and can revalidate the
  committed hashes, counts, partitions, and schema without rerunning the
  preparer or obtaining hidden-label files.
- Fixture results prove that partial, duplicated, misassigned, malformed, or
  tampered predictions cannot be merged.
- The existing trainer/evaluator are reused through dynamic paths; no model
  architecture change or real GPU training occurs in this phase.
