---
title: "DBLP-ACM Integration Verification and Handoff"
date: 2026-09-03
status: verified-offline-handoff
branch: refactor/full-label-er-migration
working_tree_base_commit: f93b74a91d89b9c9148a145a6e16223df57ed185
---

# DBLP-ACM Integration Verification and Handoff

## Summary

The DBLP-ACM integration is source-verified, dataset-frozen, deterministically
prepared, and offline-pipeline-ready. All verification ran on CPU in WSL2 with
Python 3.13.7. No network download, paid LLM call, CUDA/model load, compact-model
training, DBLP validation prediction, or test materialization/evaluation
occurred.

One cross-phase defect was found and fixed: preparation `--verify-only`
previously treated the legitimate downstream `teacher_labels/` directory as
dataset-publication corruption. It now verifies only preparation-owned root and
`serialized/` files, continues to reject drift there, and leaves independent
downstream artifact directories untouched.

## Status Vocabulary

| State | Result | Meaning |
|---|---|---|
| Fixture ready | Yes | Hostile, lifecycle, relocation, fake-client, and publisher fixtures pass. |
| Source verified | Yes | Fresh inspector output exactly matches the committed observation. |
| Dataset frozen | Yes | Source/profile/normalization/identity/split contracts are approved and reproducible. |
| Production labels produced | No | Only deterministic fake labels exist; API calls and cost are zero. |
| Production targets published | No | `full_label_targets_fake_phase5/` is explicitly synthetic evidence only. |
| Experiment completed | No | No DBLP model was trained and no DBLP validation/test metric exists. |

## Source and Preparation Evidence

- Archive SHA-256:
  `a15b752ffc318a714690cf13286d31c2012f686525803ca803c392ceff4aa4f3`.
- Fresh observation SHA-256 equals the committed observation:
  `05fb8face80c180f301a0a8c2757455417058bf925f542479a18acda1f55120b`.
- DBLP table: 2,616 records; no missing values in the frozen columns.
- ACM table: 2,294 records; 14 missing `authors`; other frozen columns complete.
- Train: 7,417 pairs = 1,332 match + 6,085 non-match.
- Validation: 2,473 pairs = 444 match + 2,029 non-match.
- Locked test contract: 2,473 source rows; not materialized.
- Canonical train/validation pair overlap: 0.
- Allowed record reuse across train/validation: 1,058 DBLP and 1,027 ACM IDs.

Two independent temporary workspaces produced byte-identical artifacts:

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `f60e7fb32ff149e177088f55d9deff03f8d3cec7acc1998ec257482462135296` |
| `stats.json` | `5efd9c03769173cc5c71b6b27e2220bee2eaa8f444b4ad0e8d92eb170b43a213` |
| `serialized/train.jsonl` | `1b6c9c6d50bfa7b43ebff357295b98a6f31af827bb7fbf79a847aebc84429080` |
| `serialized/validation.jsonl` | `b2f81211017a8ae26599694c424c35b73e1ecb4fa13684f895c7942a40cb6257` |

The real cache passes `--verify-only`. No DBLP `test.jsonl` exists.

## Supervision and Target Integration

The completed fake run contains 7,417 blinded inputs, 7,417 deterministic fake
dispatches, zero API calls, zero cost, and `paid_execution_authorized=false`.
Tests inspect the actual fake-client payload and prove that the provider messages
contain only the frozen instruction and `input_text`; `pair_id`, truth, and
metadata remain local.

The unchanged publisher consumed those completed artifacts and emitted a
7,417-row synthetic gold/LLM-hard bundle. The independent validator rederived
all five outputs byte-for-byte. This bundle is located at
`data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/full_label_targets_fake_phase5/`
and must never be treated as production supervision. Its 3,626 disagreements
are behavior of the deterministic fake client, not an accuracy result.

The interrupted replay used during verification correctly left an unresolved
inflight journal and refused automatic retry. That incomplete staging directory
was moved intact to `/tmp/dblp-phase5-interrupted-s3zPz9/` for recovery; the
previous completed fake run was not changed.

## Qwen Readiness

- DBLP instruction is the only model-config difference from frozen WDC Qwen.
- Schedule: 464 optimizer steps/epoch, 4,640 planned steps, 464 warmup steps.
- Portable identity, canonical pair separation, target-manifest binding,
  result corruption, symlink, relocation, and gold-first package fixtures pass.
- `preflight` exits before model access because production targets are absent.
- Confirmed `train-gold` exits before CUDA/output mutation because GPU execution
  is unauthorized.

## WDC Compatibility

- Fresh WDC train/validation/test serialization is byte-identical to the
  committed cache.
- 26 protected tracked WDC files have Git blob mismatch count 0.
- Protected SHA-256 values remain:
  - preflight: `9eb370013cd169997c3ebaa905e6151cf39e3917b59d605d012cfd359bb8e2da`
  - runner: `e74af0f7650e5f523678a5e826559b271dcf9c324f67b85a5846d681876e0c40`
  - Qwen config: `bc0a20186bafef993f6516d34bfde245c9798429ef5bcfc2a34ce0b0612bc02e`
  - gold target: `f4e4120404ccd83295b788706bbf1afeb1b2a26757ab53b50568d2428159424c`
  - LLM-hard target: `7f45401de90157d11da689c7efd26b2f0934124b953864f8a014694d98f1cdc0`
- Downloaded combined WDC package and both arm archives pass their SHA-256 files.

## Test Results

| Suite | Result |
|---|---|
| Four focused DBLP modules | 41/41 passed |
| WDC serialization + contract/alignment/Qwen checks | 30/30 passed |
| Focused WDC Qwen module | 21/21 passed |
| Full repository | 181/181 passed |
| Labeler screening | 12/12 passed |

Hostile tests cover origin rejection before secret resolution, redirects,
unsafe output paths, symlink aliases, forged caches/manifests, malformed
responses, missing predictions, unresolved inflight state, result corruption,
and lifecycle ordering.

## Reproduction Commands

```bash
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

.venv/bin/python -m unittest \
  tests.test_dblp_acm_loader \
  tests.test_dataset_preparation \
  tests.test_dataset_aware_supervision \
  tests.test_dblp_acm_qwen_preflight -v
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s labeller-screening/tests -v

bash scripts/run_dblp_acm_qwen_vertical_slice.sh config
bash scripts/run_dblp_acm_qwen_vertical_slice.sh identity
bash scripts/run_dblp_acm_qwen_vertical_slice.sh plan
```

Two independent real-source preparations and byte comparison used this pattern:

```bash
ws_one=$(mktemp -d /tmp/dblp-phase5-one-XXXXXX)
ws_two=$(mktemp -d /tmp/dblp-phase5-two-XXXXXX)
for ws in "$ws_one" "$ws_two"; do
  mkdir -p "$ws/configs/datasets/observations" \
    "$ws/data/raw/dblp_acm/archive-2026-09-01/exp_data" \
    "$ws/data/cache/dblp_acm"
  cp configs/datasets/dblp_acm.json "$ws/configs/datasets/dblp_acm.json"
  cp configs/datasets/observations/dblp_acm_2018_06_29_a15b752f.json \
    "$ws/configs/datasets/observations/dblp_acm_2018_06_29_a15b752f.json"
  for source_file in tableA.csv tableB.csv train.csv valid.csv test.csv; do
    cp "data/raw/dblp_acm/archive-2026-09-01/exp_data/$source_file" \
      "$ws/data/raw/dblp_acm/archive-2026-09-01/exp_data/$source_file"
  done
  .venv/bin/python -m data.prepare_benchmark \
    --dataset-config "$ws/configs/datasets/dblp_acm.json" \
    --source-root "$ws/data/raw/dblp_acm/archive-2026-09-01/exp_data" \
    --output-root "$ws/data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f"
done
cmp -s "$ws_one/data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/serialized/train.jsonl" \
  "$ws_two/data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f/serialized/train.jsonl"
# Repeat cmp for validation.jsonl, stats.json, and manifest.json.
```

The completed fake artifacts were connected to the unchanged publisher and
independent validator with:

```bash
cache_root=data/cache/dblp_acm/deepmatcher-structured-dblp-acm-2018-06-29-a15b752f
fake_root="$cache_root/teacher_labels/fake_sol_high_phase3"
.venv/bin/python -m supervision.build_full_label_targets \
  --pairs "$cache_root/serialized/train.jsonl" \
  --predictions "$fake_root/predictions.csv" \
  --attempts "$fake_root/attempts.jsonl" --audit "$fake_root/audit.jsonl" \
  --labeler-run "$fake_root/run.json" --blinded-inputs "$fake_root/inputs.jsonl" \
  --blinded-input-manifest "$fake_root/inputs.manifest.json" \
  --labeler-settings "$fake_root/settings.json" \
  --output-dir "$cache_root/full_label_targets_fake_phase5" \
  --dataset-id dblp_acm \
  --dataset-version deepmatcher-structured-dblp-acm-2018-06-29-a15b752f \
  --expected-count 7417
.venv/bin/python -m supervision.validate_full_label_targets \
  --target-dir "$cache_root/full_label_targets_fake_phase5"
```

WDC regeneration, protected-blob, and downloaded-package verification commands:

```bash
.venv/bin/python -m unittest tests.test_wdc_serialization_compatibility -v
git ls-files experiments/wdc_qwen_preflight.py \
  scripts/run_wdc_qwen_vertical_slice.sh \
  configs/students/qwen3_reranker_0_6b.json \
  labeller-screening data/cache/wdc_products > /tmp/wdc-protected.txt
while IFS= read -r path; do
  test "$(git hash-object "$path")" = "$(git rev-parse "HEAD:$path")"
done < /tmp/wdc-protected.txt
(cd outputs/new && sha256sum -c wdc-qwen-gold-vs-llm-hard.tar.gz.sha256)
(cd outputs/new/wdc-qwen-gold-vs-llm-hard && \
  sha256sum -c gold.tar.gz.sha256 && sha256sum -c llm_hard.tar.gz.sha256)
```

## Remaining Manual Gates and Sequence

The source, attribution handling, acquisition, and deterministic preparation are
already approved and frozen. The next work must remain separately authorized:

1. Finish the broader experiment contract decisions that still gate non-WDC
   production cells.
2. Review current OpenRouter pricing, freeze a DBLP spend ceiling, and explicitly
   authorize the 7,417-row GPT-5.6 Sol-high production labeling run.
3. Reconcile every inflight request and require 100% valid unique production
   labels before publication.
4. Publish and independently validate the real `gold` and `llm_hard` targets.
5. Review the tokenizer input-length audit, flip the separately reviewed target
   and GPU guards, then authorize the rented-GPU validation slice.
6. Keep the official test split locked until the final global evaluation gate.

None of these steps is authorized or performed by this handoff.
