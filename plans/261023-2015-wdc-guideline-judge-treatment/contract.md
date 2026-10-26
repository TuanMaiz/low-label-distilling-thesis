# Contract — wdc-guideline-judge-treatment (version 1, category rule)

Frozen: 2026-10-24, researcher. Any change after this date is a dated amendment
at the end of this file. Any change after Phase 3 training starts also needs a
new arm name.

## 1. Inputs (SHA-256)

| File | SHA-256 |
|---|---|
| `supervision/judge_guideline.md` (v3, signed off) | `b2745dac96531a23bdf4c818e900acb322b5b3db2177936193ff44896e01504a` |
| `outputs/uncertainty_refinement/wdc-evidence-check/diagnoses.csv` | `2b94014ab4318e00bf7308925eb7a46faed518dc04c0a0c669effcfd19eb7dd5` |
| `outputs/uncertainty_refinement/wdc-evidence-check/selected_pairs.json` | `a11f1e37e89671d2fef24551086789cc879a9a4583d23f95fbaba6ff743a4800` |
| `data/cache/wdc_products/majority_labels/llm_unrefined.csv` | `b4dc5050e125e930c6b4b99b40004acfafedc9e9ba3fd22c608c5c3d0128f4c1` |
| `data/cache/wdc_products/teacher_labels/full_sol_high/wdc_train_full.inputs.jsonl` | `e443fc38fe1206ce961c6f71dce28e50b4e148d720dda3b2bdc688abf196e1ea` |
| `configs/students/qwen3_reranker_0_6b.json` | `bc0a20186bafef993f6516d34bfde245c9798429ef5bcfc2a34ce0b0612bc02e` |
| `scripts/run_wdc_qwen_variant_prompt.sh` (recipe source) | `bf5fc0e285cbddfad1cb4e4c6b8fc2bec606f7bb0bffc1051481246646ed2d2d` |

Detection set: the 158 pairs in `selected_pairs.json` (149 OOF teacher–student
disagreements ∪ 13 teacher `2/3`, 4 overlap). Teacher label = three-pass
majority in `llm_unrefined.csv`.

## 2. Category rule (guideline steps applied to saved diagnoses)

Applied in order to each diagnosis with `status = categorized`; the first rule
that fires sets the verdict.

| Step | Rule | Verdict |
|---|---|---|
| 1 | any case `intra_record_contradiction` | 0 |
| 2 | any case `cross_record_conflict` with at least one evidence quote whose field is not `price` or `priceCurrency` | 2 |
| 3 | any case `ambiguous_variant_bundle` or `ambiguous_role_scope` | 0 |
| 6 | otherwise | 1 |

- Step 4 (one-sided information, option B) and step 5 (noise) never block.
- A price-only conflict does not count at step 2 (supporting information alone
  never decides non-match).
- Diagnoses with `status` other than `categorized` (3 invalid): no verdict,
  pair kept unchanged.
- No price threshold (guideline v3).

## 3. Action rule

| Verdict | Teacher label | Action |
|---|---|---|
| 0 | any | remove |
| 1 or 2 | same as verdict | upweight: one duplicate row, `pair_id = "<id>#dup1"`, `duplicate_of = "<id>"` |
| 1 or 2 | different | flip to the verdict |

Pairs outside the 158 and the 3 invalid diagnoses: keep, teacher label.

## 4. Arms

| Arm | Training data |
|---|---|
| A | All 2,500 pairs, teacher labels |
| B | A minus the 158 detected pairs |
| C | A with the action rule applied |

Arm D (random upweight control) is not run in version 1.

## 5. Training recipe (every arm, same GPU type)

`experiments.train_student` with `configs/students/qwen3_reranker_0_6b.json`
(original instruction), `--batch-size 1 --validation-batch-size 1
--gradient-accumulation-steps 16 --num-epochs 10 --learning-rate 0.0002
--weight-decay 0.01 --warmup-steps 0 --warmup-ratio 0.1 --max-input-length 4096
--seed 42 --early-stopping-patience 3 --precision fp16 --device cuda`.
Validation: `data/cache/wdc_products/serialized/validation.jsonl`. Threshold by
`utils.classification_threshold.select_decision_threshold` on validation.

## 6. Metrics and tests

- Primary: match F1. Also precision, recall, macro F1, accuracy.
- Paired comparisons C vs A and C vs B on validation: bootstrap of ΔF1
  (2,000 resamples, seed 0, 95% percentile CI) and exact McNemar on per-pair
  correctness.
- Test: once, after explicit authorization, for the frozen arms only.

## 7. Gold boundary

| Gold | Allowed use |
|---|---|
| Train gold | Phase 4 final analysis only |
| Validation gold | Checkpoint and threshold selection; paired comparison |
| Test gold | Phase 4, once, frozen arms, explicit authorization |

The builder refuses `serialized/train.jsonl`, `full_label_targets/gold.jsonl`,
`serialized/validation.jsonl` and `serialized/test.jsonl`. WDC test already
influenced an earlier decision (D13) and is not a clean confirmation set.

## 8. Expected counts (preview, 2026-10-23)

C: 93 remove, 52 upweight (47 non-match, 5 match), 10 flip (match → non-match),
3 kept with invalid diagnoses. Phase 2 reports the built counts against this.

## Amendments

### A1 — 2026-10-24, clarification before any dataset was published

Section 2, step 2 reads "at least one evidence quote whose field is not price".
The builder requires a non-price quote on **both** sides of the conflict, as the
guideline does ("both records state values"). Checked on the saved diagnoses
before building: 0 pairs differ between the two readings, so no count changes.

### A2 — 2026-10-24, recipe source restored to the committed runner

Section 1 hashed an uncommitted working copy of
`scripts/run_wdc_qwen_variant_prompt.sh` (`bf5fc0e2…`) that had dropped the
setup, preflight, evaluate-test and package-results actions, which made 8 runner
tests fail. The researcher had it restored to the committed version
(`216ff31`), SHA-256
`32db42a9e7e7e5b878a99d7a002174d200a39de6a0dee62fb95083d29f6afde9`. The
training command flags in section 5 are identical in both versions (checked
by diff), so the recipe does not change. No dataset or result depends on this.

### A3 — 2026-10-26, GPU type fixed; precision stays fp16

All arms run on **Tesla V100-SXM2-32GB** (no native bf16) with **fp16**, as in
section 5 (researcher, 2026-10-26: cheapest option). The runner
`scripts/run_wdc_qwen_treatment_arms.sh` makes these the defaults; its
preflight refuses any other GPU name unless `EXPECTED_GPU` is set, which needs
a new amendment. The 2026-08-31 gold run (bf16, RTX 3090) is therefore not a
comparable reference; a gold reference arm, if added, must use this setup.
No training had started.
