# Contract — Companies treatment arms (plan 261106)

Mirrors `plans/261023-2015-wdc-guideline-judge-treatment/contract.md`. Anything
not stated here follows that contract. Status: **draft, 2026-11-06.**

## 1. Inputs

| Input | Path |
|---|---|
| Gold-free teacher inputs (1,968) | `data/cache/madi_companies/madi-bench-companies-base-062058d/teacher_labels/inputs.jsonl` |
| Majority labels | `.../majority_labels/llm_unrefined.csv`, `.../consistency.csv` |
| Folds | `.../oof/majority-3fold` (`prepare_oof_folds`, 20% inner validation) |
| Guideline | `supervision/judge_guideline.md` v3, SHA-256 `b2745dac96531a23bdf4c818e900acb322b5b3db2177936193ff44896e01504a` (unchanged) |
| Evidence prompt | `supervision/evidence_case_prompt_generic.md` (D25, signed off 2026-11-06, SHA-256 `48b2c4615336f09dc6240a04a8d71182b32d81c8306dd10048f8adb8dce8ec2c`), `{FIELDS}` = the eight company fields. Runner `supervision/run_evidence_check_v2.py --dataset madi_companies`. |
| Teacher settings | `configs/labelers/evidence_check_sol_high_flex.json` (Sol high, flex tier, max price 2/10; shared by all datasets) |

## 2. Detection

Teacher 2/3 agreement OR OOF teacher–student disagreement (the WDC rule), from
the merged OOF predictions of three folds.

## 3. Category rule

WDC section 2 unchanged, with one dataset setting: at step 2 a conflict counts
only with a quote outside the supporting fields on both sides. Supporting
fields for Companies: `assets`, `revenue`, `keypeople` (WDC: `price`,
`priceCurrency`). Invalid diagnoses: pair kept unchanged.

## 4. Action rule and arms

WDC sections 3 and 4 unchanged: A all 1,968 pairs with teacher labels (already
trained), B = A minus detected pairs, C = action rule applied (remove /
upweight with `#dup1` / flip).

## 5. Training recipe

As the Companies arms (`scripts/run_madi_companies_arms.sh`): WDC recipe,
`configs/students/qwen3_reranker_0_6b_madi_companies.json`, Tesla
V100-SXM2-32GB, fp16, seed 42. OOF folds use the same recipe with the fold's
inner validation. Validation: official Companies validation (948).

## 6. Metrics and tests

Match F1 primary, pooled and per source pair. Paired bootstrap (2,000, seed 0)
and exact McNemar for C vs A, C vs B, B vs A on validation, then test once for
the frozen arms. A and G are already frozen; B and C are frozen before any of
their test results exist. B and C use the arm runner with B/C routing added
after A and G ran; the A/G code path and commands are byte-identical (checked).

## 7. Gold boundary

| Gold | Use |
|---|---|
| Train gold | Final analysis only |
| Validation gold | Checkpoint and threshold selection; paired comparison |
| Test gold | Once, frozen arms |

**Disclosure.** Companies train gold was already inspected on 2026-11-05 for the
label-quality analysis (journal `2026-11-05-companies-paired-test-and-label-quality`).
The evidence prompt is generic and has no dataset-specific part, so its wording
is not tailored to Companies. The one Companies-specific setting chosen afterwards is the supporting
fields (section 3); no pair-level gold enters any prompt, selection or rule.

## 8. Costs and stop rules

- Evidence check: about USD 0.02 per pair (WDC: 158 pairs, USD 3.25). The count
  is known after OOF (`prepare`). Stop and ask if more than 400 pairs are selected.
- GPU: 3 folds + 2 arms on the V100.

## 9. Open decisions (researcher)

1. Prompt: decided and signed off 2026-11-06 (D25). Consequence for WDC: its evidence check is rerun with
   the generic prompt (about USD 3.3, flex tier), the C bundle is rebuilt and
   arm C retrained and tested once as a new frozen arm (C v2), with an amendment
   to the WDC contract (plan 261023). WDC A, B, G, OOF and the 158 detected
   pairs are unchanged.
2. Supporting fields: `assets`, `revenue`, `keypeople`.
3. Whether to run at all, given the caveat in the plan (guideline and teacher
   share the strict "same specific entity" view on related companies).

## Amendments

None yet.
