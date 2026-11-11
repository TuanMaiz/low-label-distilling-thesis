# Contract — Geo-ER treatment arms (plan 261111)

Same as `plans/261106-1330-companies-treatment-arms/contract.md` (which mirrors
the WDC contract, plan 261023) except where stated. Status: **draft, 2026-11-11.**

| Item | Geo-ER |
|---|---|
| Inputs | `data/cache/geo_er/geo-er-osm-fsq-pit-83063f7/teacher_labels/inputs.jsonl` (2,500), majority labels and consistency in `majority_labels/` |
| Folds | `.../oof/majority-3fold`, `prepare_oof_folds`, 20% inner validation |
| Evidence prompt | D25 prompt (SHA-256 `48b2c4615336f09dc6240a04a8d71182b32d81c8306dd10048f8adb8dce8ec2c`), `{FIELDS}` = `name/latitude/longitude/address/postalCode` |
| Teacher settings | `configs/labelers/evidence_check_sol_high_flex.json` |
| Supporting fields (rule step 2) | `latitude`, `longitude` |
| Training recipe | `scripts/run_geo_er_arms.sh`: WDC recipe, `qwen3_reranker_0_6b_geo_er_pit.json`, V100-SXM2-32GB, fp16, seed 42 — the same GPU and precision for G, A, OOF, B and C |
| Validation / test | official 999 / 1,502; test once per frozen arm |

## Notes

- `distance_m` is part of every input (D24) and is sent with the records, but it
  is a computed pair line, not a record field, so it cannot be quoted as evidence.
- **Disclosure.** Geo-ER train gold was inspected on 2026-11-05 (label-quality
  analysis: the teacher's extra matches are often nearby places). The choice of
  supporting fields was made afterwards; its reason does not depend on labels
  (two sources never give identical coordinates, and distance is given separately).

## Open decisions (researcher)

1. Supporting fields: `latitude`, `longitude` (postalCode stays an identity attribute).
2. Whether to add the free label audit before training B and C as a gate (all datasets).

## Amendments

None yet.
