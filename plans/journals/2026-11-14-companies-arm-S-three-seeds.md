# 2026-11-14 — Companies arms A, G, B, S on one card; S over three seeds

Branch `feat/soft-target-arm`. Plan 261106, amendment A1 (draft, not yet signed
off): arm S, B without diagnoses, Colab GPU, seeds 43/44 for A, S and G.
Decision D26 revised today: one card per comparison set, card recorded per run.
Companies records are not republished; outputs are git-ignored.

## Runs

- **OOF (2026-11-13):** 3 folds in parallel on a Colab L4 (OOF only selects
  pairs and is not compared). Merged locally: 1,968 rows.
- **Detection:** 88 pairs flagged (teacher 2/3: 16; OOF disagreement: 79;
  overlap 7). Evidence check for C at this count: about USD 1.85.
- **Arms (today):** A, G, S at seeds 42/43/44 and B at seed 42 — 10 runs in
  parallel on one Colab **NVIDIA RTX PRO 6000 Blackwell** (A100 unavailable).
  fp16, recipe unchanged (batch 1 × accumulation 16, 10 epochs, patience 3).
  GPU utilisation stayed about 83% from 8 to 10 processes (time-slicing between
  processes), so 10 was the useful limit.
- **B and S bundles** were built on Colab; a local rebuild gives the same
  training-file hashes (B `c6cec741…`, S `ddc0583e…`). B trains on 1,880 rows.
- **Incident:** the first test step failed for A, G and S seed 44 — the test set
  is materialised from the raw MaDI-Bench CSVs at test time and those were not
  uploaded. After uploading them, the test step was resumed from the trained
  checkpoints (no retraining, no run tested twice).
- Results: `outputs/companies-arms-all.zip`, unpacked in
  `outputs/uncertainty_refinement/madi-companies-qwen-g4/` (the V100 A and G
  results stay in `madi-companies-qwen/`). All 10 checksums and contents
  verified; merged models excluded, adapters kept.

## Match F1 per run

| Run | Test (599) | Validation (948) |
|---|---|---|
| A seed 42 / 43 / 44 | 88.3 / 89.9 / 88.3 | 93.9 / 93.3 / 93.2 |
| G seed 42 / 43 / 44 | 90.4 / 90.4 / 88.8 | 95.0 / 94.6 / 94.6 |
| S seed 42 / 43 / 44 | 90.3 / 91.1 / 88.5 | 93.7 / 93.7 / 93.6 |
| B seed 42 | 90.3 | 93.7 |

Mean ± sd over seeds, test: A 88.8 ± 0.9, S 90.0 ± 1.3, G 89.9 ± 0.9.
Validation: A 93.5 ± 0.4, S 93.7 ± 0.1, G 94.7 ± 0.3.

## Paired comparisons (paired bootstrap 2,000, seed 0; exact McNemar)

Per seed, test:

| | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| S − A | +2.0 [+0.1, +3.9], p 0.04 | +1.2 [−0.2, +2.7], p 0.18 | +0.2 [−1.4, +2.0], p 1.00 |
| S − G | −0.2, p 1.00 | +0.7, p 0.63 | −0.3, p 1.00 |
| A − G | −2.1, p 0.06 | −0.5, p 0.80 | −0.5, p 0.80 |

Seed 42 only: B − A +2.0 [+0.2, +3.8], p 0.04; S − B −0.1, p 1.00.

Pooled over seeds (resample pairs, average ΔF1 over the three seeds):

| | Test | Validation |
|---|---|---|
| S − A | **+1.12 [+0.02, +2.45], p 0.04** | +0.21 [−0.54, +0.98], p 0.59 |
| S − G | +0.08 [−1.55, +1.79], p 0.92 | −1.05 [−2.36, +0.22], p 0.10 |
| A − G | −1.05 [−2.59, +0.35], p 0.15 | **−1.26 [−2.40, −0.22], p 0.02** |

## Reading

- The teacher-to-gold gap on Companies is real: A < G in all three seeds; on
  validation it is significant (−1.26).
- On test, S closes it: S > A in all three seeds, pooled +1.1 (p 0.04), about
  the size of the A–G gap; S is level with G (+0.1). S does not surpass G
  (researcher: closing the gap is the goal, not surpassing gold).
- On validation, S recovers only a small part of the gap (+0.2 of 1.26) and
  stays about 1 point below G. Validation is used for checkpoint and threshold
  selection, so it is not fully independent; test is a single 599-pair set.
- B ties S at seed 42 on Companies, unlike WDC, where B lost to A. Whether S's
  gain comes from soft labelling or from not trusting the flagged pairs is open.
- With WDC (no gap, S did no harm), the working framing holds: S closes the
  teacher-to-gold gap where one exists and does no harm where none does.
- Proposed thesis wording: "On Companies, soft targets closed the LLM-to-gold
  gap on the held-out test set (+1.1 F1 over A across 3 seeds, p 0.04; level
  with gold), while on validation the recovery was small (+0.2)."

## Open

- Sign-off of Companies amendment A1 (still draft).
- Delete `thesis-private/` from Drive and empty Trash (D26).
- Local commits not yet pushed.

## Next

1. Geo-ER: G and A first (is there a gap?), then S.
2. B seeds 43/44 on Companies (S vs B).
3. Arm C on Companies: evidence check (about USD 1.85), then training.
