# 2026-11-12 — WDC arm S (soft targets): first exploratory run

Branch `feat/soft-target-arm`. Plan 261112-1710-soft-target-arm; WDC contract
amendment A6 (β = 0.5, seeds 43/44 not yet signed off). WDC is the development
set; this run guides direction and confirms nothing.

## What was built today

- Trainer: `models/generative_reranker_student.py` accepts an optional
  `soft_target` per row. The batch target becomes a probability pair
  [1 − s, s] (hard rows become one-hot); the loss is unchanged
  (`F.cross_entropy` with class-probability targets). The prompt and the
  "yes"/"no" target text are unchanged.
- Data: `supervision/build_soft_target_dataset.py` builds from arm A's
  `train.jsonl` (hash-checked). Only the 158 detected pairs (teacher 2/3 OR
  out-of-fold disagreement) get `soft_target = 0.5 · teacher + 0.5 · p_OOF(match)`;
  the other 2,342 keep their hard label. No API call, no gold.
  Bundle: `data/cache/wdc_products/refinement/soft-target-S/`.
- Runner: `scripts/run_wdc_qwen_treatment_arms.sh` accepts arm S.
- D26: remaining runs on Colab A100, fp16, same recipe; compared arms on the
  same card.

## Run

Colab, **Tesla T4** for training and test (`preflight/runtime.json`), not the
A100 named in D26. Seed 42, fp16, batch 1 × accumulation 16, 10 epochs,
patience 3. Early stop after 7 epochs; best epoch 4 (validation macro F1 88.1,
match F1 81.3, threshold 0.223). About 18 min per epoch on the T4.

Results: `outputs/arm-S-explore.zip`, unpacked in `outputs/arm-S-explore/`
(git-ignored); Drive `thesis-colab/arm-S-explore/` keeps the merged model.

## Test, match F1: S against the earlier arms (paired bootstrap 2,000, seed 0; exact McNemar)

| S vs | S | Other | Δ | 95% CI | Only S right / only other right | p |
|---|---|---|---|---|---|---|
| A | 76.9 | 76.5 | +0.4 | [−1.5, +2.2] | 76 / 65 | 0.40 |
| G | 76.9 | 75.9 | +1.0 | [−1.0, +3.0] | 94 / 76 | 0.19 |
| B | 76.9 | 75.1 | +1.8 | [−0.1, +3.6] | 91 / 57 | 0.01 |
| C | 76.9 | 74.8 | +2.1 | [+0.2, +4.0] | 87 / 54 | 0.01 |

S test: precision 68.2, recall 88.2 (TP 441, FP 206, FN 59).

## Reading

- S is the first treatment that does not fall below A (B and C both did).
- Keeping detected pairs with a soft label beats removing them (B) or acting on
  them with hard rules (C), p = 0.01 for both.
- S vs A and S vs G are within noise. On WDC A is already ≥ G, so there is no
  teacher-to-gold gap to close; "no harm" is the most WDC can show.
- Researcher's expectation: S lands between A and G. That applies where G > A:
  Companies (G 89.7 vs A 88.8). G is not a hard ceiling, but exceeding it is
  not expected with 3 votes + one OOF probability.
- Working framing: S closes the teacher-to-gold gap where one exists and does
  no harm where none does.

## Caveats

1. Card: S ran on a T4; A, G, B, C ran on the V100. Every row mixes treatment
   and card.
2. One seed (42).
3. WDC is the development set.

## Discussed, not decided

- Rerunning A on the A100 now was dropped for this exploratory run; a
  same-card A is needed for the confirmatory comparison (with seeds 43/44).
  D26 still says "rerun A once on A100"; wording not yet updated.
- Speed: batch 1 × accumulation 16 leaves the GPU mostly idle. Micro-batch 8 ×
  accumulation 2 keeps the effective batch at 16 and should be several times
  faster, but the trainer must first be checked for padding side (score read at
  the last real token) and per-pair loss averaging, and it is a recipe change
  (new decision; all compared arms rerun under it).

## Next

1. Companies arm S: build its soft-target bundle from the Companies OOF
   predictions (free), then train — the only dataset with a known A < G gap.
2. Geo-ER G and A, to see whether there is a gap there.
3. Confirmatory WDC: A and S on the same card, seeds 43/44 (needs A6 sign-off).
4. Optional before those: the batch-size check above.
