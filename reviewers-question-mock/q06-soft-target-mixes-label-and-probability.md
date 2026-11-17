# q06 — Arm S mixes a hard label with a probability. Is that valid, and why does it favour the teacher?

**Question.** The soft target is `0.5 × teacher label + 0.5 × student p(match)`
on flagged pairs. The teacher label is 0 or 1; the student gives a probability.
They are not the same kind of quantity, so why average them? And the formula
leans toward the teacher. Is that justified?

**Likely follow-up.**
1. The student's p(match) is not calibrated: its decision threshold is far
   from 0.5. Some pairs are flagged because the student disagrees, yet the soft
   target barely moves. Why?
2. You have three teacher votes. Why use the teacher's 0/1 majority instead of
   its vote share?

**Our answer.**
1. *Same frame.* A hard label is a probability with no uncertainty: "match" is
   p(match) = 1, "non-match" is 0. Teacher and student are both distributions
   over {match, non-match}; the teacher's is maximally confident.
2. *Equivalent to splitting the loss.* Cross-entropy is linear in its target,
   so training on the mixed target equals
   `0.5 × CE(model, teacher label) + 0.5 × CE(model, student probability)`,
   the knowledge-distillation loss (Hinton et al. 2015, hard label plus soft
   model probabilities).
3. *Teacher first, by property.* The weights are equal; the teacher dominates
   because it is certain. With β = 0.5 the target never crosses 0.5 to the
   other side of the teacher's label: teacher match gives a target in
   [0.5, 1], teacher non-match gives [0, 0.5]. The student can lower the
   teacher's confidence on a pair, at most to 0.5, but never reverse its label.
4. *Why that is right here.*
   - Most flagged labels are correct: on WDC, 82% of the 158 flagged pairs
     have the correct teacher label (17.7% wrong vs 2.2% elsewhere).
   - Flipping failed: arm C made 10 flips and the teacher had been right on
     all 10; C lost to A on test (−1.7).
   - The student is a corrector trained on the same noisy labels; Fan & Li
     (TRACE, arXiv 2608.03432) show such pseudo-targets repeat the noise, so
     the student should signal doubt, not replace the label.
5. Proposed thesis wording: "S treats the teacher as the authority and the
   student as a source of doubt: the soft target stays on the teacher's side of
   0.5, so the student can reduce the teacher's confidence on a pair but never
   reverse its label."

**Evidence.**
- Formula and β (fixed before any run): `supervision/build_soft_target_dataset.py`;
  WDC contract amendment A6; Companies amendment A1.
- Flagged-pair label quality: WDC gold audit, plan 261023;
  `docs/research/literature-uncertain-label-treatment.md`.
- Results: WDC journal `plans/journals/2026-11-12-wdc-soft-target-arm-S-exploratory.md`
  (S 76.9 vs A 76.5, S beats B and C, p 0.01); Companies journal
  `plans/journals/2026-11-14-companies-arm-S-three-seeds.md` (test S − A +1.1,
  p 0.04 over 3 seeds; level with G).
- Flagged pairs by teacher vote and student side (student side read at 0.5):

  | Teacher votes | Student vs teacher (at 0.5) | WDC | Companies |
  |---|---|---|---|
  | 3/3 match | disagrees | 65 | 28 |
  | 3/3 non-match | disagrees | 72 | 35 |
  | 3/3 non-match | agrees | 8 | 9 |
  | 2/3 match | agrees / disagrees | 3 / 4 | 3 / 4 |
  | 2/3 non-match | agrees / disagrees | 6 / 0 | 5 / 4 |

**Weak point.**
- *Calibration (follow-up 1).* A pair is flagged by the student's own
  threshold (chosen on validation, e.g. 0.22 on WDC), but the soft target reads
  the raw p as if 0.5 were the dividing line. 17 pairs (WDC 8, Companies 9) are
  flagged for disagreement with p(match) < 0.5, so their target stays close to
  the teacher's hard label. Effect: weaker softening on those pairs; it cannot
  inflate S's results. Fix for a later, predeclared arm: calibrate the student
  (temperature or Platt scaling) before mixing.
- *Teacher vote share (follow-up 2).* Using the vote share (0, ⅓, ⅔, 1) would
  put the teacher on a probability scale too. It changes only the 2/3 pairs,
  by ⅙ toward 0.5 (WDC 13 of 158, Companies 16 of 88), so a small effect is
  expected. Candidate ablation, not run.
- *β not tuned.* β = 0.5 was fixed before any run. The "never flips" property
  is a property of that choice, shown desirable by later analysis; it was not
  the stated motivation when β was fixed.

**Status.** backed
