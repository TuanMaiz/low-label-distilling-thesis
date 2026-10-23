# q05 — Why not use the judge panel to label the training data itself?

**Question.** If a panel of different LLMs (for example one GPT, one Claude,
one Kimi) judges better than one model, why not use that panel as the teacher
for all training pairs instead of only re-judging the uncertain ones?

**Likely follow-up.** A panel teacher might produce fewer wrong labels at the
source, so the whole detect-and-treat step would be unnecessary.

**Our answer.** `TODO` — draft points:
1. *Cost.* A panel multiplies the labelling cost by the number of models, and
   the judge prompt is reasoning-heavy (see q01: about USD 0.019 per pair
   against about USD 0.001 for a plain teacher label).
2. *Selective design.* Teacher labels are already good on most pairs; spending
   the expensive panel only where cheap signals disagree is the cascade
   argument. Needs the agreed-pair audit (q01) as evidence.
3. *Separation of roles.* The teacher labels; the panel adjudicates a small
   flagged set. Keeping them separate lets us measure what treatment adds.

**Evidence.** TODO: panel cost per pair once models are chosen; agreed-pair
audit result.

**Weak point.** Without a run of a panel teacher on all pairs (or a sample), we
cannot show that the selective version loses little.

**Status.** draft
