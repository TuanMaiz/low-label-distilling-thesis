# q01 — Why not run the evidence-based judge on every training pair?

**Question.** The guideline-based judge reads evidence and should label better
than the plain teacher. Why use it only on uncertain pairs instead of labelling
all of train with it?

**Likely follow-up.** "Reasoning is expensive" is vague. How much more, and is
the selective version actually losing anything?

**Our answer.**
1. *Cost per pair.* The evidence check is reasoning-heavy. It cost about
   USD 0.019 per pair (USD 3.25 for 170 received responses), against about
   USD 0.001 per pair for a plain teacher pass (about USD 2.7 per 2,500 pairs).
   That is roughly 18× more per pair: about USD 48 per pass on all 2,500
   pairs, or about USD 144 for three passes.
2. *Where it can matter.* On pairs where teacher and student agree, the judge
   has little room to change anything. Spending expensive reasoning only where
   cheap signals disagree is the usual cascade argument.
3. *Scaling.* The cost grows with dataset size, so the selective design is the
   one that transfers to larger datasets.

**Evidence.**
- Cost: `outputs/uncertainty_refinement/wdc-evidence-check/summary.json`
  (`cost_complete: false`, so the figure is a lower bound); teacher pass cost in
  `REVISIT.md` (Swapped-Order Teacher Pass).
- TODO: audit a random sample of teacher–student *agreed* pairs with the judge
  and report how often it would change the label. A near-zero rate turns point 2
  from an assumption into a result.

- Reference point: Steiner & Bizer, *Labeling Training Data for Entity
  Matching Using Large Language Models* (arXiv:2606.28823). Labelling the
  training sets of all five benchmarks (Abt-Buy, Walmart-Amazon, WDC Products,
  DBLP-ACM, DBLP-Scholar) with GPT-5.2 cost USD 28.31–40.88. Manual labelling of
  the same sets is estimated at 470 hours; the abstract gives no USD figure for
  humans. One judge pass over our 2,500 WDC pairs (about USD 48) would cost
  more than their LLM labelling of all five benchmarks, so it would undo the
  "cheap LLM labelling" premise. Selective use (about USD 3 for 158 pairs)
  keeps it.
- Per pair, WDC only (paper Table 5): GPT-5.2 labelling cost USD 10.83–13.54;
  human effort about 165.3 h at 30 s per pair, so about 19,800 pairs. That is
  about USD 0.0006 per pair, against USD 0.019 per pair for our judge: roughly
  30× more expensive per pair.

**Weak point.** Cost alone shows the full version is expensive, not that the
selective version loses nothing. That needs the audit in point 2.

**Status.** draft
