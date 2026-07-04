---
title: "Safe Dataset Option for DistillER-on-WDC Pivot"
date: 2026-07-01
phase: pivot
status: research_note
tags: [entity-resolution, datasets, distiller, wdc-products, benchmark-design]
---

# Safe Dataset Option for DistillER-on-WDC Pivot

## Context

The current pivot idea is to move away from rationale-target distillation and
benchmark a DistillER-style LLM-to-student ER setup on WDC Products and related
datasets.

The immediate concern is dataset overlap. DistillER already uses many classic
ER datasets:

- Buy-Abt / Abt-Buy
- Amazon-Google Products
- Walmart-Amazon
- ACM-DBLP / DBLP-ACM
- DBLP-Scholar
- IMDb-TMDb
- IMDb-TVDb
- TVDb-TMDb

Therefore, using only Abt-Buy, Walmart-Amazon, DBLP-ACM, and DBLP-Scholar would
mostly place the thesis on DistillER's terrain. This is not automatically bad,
but the claim cannot be dataset novelty.

## Safe Framing

The safer framing is:

> Reproduce or adapt DistillER-style supervision on shared classic benchmarks
> for comparability, then stress-test it on WDC Products and additional
> non-shared datasets.

In this framing:

- Shared datasets provide credibility and comparability against prior ER/KD
  work.
- WDC Products provides the main stress benchmark, because it is harder and was
  not used in DistillER.
- Additional non-shared datasets prevent the thesis from looking like a direct
  DistillER rerun.

## Recommended Dataset Suite

Use partial overlap instead of avoiding overlap completely:

| Role | Dataset | Reason |
|---|---|---|
| Main stress dataset | WDC Products | Not in DistillER; supports harder product matching, unseen products, corner cases, and class-prior stress. |
| Shared product baseline | Walmart-Amazon or Abt-Buy | Gives comparability with classic ER and DistillER-style product benchmarks. |
| Shared citation baseline | DBLP-Scholar or DBLP-ACM | Tests a non-product domain while keeping comparison to prior work. |
| Non-DistillER domain | BeerAdvocate-RateBeer | Adds messy review/catalog-style entities outside DistillER's dataset set. |
| Non-DistillER domain | Fodors-Zagats / Restaurants | Adds a small classic restaurant ER benchmark outside DistillER's dataset set. |

Concrete 5-dataset suite:

1. WDC Products
2. Walmart-Amazon
3. DBLP-Scholar
4. BeerAdvocate-RateBeer
5. Fodors-Zagats

Alternative if the advisor prefers more direct comparability:

1. WDC Products
2. Abt-Buy
3. Walmart-Amazon
4. DBLP-ACM
5. DBLP-Scholar

This alternative is easier to compare with recent LLM-for-ER work, but overlaps
more heavily with DistillER.

## Decision

The safest dataset strategy is:

> 2 shared datasets for comparability, WDC as the main hard benchmark, and 2
> non-shared datasets for contribution breadth.

This keeps the thesis academically safer than either extreme:

- Not too much overlap: avoids becoming only a DistillER rerun.
- Not too little overlap: avoids losing comparability with known ER baselines.

## Next Research Direction

After settling this dataset strategy, the next research thread should examine
the cheaper-supervision aspect:

- selective LLM labeling
- pair selection under a fixed LLM budget
- cheap-model-to-LLM cascades
- constrained decision scoring
- cost per F1 point
- inference cost versus labeling cost
