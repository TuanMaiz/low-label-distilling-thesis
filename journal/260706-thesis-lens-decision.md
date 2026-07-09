# 2026-07-06 - Thesis Lens Decision

## Decision

The thesis should not frame LLM-label distillation for Entity Matching as a new
method. Steiner/Bizer-style LLM-labeling work and DistillER already cover the
core method family. The safer framing is:

> A cost-aware, low-label, failure-slice study of LLM-label distillation for hard
> product matching, with WDC Products as the thesis-core stress test.

## Why This Helps

- It avoids looking identical to prior LLM-labeling papers.
- It makes the contribution empirical and defensible for a master's thesis.
- It turns cost accounting from a side note into the main evaluation lens.
- It uses WDC-specific difficulty patterns as a source of analysis rather than
  treating WDC as just another benchmark row.

## Concrete Lens

- Cost lens: compare repeated direct LLM matching with one-time teacher labeling
  plus compact-student inference.
- Low-label budget lens: report curves over `16 / 32 / 64 / 128`, with `256`
  and `full` only if feasible.
- WDC difficulty lens: analyze hard negatives, missing fields, brand conflicts,
  title/model-number overlap, long descriptions, and price/currency mismatch.
- Teacher-noise lens: check whether students inherit, amplify, or smooth teacher
  mistakes.
- External-validity lens: add Abt-Buy, Walmart-Amazon, or a DBLP dataset later
  only after the WDC pilot has a clear signal.

## Plan Updates

- Updated the active execution plan with the revised research question and
  thesis lens.
- Updated the experiment contract with analytical-lens, novelty-boundary,
  optional-dataset, cost, and failure-slice guidance.
- Updated Phase 7 so failure analysis is responsible for slice-level and
  teacher-noise evidence, not only aggregate F1 and cost.

## Next Step

Keep Phase 3 focused on generating validated answer-only teacher labels and a
direct LLM baseline. Do not add new datasets until the WDC Phase 5 pilot and
Phase 7 analysis pipeline show a usable signal.
