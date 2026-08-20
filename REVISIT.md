# Revisit Notes

This file tracks choices that are acceptable for moving the thesis forward now,
but should be revisited before final claims, thesis writing, or a larger
experiment run.

## Active Selection Bucket Scoring

Status: retired
Created: 2026-07-07
Updated: 2026-08-20
Context: Historical Phase 3 `llm_active_bucketed_v1` selector; removed from the
active tree during the full-label migration and retained in Git history.

### Current Decision

The first active-selection pilot now uses a four-bucket manifest:

- `easy_match_candidate`.
- `hard_match_candidate`.
- `easy_non_match_candidate`.
- `hard_negative_candidate`.

The default budget allocation is 25 percent per bucket. For budget 128 this is
32 rows per bucket. The ratio is controlled by function/CLI parameters so later
runs can predeclare a different mix without changing the selector code.

Within each bucket, rows are ranked by transparent product-pair features such
as title/description similarity, model-token overlap, brand agreement/conflict,
price/currency mismatch, and missing key fields. Gold labels are not used for
active scoring.

### Why Revisit

The equal 25 percent bucket ratio is easier to defend than a single blended
weighted score because it directly matches the WDC-motivated four-way difficulty
view: easy matches, hard matches, easy non-matches, and hard negatives.

Reviewers may still reasonably ask why the within-bucket feature scores were
chosen. The current answer is that they encode product-matching intuition, not
learned optimization. That is defensible only if the thesis frames the selector
as a predeclared heuristic policy and avoids claiming a new active-learning
algorithm.

### Revisit Options

1. Compare simple named selectors:
   - `random`
   - `llm_active_bucketed_v1`
   - `title_uncertainty`
   - `model_overlap`
   - `hard_negative_candidate`
2. Keep the bucketed selector but add an ablation table showing that conclusions
   do not depend on one arbitrary within-bucket score.
3. Try predeclared alternative ratios such as 20/30/20/30 if the equal pilot is
   noisy, but only before inspecting final validation/test student metrics.
4. Learn or tune bucket weights only from a training-only development protocol, never
   from validation/test results used for final reporting.

### Guardrails

- Do not tune selector scores or bucket ratios after inspecting student
  validation/test metrics.
- Do not present the bucketed heuristic as a novel active-learning algorithm.
- Always compare against the fixed random manifest at the same labeling budget.
- Report the active manifest label distribution as an audit artifact, not as a
  selection input.

### Thesis Wording

Safe wording:

> We use a predeclared, transparent heuristic selector to construct an active
> LLM-labeling subset with equal default coverage of four WDC-motivated
> candidate buckets. The selector is not optimized on evaluation results; it is
> compared against a random budget-matched control to test whether cheap
> product-specific preselection can improve cost-aware LLM-label distillation.

Avoid:

> We propose an optimized active-learning strategy.
