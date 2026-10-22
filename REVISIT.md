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

## Swapped-Order Teacher Pass (WDC train)

Status: deferred
Created: 2026-10-21
Context: `plans/261021-2235-wdc-order-swap-and-hard-pair-upweighting/` Phase 5;
record-order probe findings in `plans/journals/2026-10-20-record-order-swap-findings.md`.

### Current Decision

Not run. The plan uses only the student order-flip signal (OOF fold models
scored in both orders).

### Why Revisit

Teacher consistency has little resolution: 2,487/2,500 pairs are 3/3. One
Sol-high pass with Record A and Record B exchanged would perturb the input
instead of repeating it, and could expose teacher uncertainty that repetition
hides. It would also make the teacher-trust selection rule stricter (3/3 and no
teacher order flip).

### Revisit Options

1. One pass on all 2,500 train pairs, same model, prompt, provider pinning and
   settings; about USD 2.7; add `--swap-record-order` to
   `labeller-screening/run_full_wdc.py` with tests and a dry run first.
2. Skip, and report teacher order sensitivity as untested.

### Guardrails

- Needs explicit paid authorization and a spend ceiling; the standing rule
  "Do not relabel WDC" applies until the researcher lifts it for this pass.
- The swapped pass is a separate artifact; it never replaces the three passes.
- Decide before Phase 6 freezes the selection rule; never after seeing test.

## LLM-Labeled Validation (replacing human validation)

Status: deferred
Created: 2026-10-21
Context: researcher point on 2026-10-21: if LLM labels can replace people for
train, the same may hold for validation. Plan Phase 9.

### Current Decision

Validation gold stays the human development set for checkpoint and threshold
selection. Phase 9 only checks how sensitive results are to a small (500-pair)
human validation subset.

### Why Revisit

WDC validation (2,500) is as large as train, which weakens the defense that
human validation is cheap. Showing that LLM-labeled validation selects nearly
the same checkpoint and threshold would make the pipeline human-free except
for test.

### Revisit Options

1. Sol-high labels the 2,500 validation pairs (one pass about USD 2.7, or three
   passes plus majority about USD 8). Re-select epoch and threshold on LLM
   labels from saved per-epoch predictions; report test F1 and teacher
   accuracy on validation. No retraining needed.
2. Keep the small-subset check only and state the limitation.

### Guardrails

- Needs explicit paid authorization and a contract change: AGENTS.md says LLM
  machine labeling operates on training pairs only.
- Analysis on frozen arms only; results must not feed back into method choices
  for the same WDC runs.

## Order-Swap and RHO-Style Hard-Pair Upweighting Plan

Status: backlog
Created: 2026-10-22
Context: `plans/261021-2235-wdc-order-swap-and-hard-pair-upweighting/` (written
2026-10-21, no phase started).

### Current Decision

Out of scope. The thesis question is now how to find uncertain training pairs
and treat them. Detection keeps teacher–student (OOF) disagreement plus teacher
2/3 pairs, all 158 already diagnosed. Treatment is chosen from the
human-defined evidence categories, not from a loss-based score over all 2,500
pairs.

### Why Revisit

- Swap training and order-flip detection: student order instability is real
  (`plans/journals/2026-10-20-record-order-swap-findings.md`); see `ideas.md` I16.
- RHO-style or teacher-trust selection (K = 250) is an alternative way to find
  hard-but-useful pairs outside the flagged set, and a possible comparison.

### Revisit Options

1. Reopen only after the category-based treatment has a validation result.
2. Reuse its gold boundary, k = 2 duplication, random control and paired tests
   in the new plan (these parts are not deferred).

### Guardrails

- No GPU run or code from this plan while it is in backlog.
- Swapped-Order Teacher Pass above depends on this plan and stays deferred.

