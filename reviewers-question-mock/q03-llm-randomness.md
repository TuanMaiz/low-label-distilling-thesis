# q03 — The LLM is stochastic, so how is the method reproducible?

**Question.** Running the judge again could give different decisions. How can
others reproduce the training set?

**Our answer.** Prompt, model, provider and settings are fixed, and every raw
response is stored. The rule that turns judge outputs into actions is
deterministic, so the same saved outputs always give the same training set.
Randomness is reported separately as reliability: run the judge several times
and report agreement, or use the majority.

**Evidence.** Saved responses under
`outputs/uncertainty_refinement/wdc-evidence-check/responses/`. TODO:
agreement across repeated judge runs.

**Weak point.** Reproducing from saved outputs is not the same as the method
being stable; the agreement number is needed.

**Status.** draft
