# q04 — If the goal is no human labels, why use human validation and test?

**Question.** The thesis claims LLMs can replace human labelling, yet
validation and test are human-labelled.

**Our answer.** Validation and test need humans to make sure they are correct,
and producing them is cheaper than labelling train. Train gold is used only in
the final step to validate, never to design or select the method.

**Evidence.** Gold boundary in the active plan.

**Weak point.** WDC validation (2,500 pairs) is as large as train, which weakens
"cheaper". See `REVISIT.md` (LLM-Labeled Validation) for the planned check.

**Status.** draft
