# q02 — Isn't the LLM judge just a fourth teacher vote?

**Question.** If the judge is the same model as the teacher, it shares the
teacher's blind spots. Why is its decision better than a fourth vote?

**Likely follow-up.** Show that it changes decisions in a way plain re-voting
does not.

**Our answer.** The judge decides from a human-written guideline (categories,
what makes a pair decidable, which attributes count for identity) and must cite
verbatim evidence. A plain vote has neither. Humans write the guideline, not the
labels, as in annotation guidelines.

**Evidence.** TODO: guideline file; judge outputs; comparison with the three
plain passes on the same pairs.

**Weak point.** Same model as the teacher. A different judge model would make
the answer stronger.

**Status.** draft
