# Reviewer question mock

Questions a reviewer or the defense committee may ask, with our current answer.
Used to prepare for the defense and to spot weak points early.

One file per question: `qNN-short-name.md`. Each file has:

- **Question** — as a reviewer would ask it.
- **Likely follow-up** — the harder question that comes next.
- **Our answer** — what we say now.
- **Evidence** — numbers, files or runs that back the answer; mark missing ones `TODO`.
- **Weak point** — where the answer is still thin.
- **Status** — `draft` (answer only), `backed` (evidence exists), `open` (no good answer yet).

## Index

| # | Question | Status |
|---|---|---|
| [q01](q01-judge-on-all-pairs.md) | Why not run the evidence-based judge on every training pair? | draft |
| [q02](q02-judge-is-fourth-vote.md) | Isn't the LLM judge just a fourth teacher vote? | draft |
| [q03](q03-llm-randomness.md) | The LLM is stochastic, so how is the method reproducible? | draft |
| [q04](q04-human-validation-test.md) | If the goal is no human labels, why use human validation and test? | draft |
| [q05](q05-jury-for-teacher-labels.md) | Why not use the judge panel to label the training data itself? | draft |
