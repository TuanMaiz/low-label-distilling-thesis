---
date: 2026-08-25
session: wdc-full-label-target-publication
---

# Journal: 2026-08-25 — WDC Full-Label Target Publication

## Context

The WDC vertical slice needed deterministic training targets for the first compact-model experiment: official gold supervision and GPT-5.6 Sol-high machine supervision. This step prepared and validated artifacts only; it made no paid API calls and started no model training.

## What Happened

- Built 2,500 gold and 2,500 Sol-high LLM-hard targets from the official WDC training split and the completed labeler outputs.
- Preserved the gold distribution of 500 matches and 2,000 non-matches; the LLM targets contain 495 matches and 2,005 non-matches.
- Recorded 79 disagreements and 96.84% agreement between the two supervision sources.
- Bound the LLM artifact to the completed run evidence: USD 2.693225, 995,653 input tokens, and 63,716 output tokens.
- Added strong validation that independently rederives all five published artifacts from source evidence and checks audit/retry consistency, exact model identity, and path portability.
- Verified 105 repository tests and 12 labeling-workflow tests.

## Reflection

The target pair is now reproducible enough to support the WDC gold-versus-LLM training comparison and later thesis tables. Agreement is useful evidence about label-source similarity, but it is not an independent estimate of machine-label accuracy because benchmark gold is the comparison reference and disagreements can have different causes.

## Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| Publish both target sources with exact provenance | The experiment must isolate supervision source while keeping training pairs fixed | Future WDC training cells can consume comparable, auditable inputs |
| Require independent full-bundle rederivation during validation | File hashes alone cannot detect internally consistent semantic tampering | Model, cost, audit, retry, and source metadata are checked against upstream evidence |
| Keep Phase 3 incomplete | Only the WDC target pair exists; the other datasets and broader contract are not frozen | No claim of completing the full target-construction phase |

## Next

- Use the two WDC targets for the single approved Qwen reranker gold-versus-Sol-high experiment after its training contract is ready.
- Freeze the remaining Phase-1 choices before publishing targets for other datasets or running additional paid labeling.
- Reuse the provenance and disagreement artifacts as thesis-writing evidence without treating agreement as standalone accuracy.
