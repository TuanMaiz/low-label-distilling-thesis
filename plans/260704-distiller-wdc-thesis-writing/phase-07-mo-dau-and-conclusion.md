---
phase: 7
title: "Mo Dau And Conclusion"
status: pending
priority: P1
effort: "3-5 days"
dependencies: [4]
---

# Phase 7: Mo Dau And Conclusion

## Overview

Write `Mo dau` and the final conclusion after the results chapter is stable, so the thesis claim matches the evidence.

## Requirements

- Functional: write reason for topic, purpose, object/scope, meaning, structure, conclusion, limitations, and future work.
- Non-functional: no claims unsupported by Chapter 5.
- Official structure: `Mo dau` is before Chapter 1; conclusion is final chapter.

## Architecture

```text
Chapter 5 findings
  -> safe active-selection thesis claim
  -> Mo dau
  -> conclusion and recommendations
```

## Related Files

- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/phase-04-chapter-5-results-draft.md`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/thesis_artifact_index.md`

## Implementation Steps

1. Write `Mo dau` sections:
   - reason for choosing the topic.
   - research purpose.
   - research object and scope.
   - scientific meaning.
   - practical meaning.
   - thesis structure.
2. Write final chapter sections:
   - completed work.
   - main findings.
   - scientific contribution.
   - practical contribution.
   - limitations.
   - recommendations/future work.
3. State the final answer to whether active LLM-label selection improves over random selection under low budgets.
4. Check every claim against Chapter 5.
5. Remove any novelty language that results do not support.

## Success Criteria

- [ ] `Mo dau` draft exists.
- [ ] Conclusion draft exists.
- [ ] Claims match Chapter 5.
- [ ] Limitations are explicit.

## Risk Assessment

- Risk: introduction sounds stronger than results.
  Mitigation: write it after results and use conservative wording.
- Risk: conclusion repeats abstract only.
  Mitigation: include concrete numeric findings from Chapter 5.
