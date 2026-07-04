---
phase: 2
title: "Pivot Cleanup"
status: pending
priority: P1
effort: "0.5-1 day"
dependencies: [1]
---

# Phase 2: Pivot Cleanup

## Overview

Make the pivot explicit in project guidance and preserve the failed rationale-distillation result as negative evidence rather than deleting it.

## Requirements

- Functional: update guidance only after Phase 1 contract is accepted.
- Non-functional: do not remove old rationale code; mark it as legacy/optional.
- Safety: do not overwrite user work or erase past experiment evidence.

## Architecture

The old rationale modules become optional ablation support. The new active path becomes:

```text
WDC pair -> LLM label -> validated label cache -> compact student target
```

instead of:

```text
WDC pair -> structured rationale -> rationale target -> seq2seq student
```

## Related Code Files

- Modify: `/mnt/d/Study/Cao-hoc/luan-van/code/AGENTS.md`
- Modify: `/mnt/d/Study/Cao-hoc/luan-van/code/CLAUDE.md`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/journal/260701-1653-phase-03-pivot-reflection.md`
- Read: `/mnt/d/Study/Cao-hoc/luan-van/code/journal/260701-1832-safe-dataset-option.md`
- Optional create: `/mnt/d/Study/Cao-hoc/luan-van/code/journal/260704-distiller-wdc-pivot-plan.md`

## Implementation Steps

1. Summarize the old Phase 3 result:
   - label-only was stronger overall.
   - structured rationale increased recall but hurt precision and F1.
   - rationale distillation is no longer the main thesis.
2. Update `AGENTS.md` and `CLAUDE.md` to point to this new active plan.
3. Keep old rationale files under `rationales/` because provider, cache, and prompt infrastructure can be reused.
4. Mark structured-rationale experiments as optional negative-history ablation.
5. Add a diary entry explaining why the pivot happened and what the new safe direction is.

## Success Criteria

- [ ] Future agents see label-level LLM-to-student distillation as the active plan.
- [ ] Old rationale code is not treated as the main workflow.
- [ ] Pivot reason is documented, not hidden.

## Risk Assessment

- Risk: project docs drift from actual work.
  Mitigation: update both guidance layers after the contract is stable.
- Risk: old files confuse future agents.
  Mitigation: label the old rationale path as legacy/optional.
