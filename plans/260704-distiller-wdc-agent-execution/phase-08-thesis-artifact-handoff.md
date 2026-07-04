---
phase: 8
title: "Thesis Artifact Handoff"
status: pending
priority: P1
effort: "2-3 days"
dependencies: [7]
---

# Phase 8: Thesis Artifact Handoff

## Overview

Package experiment evidence into writing-ready materials for the official thesis structure.

## Requirements

- Functional: provide tables, figures, commands, artifact paths, and short interpretation notes.
- Non-functional: keep claims conservative and trace every number to an output file.
- Writing alignment: map artifacts to `Chuong 3`, `Chuong 4`, and `Chuong 5`.

## Architecture

```text
metrics + figures + failure examples + cost summary
  -> thesis artifact index
  -> writing plan updates
  -> advisor package
```

## Related Files

- Create: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/thesis_artifact_index.md`
- Update: `/mnt/d/Study/Cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`
- Optional create: `/mnt/d/Study/Cao-hoc/luan-van/code/journal/260704-distiller-wdc-experiment-summary.md`

## Implementation Steps

1. Create artifact index with:
   - dataset statistics.
   - run matrix.
   - metrics table paths.
   - figure paths.
   - failure table paths.
   - cost table paths.
2. Write short interpretation notes:
   - what result supports the thesis.
   - what result weakens the thesis.
   - what limitation must be admitted.
3. Map artifacts to thesis chapters:
   - Chapter 3: pipeline and system files.
   - Chapter 4: experiment matrix and hyperparameters.
   - Chapter 5: metrics, plots, failure analysis, cost analysis.
4. Prepare advisor checkpoint package:
   - one-page summary.
   - main table.
   - main plot.
   - next decision.
5. Update the thesis writing plan statuses if `ck plan check` is available and appropriate.

## Success Criteria

- [ ] Thesis artifact index exists.
- [ ] Every thesis result has a source file path.
- [ ] Advisor package is ready.
- [ ] Writing plan has concrete evidence to use.

## Risk Assessment

- Risk: outputs are scattered.
  Mitigation: centralize paths in the artifact index.
- Risk: writing overclaims.
  Mitigation: include both positive and negative interpretations in the handoff.
