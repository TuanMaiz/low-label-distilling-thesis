---
phase: 7
title: "Verify and Handoff"
status: pending
priority: P1
effort: "3-5d"
dependencies: [1, 2, 3, 4, 5, 6]
---

# Phase 7: Verify and Handoff

## Overview

Verify the migration from a clean checkout and provide reproducible
commands/artifact maps for running and writing
the thesis without reopening scope.

## Context Links

- Master plan: `./plan.md`
- Blocked writing plan: `/mnt/d/study/cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`
- Repository guidance: `/mnt/d/study/cao-hoc/luan-van/code/AGENTS.md`, `/mnt/d/study/cao-hoc/luan-van/AGENTS.md`

## Requirements

- From a clean clone, review the scientific checklist and verify datasets,
  target dry run, matrix listing,
  one smoke cell, aggregation fixtures, and packaging using only documented
  commands and the uv-managed environment.
- Full suite passes; active docs/source contain no runnable references to
  low-label budgets, active selection, rationale targets, or deleted Phase-05
  components. Cite deleted context by Git commit only when necessary.
- Runbook lists environment/dependencies, acquisition/license checks, secrets,
  dry-run/manual confirmation, resume/recovery, validation completion, explicit
  `--allow-final-test`,
  aggregation, expected artifact cardinalities, and checksum verification.
- Update both AGENTS files after the workflow is genuinely implemented. Unblock
  the writing plan only when the scientific plan and aggregate schema are stable.

## Architecture

Verification follows the same dependency chain as production and consumes no
undeclared state: `clean clone -> plan/configs -> verify/acquire -> targets -> smoke
matrix -> aggregate fixture -> artifact verification`. A handoff manifest links
code commit, plan/config hashes, runbook, artifacts, and known limitations.

## Related Code Files

- Create: `/mnt/d/study/cao-hoc/luan-van/code/docs/full-label-experiment-runbook.md`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/docs/full-label-artifact-schema.md`
- Create: `/mnt/d/study/cao-hoc/luan-van/code/docs/journals/{timestamp}-full-label-migration-handoff.md`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/README.md`, `/mnt/d/study/cao-hoc/luan-van/code/QUICKSTART.md`, `/mnt/d/study/cao-hoc/luan-van/code/data/README.md`
- Modify: `/mnt/d/study/cao-hoc/luan-van/code/AGENTS.md`, `/mnt/d/study/cao-hoc/luan-van/AGENTS.md`
- Modify dependency/status only when verified: `/mnt/d/study/cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`
- Delete/update obsolete runner-exclusive tests and active documentation; do not
  recreate the deleted execution-plan directory or Phase-05 runner.

## Tests Before

Add a clean-fixture acceptance test/script that verifies documented commands,
config cardinalities, absence of active retired imports/scripts, artifact
hash validation, final/partial aggregate labeling, and package contents. Run
the current migrated suite, including all newly introduced tests.

## Implementation Steps

1. Audit git status/diff and active references; remove/update obsolete
   runner-exclusive tests/docs and use Git references for deleted context.
2. Write runbook and artifact schema from tested commands—not anticipated ones.
3. Execute the full unit/integration suite and clean-clone fixture workflow;
   record commands, environment, counts, hashes, and documented deviations.
4. Verify matrix expectations: 3 datasets, 6 targets, 3 configs, 18 student
   cells, 3 direct baselines, 9 paired comparisons; verify packages from hashes.
5. Update both AGENTS files to the new research question, architecture, commands,
   immediate next steps, scope guardrail, and historical-plan status.
6. Write concise handoff journal; mark this plan complete only when every gate
   passes. Then update the writing plan dependency/status to consume the new
   scientific-plan/result tables.

## Test Scenario Matrix

| Scenario | Expected |
|---|---|
| Fresh checkout + documented setup | Tests and config verification pass |
| No API key | Offline verification works; call stages explain requirement |
| Deleted Phase-05 path in active docs/tests | Acceptance test fails |
| Active retired import/command | Acceptance test fails |
| Artifact package tampered/missing | Hash verification fails |
| Complete matrix handoff | Exact 18/3/9 cardinalities and stable schema |

## Success Criteria

- [ ] Clean-clone acceptance and complete test suite pass with documented output.
- [ ] Active docs/AGENTS reflect the new workflow and do not depend on deleted
  plans/runner/tests.
- [ ] Handoff journal/runbook/schema identify plan hash, commit, commands,
  artifacts, cost assumptions, limitations, and next owner action.
- [ ] Writing plan is unblocked only after verified stable schemas.

## Risk Assessment

Documentation drift can harm reproducibility. Mitigate by deriving docs from
tested commands, citing deleted context by Git commit, and clean-clone checks.

## Security/Data Integrity

Scrub archives and logs for `.env`, tokens, provider headers, and local absolute
secret paths. Verify package manifests/checksums before sharing; document dataset
licenses and model/provider terms.

## Next Steps

Start the blocked writing plan with the committed scientific plan and aggregate schemas;
do not add methods/datasets/models unless the supervisor requires a scope change.
