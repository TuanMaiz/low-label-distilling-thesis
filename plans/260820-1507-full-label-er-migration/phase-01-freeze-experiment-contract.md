---
phase: 1
title: "Freeze Experiment Contract"
status: pending
priority: P1
effort: "3-5d"
dependencies: []
---

# Phase 1: Freeze Experiment Contract

## Overview

Write one human-readable scientific plan/checklist, then freeze it by Git
commit. Code hashes this file for provenance but never parses its Markdown or
treats it as executable configuration.

## Context Links

- Scientific plan: `/mnt/d/study/cao-hoc/luan-van/code/plans/260820-1507-full-label-er-migration/research/experiment-contract.md`
- Blocked writing plan: `/mnt/d/study/cao-hoc/luan-van/code/plans/260704-distiller-wdc-thesis-writing/plan.md`
- Deleted earlier execution plan/runner: Git history only.

## Requirements

- Decide exactly three datasets: IDs, versions, upstream URLs, licenses,
  official splits, expected counts/balance, source hashes, fields, and overlap
  policy. Phase 2 records normalized hashes in generated manifests.
- Decide exactly three jointly encoding compact cross-encoder models: repository,
  immutable revision, backend, parameter/license/context facts, and eligibility
  evidence. Record candidates, exclusions, and final choices.
- Freeze LLM labeler/provider, answer-only prompt/version, temperature, token limit,
  parser, maximum attempts, terminal reject behavior, pricing snapshot, and cost
  ceiling. No automatic fallback labeler.
- Freeze the 3×3×2 grid, one seed, match-F1 primary metric, supporting metrics,
  direct baseline scope, timing/cost fields, and break-even equation.
- Freeze leakage rules: gold train labels feed only the gold arm; validation/test
  gold labels are evaluation-only; prompts exclude gold/entity truth; direct and
  compact-model accuracy use identical test IDs. A smaller direct cost sample is not
  an accuracy comparison.
- Audit likely benchmark contamination for the LLM labeler and compact models using release dates
  and training disclosures; record low/unknown/high risk and claim limits.
- Include manual checklist confirmations before paid labeling and final test.
  Git history records later scientific edits.

## Architecture

Markdown is the scientific source of truth; dataset/model JSON files hold
executable settings. Run manifests later record the Git commit and hashes of
this file, used configs, targets, prompt version, and runtime.

## Related Code Files

- Create: `/mnt/d/study/cao-hoc/luan-van/code/plans/260820-1507-full-label-er-migration/research/experiment-contract.md`
- Read: `/mnt/d/study/cao-hoc/luan-van/code/supervision/prompts.py`
- Read: `/mnt/d/study/cao-hoc/luan-van/code/models/student_config.py` (legacy identifier)
- Read: `/mnt/d/study/cao-hoc/luan-van/code/utils/cost_accounting.py`
- Do not create a global experiment JSON, Markdown parser, or confirmation files.

## Tests Before

No machine test parses the Markdown. Review it manually against authoritative
sources and have the researcher confirm every choice. Executable config tests
begin in Phases 2 and 4.

## Implementation Steps

1. Research authoritative dataset/model sources and document candidates.
2. Choose the two non-WDC datasets; record exact facts and source checksums.
3. Choose exactly three eligible models; record revisions and contamination risk.
4. Record LLM labeler/prompt, retry policy, direct scope, grid, seed, metrics,
   leakage rules, ceilings, stopping rules, and break-even definition.
5. Add unchecked manual items for paid labeling and final testing.
6. Review with the researcher, remove placeholders, and commit the Markdown.

## Test Scenario Matrix

| Review scenario | Expected |
|---|---|
| Fact lacks authoritative source | Checklist remains blocked |
| Not exactly 3 datasets/models | Checklist remains blocked |
| Prompt exposes truth | Rewrite before commit |
| Contamination risk unknown | Record `unknown` and claim limitation |
| Projected cost exceeds ceiling | Revise before confirmation |

## Success Criteria

- [ ] Plan names exact datasets/models, LLM labeler/prompt, grid, seed, metrics,
  baselines, leakage rules, ceiling, and manual confirmation points.
- [ ] Source/license/contamination evidence is complete.
- [ ] Reviewed Markdown is frozen by a dedicated Git commit.

## Risk Assessment

Ambiguous versions, architecture claims, or contamination can invalidate the
comparison. Use primary sources, immutable revisions/checksums, and explicit
unknown ratings before implementation or spend.

## Security/Data Integrity

Do not store API keys. Record only public identifiers/pricing assumptions and
source hashes. Never copy gold/entity truth into machine-labeling prompts.

## Next Steps

Phase 2 implements dataset configs; Phase 4 implements compact-model configs. Amend
scientific decisions only through a reviewed Git commit.
