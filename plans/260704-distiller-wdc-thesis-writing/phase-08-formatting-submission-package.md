---
phase: 8
title: "Formatting Submission Package"
status: pending
priority: P1
effort: "1 week"
dependencies: [2, 3, 4, 5, 6, 7]
---

# Phase 8: Formatting Submission Package

## Overview

Bring the thesis into the official school format and prepare the advisor/submission package.

## Requirements

- Functional: enforce official formatting, lists, references, appendix discipline, and final consistency checks.
- Non-functional: avoid last-minute content changes unless a blocking issue is found.

## Architecture

```text
chapter drafts
  -> formatting pass
  -> figure/table numbering
  -> references
  -> appendices
  -> advisor package
  -> submission draft
```

## Related Files

- Official requirement PDF: `/mnt/d/Study/Cao-hoc/luan-van/quy chế/cach_thuc_trinh_bay_luan_van_quy_che_dao_tao_thac_si.pdf`
- Thesis example PDF: `/mnt/d/Study/Cao-hoc/luan-van/docs/Ví dụ luận văn thạc sĩ.pdf`
- Experiment artifact index: `/mnt/d/Study/Cao-hoc/luan-van/code/outputs/distiller_wdc/thesis_artifact_index.md`

## Implementation Steps

1. Apply official page settings:
   - A4.
   - one-sided.
   - Times New Roman 13 body.
   - line spacing 1.5.
   - margins: top 2.5 cm, bottom 2.5 cm, left 3.5 cm, right 2 cm.
2. Verify front matter:
   - sub-cover.
   - declaration.
   - table of contents.
   - abbreviations.
   - list of tables.
   - list of figures.
3. Verify table and figure numbering:
   - tables above.
   - figures below.
   - numbering tied to chapter.
4. Verify citations:
   - square brackets.
   - no uncited references.
   - no missing cited references.
5. Group references by language.
6. Check appendices:
   - prompt templates.
   - extra tables.
   - extra examples.
   - appendices not longer than main content.
7. Prepare advisor package:
   - thesis PDF or doc.
   - one-page result summary.
   - main result table.
   - main figure.
   - open questions.

## Success Criteria

- [ ] Formatting follows official requirement.
- [ ] Figure/table lists are correct.
- [ ] References are grouped and complete.
- [ ] Appendices are controlled.
- [ ] Advisor package is ready.

## Risk Assessment

- Risk: official formatting conflicts with example thesis.
  Mitigation: official requirement wins.
- Risk: reference style is inconsistent.
  Mitigation: run a dedicated reference pass before final PDF.
