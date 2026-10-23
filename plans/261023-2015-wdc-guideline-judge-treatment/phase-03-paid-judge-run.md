# Phase 3: Paid judge run and aggregation

## Context Links

- Runner: `supervision/run_identity_judge.py`
- Contract: `./contract.md` items 3–5, 11

## Overview

Priority P1. Pending. Blocked on explicit researcher authorization and the
USD 15 ceiling.

## Requirements

- Three passes on the 158 uncertain pairs; one pass on the 100 audit pairs.
- One predeclared retry per invalid output; still invalid → verdict 0.
- New `supervision/aggregate_judge_verdicts.py` → `final_verdicts.csv`
  (pair_id, verdicts per pass, final verdict, agreement) by contract item 5.
- Report: inter-pass agreement (share of pairs with 3/3, 2/3, no majority),
  verdict and step distribution, total cost.
- Audit report: share of agreed pairs where the judge verdict differs from the
  teacher label (would flip) or is 0 (would remove). Feeds
  `reviewers-question-mock/q01`.
- No gold read.

## Implementation Steps

1. Researcher authorizes spend.
2. Run pass 1 on 10 pairs, inspect, then the rest; passes 2–3; audit pass.
3. Retry invalids once; aggregate; report.
4. Update q01 and q03 in `reviewers-question-mock/` with the numbers.

## Todo List

- [ ] Authorization recorded
- [ ] 3 passes + audit complete
- [ ] Aggregation and reports
- [ ] Reviewer notes updated

## Success Criteria

- Every one of the 158 has a final verdict; cost within ceiling; all raw responses kept.

## Risk Assessment

- Low inter-pass agreement. Mitigation: report it; the aggregation rule already sends no-majority pairs to remove.
- Spend overrun. Mitigation: stop at ceiling; resume only with new authorization.

## Next Steps

- Phase 4.
