---
title: "Phase 3: Build WDC Teacher Majority and Consistency Artifacts"
status: complete
priority: P1
effort: "2-4h"
dependencies: [2]
---

# Phase 3: Build WDC Teacher Majority and Consistency Artifacts

## Overview

Combine the three completed WDC teacher prediction CSVs offline. Produce one
downstream-compatible majority-label CSV, one consistency sidecar, and a small
summary. This phase makes no API calls, starts no training, and reads no gold,
validation, or test labels.

Pass 3 is complete, and all three passes contain the same 2,500 ordered pair
IDs as the frozen blinded training input.

## Files

- Create `supervision/build_teacher_consensus.py`.
- Create `tests/test_teacher_consensus.py`.
- Write ignored artifacts under `data/cache/wdc_products/majority_labels/`.

## Output Contract

- `llm_unrefined.csv`: exactly `pair_id,result`, preserving frozen input order.
- `consistency.csv`: `pair_id,pass_01,pass_02,pass_03,majority_label,majority_votes,consistency`.
- `summary.json`: row/class/consistency counts plus SHA-256 hashes of the three
  source CSVs and two output CSVs.

Labels remain `match` or `non_match`; `majority_votes` is `2` or `3`, and
`consistency` is correspondingly `2/3` or `3/3`.

## Implementation Steps

1. Load the frozen blinded WDC training IDs and the three strict
   `pair_id,result` teacher CSVs.
2. Reject missing, duplicate, reordered, extra, or invalid rows before writing.
3. Compute the binary majority and consistency deterministically, then publish
   all outputs atomically.
4. Test `3/3`, both possible `2/3` directions, deterministic bytes, alignment
   failures, and the absence of any gold/validation/test dependency.
5. Run the focused suite and full repository suite, then execute the builder on
   the verified WDC passes.

## Todo

- [x] Verify pass 3 and three-pass ordered ID alignment.
- [x] Implement the builder and focused tests.
- [x] Build and inspect the WDC majority/consistency artifacts.
- [x] Record counts and hashes for the later OOF phase.

## Success Criteria

- Exactly 2,500 unique pairs appear once in each output and in frozen order.
- `3/3 + 2/3 = 2,500`; majority class counts also sum to 2,500.
- Re-running from unchanged inputs produces byte-identical outputs.
- Focused and repository tests pass.
- `llm_unrefined.csv` is ready to label the three-fold OOF training subsets.
- No paid call, GPU action, model training, or hidden-label access occurs.

## Completion Evidence

Completed offline on 2026-09-29. The outputs contain 2,500 unique ordered
pairs: 495 majority `match`, 2,005 majority `non_match`, 2,487 at `3/3`
consistency, and 13 at `2/3` consistency. Re-running the builder produced the
same bytes.

```text
llm_unrefined.csv sha256
b4dc5050e125e930c6b4b99b40004acfafedc9e9ba3fd22c608c5c3d0128f4c1

consistency.csv sha256
7280100a249e623858969da4ebea376c81a0f81aa6121eb2f3fe4d98711bebbb

summary.json sha256
10e615379b98fec5a9ef8d83faa84c677366fb18585ebb4eb0e83735239ed93a
```

The focused suite passes 5/5 and the repository suite passes 186/186. The test
module also accepts caller-supplied real artifact paths through its
`real-files` mode; this mode reproduced the same counts and hashes in a
temporary directory and deleted those temporary outputs afterward. The builder
read only the blinded training input and the three teacher prediction CSVs; it
made no API call and started no model run.
