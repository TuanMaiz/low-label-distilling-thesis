---
title: WDC teacher majority phase implemented
date: 2026-09-29
summary: Implemented and executed the offline majority-vote and consistency phase after the three WDC teacher passes.
---

## What happened

Added and completed Phase 3 of the uncertainty-aware label-refinement plan. The
offline builder consumed the three verified WDC `pair_id,result` files and
produced a downstream-compatible majority-label CSV, a `3/3` or `2/3`
consistency sidecar, and a hash/count summary.

## Decision

Keep implementation small: one offline builder and one focused test file.
Reuse the frozen blinded training order, reject any pass misalignment, and do
not read gold, validation, or test labels.

## Result

- 2,500 unique majority labels in frozen input order.
- 495 `match` and 2,005 `non_match` majority labels.
- 2,487 `3/3` pairs and 13 `2/3` pairs.
- Byte-identical outputs on a second run.
- Focused tests pass 5/5; full repository tests pass 186/186.
- The test module's `real-files` mode accepts explicit input/pass paths and
  reproduced the real WDC counts and hashes through disposable outputs.
- No API call, GPU action, training, or hidden-label read occurred.

## Next steps

Design the three-fold OOF student-prediction phase around
`majority_labels/llm_unrefined.csv`, joining later evidence by `pair_id` rather
than modifying the original pair JSONL.

AgentWiki publish was not requested and was skipped.
