# Separate WDC teacher passes

Implemented the optional `--no-reuse-existing` switch in the current WDC
runner. It omits both reuse paths; default screening reuse is preserved.
The existing request construction, response records, cost accounting, and
resume behavior are reused.

Verification: 13 screening tests pass. The pass-2 dry run reports 2,500
training rows, zero reused labels, and 2,500 new requests. The existing
pass-1 CSV contains 2,500 unique IDs with valid binary labels.

No paid calls were made. Passes 2 and 3 require current pricing review and
separate approval under the phase plan. The August pricing snapshot printed
by the dry run is historical, not a newly verified price.

The broad repository unittest command produced no output and was interrupted;
no full-suite success is claimed for this revision. Focused screening checks
and `git diff --check` passed.

On 2026-09-29, paid pass 2 completed with 2,500 fresh valid labels, zero
retries, zero reuse, and USD 2.709705 cost. Independent checks confirmed exact
ordered input alignment, unique response IDs, frozen model/provider identity,
valid request hashes, matching input digest, and cost below the USD 5 ceiling.
The screening suite passed 13/13 and the full repository suite passed 181/181.
