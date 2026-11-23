# Development screen D9: refining recent insertions in the product path

Recorded on 23 November 2026, before it was run, after screen D8.

## Question

Screen D8 showed that in Epzero's product path the refine pass after the bulk build is itself a
source of arrival bias: it treats only the documents present at the bulk build. D9 tests refining the
documents inserted since the last refine, once per snapshot (as a checkpoint would), using the new
`refine_links(first=...)` (Epzero branch `min-backlinks`), alone and with `extra_backlinks` 4.

## Design

As D7, grown mode (8-bit codes, `fast_build` on, M 16, efConstruction 100, seed 11; snapshot 1
bulk-built and refined; snapshots 2 and 3 inserted one at a time), and after each later snapshot's
insertions, `refine_links(threads=all, ef=50, first=position before that snapshot)`. Arms:
`extra_backlinks` 0 and 4. Compared with D7's grown (no refine of later snapshots) and rebuilt arms.
Measures as D7, plus the refine time.

## Prediction

Refining each snapshot's insertions cuts the grown excess at ef 200 (2.27 points) by at least half,
with the oldest cohort's miss rate rising at most 0.3 points and recall@100 not falling; the refine
time on all cores is at most the single-threaded insertion time of those documents. Adding
`extra_backlinks` 4 lowers the excess further.
