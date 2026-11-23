# Development screen D7: arrival bias and `extra_backlinks` in Epzero's product path

Recorded on 23 November 2026, before it was run. Development data only (300K LongEval sample, 381
test queries).

## Why

Screens D3–D6 used a plain configuration (full-precision vectors, no refine pass, every document
inserted one at a time). Epzero's automatic build instead uses 8-bit codes (SQ8), builds the graph
on the codes, and runs a global refine pass after the bulk build. Documents added later are inserted
one at a time with no refine pass. That is where arrival bias would reach users.

## Design

Epzero development build, SQ8 with full vectors kept, `fast_build` on, M 16, efConstruction 100,
seed 11:

- **Grown** (`extra_backlinks` 0 and 4): snapshot-1 documents bulk-built (training the 8-bit ranges on
  them, all cores), refine pass, then snapshot-2 and snapshot-3 documents inserted one at a time in
  arrival order (one thread).
- **Rebuilt** (reference, `extra_backlinks` 0): all documents trained on and bulk-built at once (all
  cores, so no meaningful insertion order), refine pass.

Search: ef 100, 200 and 400, re-ranking the whole beam with full vectors (rerank = ef). Measures: the
newest cohort's within-query excess (snapshot 3 against 1), miss rate per cohort, recall@100 and
recall@10, distance computations, insertion time for snapshots 2 and 3.

## Prediction

Grown without the option shows a newest-cohort excess clearly above the rebuilt reference (by at
least 1 point at ef 200); `extra_backlinks` 4 cuts the grown excess by at least half, with the
oldest cohort's miss rate rising by at most 0.3 points and recall not falling.
