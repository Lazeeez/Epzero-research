# Development screen D10: the combined repair over three seeds

Recorded on 24 November 2026, before it was run, after screen D9 (seed 11).

## Design

As D9 (product path: 8-bit codes, `fast_build` on, M 16, efConstruction 100; snapshot 1 bulk-built
and refined; snapshots 2 and 3 inserted one at a time), seeds 29 and 47, three arms each: grown with
no repair; grown with `extra_backlinks` 4 and each snapshot's insertions refined; rebuilt (reference).
With D7/D9's seed 11 this gives three seeds per arm. Measures as D9.

## Prediction

On each seed, the combined repair cuts the grown newest-cohort excess at ef 200 by at least half,
raises recall@100, and raises the oldest cohort's miss rate by at most 0.3 points; averaged over the
three seeds, the repaired excess is within 0.5 points of the rebuilt one.
