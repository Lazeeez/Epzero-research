# Development screen D13: a stronger repair (8 links and refining new insertions)

Recorded on 24 November 2026, before it was run, after the full-scale engine arm missed the
1.0-point target on the test queries (repaired excess 1.54 points, `scripts/pace/LONGEVAL_ENGINE_ARM.md`).

## Design

As D10 (product path on the 300K development sample: 8-bit codes, `fast_build` on, M 16,
efConstruction 100; snapshot 1 bulk-built and refined; snapshots 2 and 3 inserted one at a time,
each snapshot's insertions refined with `refine_links(threads=all, ef=50, first=...)`), with
`extra_backlinks` 8 instead of 4, seeds 11, 29 and 47. Compared with the grown, repaired (4 links)
and rebuilt arms of D7, D9 and D10 on the same seeds. Measures as D10: the newest cohort's
within-query excess, the oldest cohort's miss rate and recall@100 at ef 200 on the test queries, plus
insertion and refine time.

## Prediction

Averaged over the three seeds, 8 links cut the excess below the 4-link repair's 0.63 points to at most
0.40 points, raise the oldest cohort's miss rate against grown by at most 0.4 points (D7b, without
the refine of new insertions, gave +0.31 with 8 links), and leave recall@100 no lower than the 4-link
repair. If the oldest cohort rises by more than 0.4 points, 8 links do not go to a full-scale run.

## Result
