# Development screen D15: refining each new batch harder

Recorded on 24 November 2026, before it was run, after screen D14.

## Why

Screen D14 found that refining a node helps that node, and that nodes refined more often gain more.
Refining the older side widened the gap (0.28 → 0.58 points). The other direction is to give the
newest documents more refinement than they get now, without touching the older ones. In the current
repair (D13), a later document is inserted one at a time into a denser graph and refined once, at
width 50, at its snapshot's end. A snapshot-1 document was bulk-built and refined once at the same
width, then kept receiving links from every later arrival. At full scale the repair still leaves 1.17
points against the 1.0 target (a rebuild gives 0.43).

## Design

As D13 (product path on the 300K development sample: 8-bit codes, `fast_build`, M 16,
efConstruction 100; snapshot 1 bulk-built and refined at width 50; snapshots 2 and 3 inserted one at a
time; `extra_backlinks` 8; seeds 11, 29, 47; test queries; ef 100, 200, 400). Only the refine of each
later snapshot's insertions changes (`refine_links(first=position before the snapshot)`; no engine
change):

- **wide:** one pass at width 100 instead of 50.
- **twice:** two passes at width 50, the second searching the graph the first left.

Compared with D13 (excess 0.28, oldest 4.45%, recall 0.9506, refine 8 s) and the rebuild (0.12,
0.9520). Measures as D13.

## Predictions

Averaged over the three seeds, at ef 200:

1. Each arm lowers the excess below D13's 0.28 points, to at most 0.22.
2. Each arm raises the oldest cohort's miss rate against grown (4.09%) by at most 0.45 points.
3. Each arm keeps recall@100 at least 0.9501 (D13's, less the seeds' spread).
4. Each arm's refine takes at most 2.5× D13's 8 s on all cores.

The screen's room is small (D13 is within 0.16 points of a rebuild, with seed spreads of about 0.15),
so a pass here is a reason for a full-scale run, not a result.

## Decision

An arm goes to a full-scale run (a third engine-arm package, against `grown`, `repaired8` and
`rebuilt`, under decision 3's margins) if it meets predictions 1–3. If both do, the one with the
lower excess goes, or `twice` if the two are within 0.05 points, as it keeps the width the build uses.
If neither does, the newest documents' remaining disadvantage is not about how well their own links
are chosen. That is reported, and points to the in-link supply alone (section 7.3).
