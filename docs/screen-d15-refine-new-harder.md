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

## Result

Run on 24 November 2026, after the design above was committed (Epzero `min-backlinks` at 2d279c0e; the
engine code these arms use is unchanged from 4d4a64cc). Ef 200, test queries of the 300K sample, means
over seeds 11, 29 and 47. The D13 row is from screen D13 on the same seeds. Raw results are in
`Oryx/build/screen-d15/*.json`.

| Arm (8 links) | Newest excess (11 / 29 / 47; mean) | Miss by cohort 1 / 2 / 3 | Recall@100 | Recall@10 | Distance computations | Refine, all cores |
| --- | --- | --- | --- | --- | --- | --- |
| refine new, width 50 (D13) | 0.28 / 0.13 / 0.43; **0.28** | 4.45 / 5.74 / 5.45% | 0.9506 | 0.9781 | 3,108 | 8 s |
| **wide** (width 100) | −0.10 / −0.09 / −0.06; **−0.08** | 4.51 / 5.24 / 5.20% | 0.9519 | 0.9779 | 3,197 | 15 s |
| **twice** (width 50, two passes) | 0.55 / 0.17 / 0.42; **0.38** | 4.44 / 5.65 / 5.63% | 0.9505 | 0.9787 | 3,113 | 17 s |
| rebuilt | **0.12** | | 0.9520 | | 3,156 | (bulk build) |

**Predictions:**

1. Excess at most 0.22: **held for wide** (−0.08, below the rebuild's 0.12, and below 0 on every
   seed). **Failed for twice** (0.38, above D13's).
2. Oldest cohort's rise against grown at most 0.45 points: **held for both on the mean** (wide +0.42,
   twice +0.35). Wide's seeds were +0.38, +0.33 and +0.55, so one seed is above the bar.
3. Recall@100 at least 0.9501: **held for both** (wide 0.9519, the same as a rebuild's 0.9520; twice
   0.9505).
4. Refine at most 2.5× D13's 8 s: **held for both** (15 s and 17 s, about 2×).

At ef 100, wide's excess is −0.74 points, and at ef 400 it is 0.00. For D13 the figures are −0.09 and
0.06.

**Reading:** the width of the refine search matters, and a second pass at the same width does not.
A new node's refine at width 50 searches a graph that is already dense around it. A wider search
finds the neighbours a bulk build would have given it, and through `connect()` it passes links back to
them. Two passes at width 50 find nearly the same candidates twice. With width 100, the newest cohort's
miss rate falls 0.25 points and the middle cohort's 0.50, while the oldest rises 0.06 points against
D13. The rise against grown, +0.42, is close to a rebuild's +0.37: again it is what removing the bias
costs that cohort. Search work is 3% above D13's and 1% above a rebuild's.

**Decision:** wide meets predictions 1–3 and goes to a full-scale run as a fifth engine arm,
`repaired8w`: 8 links, each later snapshot's insertions refined at width 100. It runs against `grown`,
`repaired8` and `rebuilt`, under decision 3's margins (excess at most 1.0 point with an interval
excluding 0, oldest rise at most 0.75, recall fall at most 0.5). Twice is dropped.
