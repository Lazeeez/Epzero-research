# Development screen D14: refining the older nodes around each new batch

Recorded on 24 November 2026, before any code for it was written or run, after the second full-scale
engine arm (`scripts/pace/LONGEVAL_ENGINE_ARM.md`, package `longeval-engine-arm-02`).

## Why

With `extra_backlinks` 8 and each later snapshot's insertions refined, the full-scale development
excess fell from 3.43 to 1.17 points, still above the 1.0-point target; a rebuild gives 0.43. Going from
4 links to 8 removed less than going from 0 to 4, so more links alone will not close the rest.

The two repairs so far act only on the **new** nodes. `extra_backlinks` asks a few older candidates to
link to each insertion, once, while it is inserted. Refining recent insertions re-searches the
finished graph for the new nodes' own out-links. Neither gives the **older** nodes around a new batch a
second look at their own out-links once the batch is in. In a rebuild, and for a document that arrived
early, those nodes chose their links among every document present at the end. That is the in-link
supply (section 7.3 of the research document: passive accumulation) the newest cohort is still
missing.

## Engine option (Epzero development branch `min-backlinks`)

`refine_links(threads, ef, first, around=False)`. With `around=True` it refines, in the same pass as the
nodes inserted from `first` on, every older node (position below `first`) that one of them links to
on layer 0. Each refined node searches the current graph and reconnects as before (`connect()` merges its
current links with what it finds and re-runs the pruning heuristic), so an older node can now choose
the new nodes beside it, which gives them incoming links. The cost grows with the batch and its
neighbourhood, not with the index. Off by default, so no existing behaviour changes. The number of
nodes refined is returned.

## Design

As D13 (default build path on the 300K development sample: 8-bit codes, `fast_build`, M 16,
efConstruction 100; snapshot 1 bulk-built and refined; snapshots 2 and 3 inserted one at a time;
`extra_backlinks` 8; seeds 11, 29, 47; refine width 50; test queries; ef 100, 200, 400). Two arms, each
run after each later snapshot:

- **around:** `refine_links(first=position before the snapshot, around=True)`.
- **all:** `refine_links(first=0)`: every node refined again. This needs no new option. It is the
  upper bound of what refining the older side can do, at a cost that grows with the index.

Compared with D13's `8 links + refine new` arm (excess 0.28, oldest 4.45%, recall 0.9506) and the
rebuilt arm (0.12, 4.46%, 0.9520) on the same seeds. Measures as D13: newest-cohort within-query
excess, miss rate by cohort, recall@100 and @10, distance computations at ef 200, and insertion and
refine time. Also recorded: how many nodes `around` refines per snapshot.

## Predictions

Averaged over the three seeds, at ef 200:

1. `around` lowers the excess below D13's 0.28 points, to at most 0.20.
2. `around` raises the oldest cohort's miss rate against grown (4.09%) by at most 0.45 points, which is
   the rebuild's rise (+0.37) plus seed noise.
3. `around` keeps recall@100 at least 0.9506 (D13's).
4. `around`'s refine takes at most 3× D13's 8 s on all cores.
5. `all` gives an excess no lower than `around`'s minus 0.10 points. In other words, refining only the
   batch's neighbourhood captures most of what refining everything does.

The screen's room is small: D13 is already within 0.16 points of a rebuild, with seed spreads of
about 0.15. A result here can rule a repair out. It cannot show the full-scale gap closing.

## Decision

An arm goes to a full-scale run (a third engine-arm package, against `grown`, `repaired8` and
`rebuilt`, under decision 3's margins as before) if it passes all three tests:

- its mean excess is below D13's 0.28;
- its oldest-cohort rise is at most 0.45 points;
- its recall@100 is at least 0.9501, within the seeds' spread of D13's.

If both arms qualify, `around` goes, unless `all`'s excess is lower by more than 0.10 points; then both
go. If neither qualifies, refining the older side is not the missing piece, and that is reported.

## Result

Run on 24 November 2026, after the design above was committed (Epzero `min-backlinks` at 2d279c0e).
Ef 200, test queries of the 300K sample, means over seeds 11, 29 and 47. The D13 and rebuilt rows are
from screens D10 and D13 on the same seeds. Raw results are in `BUILD_DIR/screen-d14/*.json`.

| Arm (8 links) | Newest excess (11 / 29 / 47; mean) | Miss by cohort 1 / 2 / 3 | Recall@100 | Recall@10 | Distance computations | Refine, all cores |
| --- | --- | --- | --- | --- | --- | --- |
| refine new (D13) | 0.28 / 0.13 / 0.43; **0.28** | 4.45 / 5.74 / 5.45% | 0.9506 | 0.9781 | 3,107 | 8 s |
| **around** | 0.75 / 0.44 / 0.56; **0.58** | 4.03 / 5.40 / 5.33% | 0.9541 | 0.9794 | 3,221 | 30 s |
| **all** | 0.67 / 0.38 / 0.60; **0.55** | 3.92 / 5.29 / 5.19% | 0.9552 | 0.9807 | 3,242 | 33 s |
| rebuilt | **0.12** | 4.46 / 5.21 / 5.39%* | 0.9520 | 0.9777* | 3,156 | (bulk build) |

\* Seed 11 only (D7); the mean excess and recall@100 are over the three seeds.

`around` is not local on this sample. After snapshot 2 it refined 228,747 nodes: the 81,467 new ones
and 147,280 older ones, 94% of snapshot 1. After snapshot 3 it refined 256,943: the 61,792 new ones
and 195,151 older ones, 82% of what was there. Every seed gave the same counts to within 35 nodes.

**Predictions:**

1. Excess at most 0.20, below D13's 0.28: **failed**. The excess roughly doubled to 0.58, and rose on
   every seed.
2. Oldest cohort's rise against grown at most 0.45 points: **held**. It fell instead, by 0.06 points
   (4.03% against 4.09%).
3. Recall@100 at least 0.9506: **held**. It reached 0.9541, above the rebuild's 0.9520.
4. Refine time at most 3× D13's 8 s: **failed** (30 s, 3.7×).
5. `all`'s excess no lower than `around`'s minus 0.10: **held** (0.55 against 0.58).

**Reading:** refining the older side lowers every cohort's miss rate, and the newest cohort's least.
Against D13, the newest cohort fell 0.12 points, the oldest 0.42. Refining is better for whichever
nodes get it more often. Under `all`, a snapshot-1 document has been refined three times by the end
and a snapshot-3 document once. Under `around`, the snapshot-1 neighbourhood is refined almost as
often. This is the same asymmetry screen D8 found for the single refine after the bulk build, one
level up. The residual excess is not missing older-side refinement. A repair that refines old nodes
again re-creates the bias it was meant to remove.

**Decision:** neither arm qualifies (the excess is not below 0.28), so neither goes to a full-scale run.
`around` stays in the engine branch as an option and is not used in the paper's repair. The paper's
repair remains 8 links with each later snapshot's insertions refined.

**Also observed, not claimed:** a second refine pass over everything raises recall@100 by 0.46 points
over D13, and by 0.32 over a single-pass rebuild, for 4% more distance computations at ef 200. Whether
that is more than the extra work buys has to be measured at equal work. It is a product lead for the
build path (screen D11's line), not a fairness result.
