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

## Engine change (Epzero branch `min-backlinks`)

`refine_links(threads, ef, first, around=False)`. With `around=True` it refines, in the same pass as the
nodes inserted from `first` on, every older node (position below `first`) that one of them links to
on layer 0. Each refined node searches the current graph and reconnects as before (`connect()` merges its
current links with what it finds and re-runs the pruning heuristic), so an older node can now choose
the new nodes beside it, which gives them incoming links. The cost grows with the batch and its
neighbourhood, not with the index. Off by default, so no existing behaviour changes. The number of
nodes refined is returned.

## Design

As D13 (product path on the 300K development sample: 8-bit codes, `fast_build`, M 16,
efConstruction 100; snapshot 1 bulk-built and refined; snapshots 2 and 3 inserted one at a time;
`extra_backlinks` 8; seeds 11, 29, 47; refine width 50; test queries; ef 100, 200, 400). Two arms, each
run after each later snapshot:

- **around:** `refine_links(first=position before the snapshot, around=True)`.
- **all:** `refine_links(first=0)`: every node refined again. This needs no engine change. It is the
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
