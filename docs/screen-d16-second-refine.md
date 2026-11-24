# Development screen D16: a second refine pass in a one-off build

Recorded on 24 November 2026, before it was run. Product question, development data only.

## Why

In screen D14, refining every node again after each snapshot raised recall@100 at ef 200 from
0.9506 to 0.9552 on the LongEval 300K sample, above a single-pass rebuild's 0.9520. It also cost 4%
more distance computations. At a fixed width, more work alone raises recall, so that is not yet a
gain. The question is whether a second pass gives a better graph: higher throughput at equal recall.
Screen D15 found that the refine's width mattered for new batches, so a single wider pass is tested
too. The automatic build's own note (`engine/src/auto.cpp`) says half the beam gives the same graph
on GloVe.

## Design

A one-off build of each 300K sample: SIFT (L2), GloVe (inner product) and LongEval (inner product,
2,560-d), in sample order. Product settings without the principal-direction walk: 8-bit codes, vectors
kept, `fast_build`, M 16, efConstruction 100. All vectors are added on all cores, then refined (all
cores). Epzero branch `min-backlinks`; no engine change. Arms:

| Arm | Refine after the build |
| --- | --- |
| one (today's, without the walk) | one pass at width 50 |
| two | two passes at width 50 |
| wide | one pass at width 100 |

Measures: build time, split into adding and refining. Then a sweep of 17 search widths at k = 10 and
k = 100 (re-rank of the whole beam with the full vectors, one thread): recall, queries per second and
distance computations. Compared with `one` at recall@10 0.90, 0.95, 0.98 and 0.99 by interpolation,
in queries per second and in distance computations. One build per arm and dataset (seed 11). Each
arm's sweep is timed in one process, one process at a time on the laptop, nothing else running.

## Predictions

- **two against one:** at least 5% fewer distance computations at recall@10 0.98 on at least two of
  the three datasets, and no worse than 2% more at any target; queries per second in the same
  direction. Refining takes at most 2.2× as long.
- **wide against one:** within 3% in distance computations at every target on all three (the
  automatic build's note).

## Decision

If `two` meets its prediction, a `refine_passes` option for `epzero.build` is designed next. As a
default it goes through a PACE round against Glass on the VIBE datasets first, as round 11 does for
the re-rank depth. If it does not, the D14 recall rise was the extra work, and nothing changes.
