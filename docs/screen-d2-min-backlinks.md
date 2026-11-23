# Development screen D2: guaranteed incoming links at insertion (Epzero `min_backlinks`)

Recorded on 23 November 2026, before it was run. Development data only (the 300K LongEval sample and
the 381 test queries).

## Question

Screen D1 showed that giving late documents more incoming links removes most of their excess misses,
and that it does not matter which nearby documents supply them. D2 tests the online version in the
engine: during each insertion, a full neighbour that would drop the new node gives up its farthest
link instead, until the new node has `min_backlinks` incoming base-layer links (Epzero, development
build; option off by default). Every insertion is treated, not only the newest cohort, and the graph
is evaluated as built.

## Design

- Epzero development build (Oryx working tree with `min_backlinks`), uncompressed, M 16,
  efConstruction 100, inner product, one thread, seed 11, the pilot's configuration.
- Orders: as arrived, shuffled, reversed blocks; `min_backlinks` 0 (baseline) and 4 in every order,
  and 2 and 8 in the as-arrived order (dose).
- Measures at ef 100, 200, 400: the newest cohort's within-query excess (rate difference, snapshot 3
  against 1); the as-arrived minus reversed contrast of that excess; miss rate per cohort; overall
  recall@100; distance computations per query (from search traces); build time (pairs of builds
  run at the same time, so they share machine load); in-degree by insertion decile.

## Prediction

With `min_backlinks` 4, in the as-arrived order at ef 200: the newest cohort's excess falls by at
least half from the baseline, the oldest cohort's miss rate does not rise by more than 0.3 points,
overall recall does not fall, distance computations rise by at most 5%, and build time rises by at
most 30%. The as-arrived minus reversed contrast shrinks. Larger `min_backlinks` helps more, with
diminishing returns.

One seed: a screen. A setting it favours is tested with three seeds, then on the confirmation
readings (BGE vectors, a public streaming workload).
