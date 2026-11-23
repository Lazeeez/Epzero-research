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

## Result (run 23 November 2026, after the design above was committed)

Epzero development build, 300K sample, seed 11, 381 test queries, exact top-100.

| Order | min_backlinks | Newest-cohort excess, ef 100 / 200 / 400 (pp) | Oldest-cohort miss, ef 200 | Recall@100, ef 200 | Distance computations, ef 200 | Last-tenth in-degree |
| --- | --- | --- | --- | --- | --- | --- |
| as arrived | 0 | 2.77 / 1.62 / 0.76 | 4.50% | 0.9457 | 2,811 | 13.3 |
| as arrived | 2 | 2.80 / 1.64 / 0.73 | 4.51% | 0.9456 | 2,811 | 13.3 |
| as arrived | 4 | 2.80 / 1.65 / 0.78 | 4.52% | 0.9456 | 2,811 | 13.3 |
| as arrived | 8 | 2.70 / 1.53 / 0.74 | 4.55% | 0.9456 | 2,809 | 13.3 |
| shuffled | 0 / 4 | 0.20 / 0.17 at ef 200 | 5.25% / 5.24% | 0.9440 / 0.9441 | 2,764 / 2,762 | 12.4 / 12.4 |
| reversed | 0 / 4 | −1.64 / −1.63 at ef 200 | 5.90% / 5.92% | 0.9417 / 0.9415 | 2,736 / 2,735 | 11.9 / 12.0 |

**Against the prediction: no effect.** The in-degree curve, miss rates, recall and work are unchanged
at every setting (build times equal within pairs run together: 336–342 s as arrived and shuffled,
287 s reversed).

**Why: the guarantee almost never triggers.** Measured directly on 60,000 documents of the sample
(as arrived, Epzero, M 16, efConstruction 100; every 200th insertion after the first 1,000): right
after insertion a node already has on average **12.5 incoming links** from the 12.6 neighbours it
chose; only 1.4% have fewer than 4 and 11% fewer than 8. Yet at the end of that build, in-degree by
insertion tenth runs from 28.3 to 13.7.

**Correction to the mechanism.** The in-degree gradient is **passive accumulation**, not rejection:
almost every new node is accepted by the neighbours it picks, and old nodes then keep receiving
forward links from every later arrival nearby, which a new node has not had time to receive. This is
the "passive opportunity" alternative raised in the second review; the earlier explanation (full
neighbours rejecting newcomers) is wrong as the main cause and is corrected in
`arrival-bias-research.md`. It also explains screen D1: the treated documents gained incoming links
from nodes they had not chosen, which is what later arrivals would have supplied.

**Consequence for the design.** A guarantee floor cannot help. A fix has to give new nodes incoming
links *beyond* the reverse of their own forward links, standing in for the later arrivals they have
not yet had: for example, at insertion, from the next-closest construction candidates the
diversity heuristic did not select (screen D3).
