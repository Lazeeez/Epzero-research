# Development screen D3: extra incoming links at insertion (Epzero `extra_backlinks`)

Recorded on 23 November 2026, before it was run. Development data only (the 300K LongEval sample and
the 381 test queries).

## Question

Screen D2 showed that the in-degree gradient is passive accumulation: old nodes keep receiving
forward links from later arrivals, which new nodes have not had. Screen D1 showed that giving late
documents extra incoming links from nearby nodes they did not choose removes most of their excess.
D3 tests the online version: during each insertion, the closest construction candidates that the new
node did not choose as neighbours are asked to link to it (`extra_backlinks` of them; a full candidate
gives up its farthest link). Every insertion is treated; links added early can later be pruned when
those candidates re-select their neighbours, so the effect on the final graph is the open question.

## Design

As screen D2 (Epzero development build, uncompressed, M 16, efConstruction 100, inner product, one
thread, seed 11; same measures), with `extra_backlinks` 0 and 4 in all three orders and 8 in the
as-arrived order.

## Prediction

With `extra_backlinks` 4, as arrived, at ef 200: the newest cohort's excess falls by at least half
from the baseline (about 1.6 points); the oldest cohort's miss rate rises by at most 0.3 points;
overall recall does not fall; distance computations rise by at most 5%; build time rises by at most
30%; the last tenth's in-degree rises by at least 3. If links added at insertion are mostly pruned
later, the in-degree and excess change little, and a periodic repair of recent batches (as in D1)
becomes the candidate instead.
