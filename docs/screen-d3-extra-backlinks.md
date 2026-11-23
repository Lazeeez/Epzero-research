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

## Result (run 23 November 2026, after the design above was committed)

Epzero development build (`min-backlinks` branch, `extra_backlinks`), 300K sample, seed 11, 381 test
queries, exact top-100. Builds of one order ran at the same time.

| Order | extra_backlinks | Newest-cohort excess, ef 100 / 200 / 400 (pp) | Oldest-cohort miss, ef 200 | Recall@100, ef 200 | Distance computations, ef 200 | Build | In-degree, first → last tenth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| as arrived | 0 | 2.77 / 1.62 / 0.76 | 4.50% | 0.9457 | 2,811 | 334 s | 28.6 → 13.3 |
| as arrived | 4 | 1.07 / 0.78 / 0.23 | 4.58% | 0.9475 | 2,821 | 341 s | 26.1 → 16.6 |
| as arrived | 8 | 0.08 / 0.41 / 0.13 | 4.66% | 0.9482 | 2,837 | 347 s | 24.3 → 19.7 |
| shuffled | 0 | −0.37 / 0.20 / 0.00 | 5.25% | 0.9440 | 2,764 | 290 s | 29.4 → 12.4 |
| shuffled | 4 | −0.05 / 0.09 / 0.14 | 5.11% | 0.9458 | 2,775 | 296 s | 26.7 → 15.9 |
| reversed | 0 | −3.03 / −1.64 / −1.03 | 5.90% | 0.9417 | 2,736 | 283 s | 29.1 → 11.9 |
| reversed | 4 | −1.25 / −0.79 / −0.51 | 5.42% | 0.9449 | 2,751 | 289 s | 26.5 → 15.4 |

**The predictions for `extra_backlinks` 4 hold**: the newest cohort's excess at ef 200 falls by 52%
(1.62 → 0.78; at least half was predicted, so this only just passes); the oldest cohort's miss
rate rises 0.08 points (limit 0.3); recall rises 0.18 points; distance computations rise 0.4%
(limit 5%); build time rises 2% (limit 30%); the last tenth's in-degree rises 3.3 (at least 3).

**Further:** with 8, the excess falls by 75% (0.41 points), the oldest cohort's miss rate rises 0.16
points, recall rises 0.25 points, and the in-degree curve is nearly flat (24.3 → 19.7). The order
effect itself halves: as arrived minus reversed at ef 200 is 3.26 points without the option and 1.57
with 4. Recall rises in every order and at every width for 0.4–1% more distance computations, which
is far cheaper than searching wider (ef 200 → 400 costs 79% more for 2.5 points of recall).

**Limits:** one seed, one collection, development queries. Next: the public datasets (screen D4),
three seeds, then the confirmation readings.
