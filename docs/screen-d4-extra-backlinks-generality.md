# Development screen D4: does `extra_backlinks` generalise across data and seeds?

Recorded on 23 November 2026, before it was run. Development data only.

## Design

The backlink screen (`research/longeval/backlink_screen.py`, Epzero development build with
`extra_backlinks`; uncompressed, M 16, efConstruction 100, one thread) on:

- **SIFT** (300K sample, L2, 1,000 queries) and **GloVe** (300K sample, inner product, 1,000
  queries), arrival groups assigned at random (`insertion_pilot prepare-public`): seed 11;
  `extra_backlinks` 0, 4 and 8 as arrived; 0 and 8 shuffled and reversed.
- **LongEval** (300K sample, 381 test queries): seeds 29 and 47, `extra_backlinks` 0 and 8 as arrived
  (seed 11 is screen D3).

Same measures as D3, at ef 100, 200 and 400.

## Prediction

For `extra_backlinks` 8, as arrived, at ef 200, on each dataset and seed: the newest cohort's excess
falls by at least half; the oldest cohort's miss rate rises by at most 0.3 points; overall recall
does not fall; distance computations rise by at most 5%. On SIFT, where the order effect is largest
(within-query ratio 1.92 as arrived), the as-arrived minus reversed contrast at least halves.
