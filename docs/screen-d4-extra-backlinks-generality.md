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

## Result (run 23 November 2026, after the design above was committed)

`extra_backlinks` 8 against 0, as arrived, ef 200 (full rows, including ef 100 and 400, shuffled and
reversed builds, and `extra_backlinks` 4, are in `BUILD_DIR/screen-d4/*/eb*/*/report.json`):

| Data | Newest-cohort excess (pp) | Oldest-cohort miss | Recall@100 | Distance computations | Build | In-degree, first → last tenth |
| --- | --- | --- | --- | --- | --- | --- |
| GloVe, seed 11 | 3.37 → 0.44 (−87%) | 19.39% → 19.60% | 0.7920 → 0.7995 | 3,635 → 3,720 (+2.3%) | 46 → 49 s | 39.0 → 13.4 becomes 35.6 → 19.1 |
| SIFT, seed 11 | 1.34 → 0.22 (−84%) | 1.43% → 1.49% | 0.9806 → 0.9841 | 2,684 → 2,824 (+5.2%) | 34 → 37 s | 28.7 → 12.5 becomes 24.6 → 19.6 |
| LongEval, seed 29 | 1.77 → 0.58 (−67%) | 4.53% → 4.66% | 0.9454 → 0.9479 | 2,830 → 2,848 (+0.6%) | 285 → 296 s | 28.6 → 13.3 becomes 24.3 → 19.7 |
| LongEval, seed 47 | 1.75 → 0.48 (−73%) | 4.67% → 4.72% | 0.9439 → 0.9475 | 2,812 → 2,837 (+0.9%) | 285 → 297 s | same |

As arrived minus reversed, at ef 200: SIFT 2.62 → 0.33 points (−87%); GloVe 7.89 → 2.23 (−72%).

**Predictions:** held on every dataset and seed for the excess (at least halved), the oldest cohort
(at most +0.3 points: +0.05 to +0.21), recall (rises on all, 0.25–0.75 points), and the SIFT
contrast (at least halved). **One prediction failed narrowly:** distance computations on SIFT rose
5.2%, above the predicted 5%; on the other three they rose 0.6–2.3%.

Recall also rises in shuffled and reversed builds (GloVe 0.7929 → 0.7997 shuffled; SIFT 0.9810 →
0.9839 shuffled), so the option looks like a general recall-per-work gain for incrementally built
graphs, not only a fairness fix. That needs matched-work recall/throughput curves to confirm.

**Limits:** development data; the LongEval seeds share one sample; matched-work comparisons and the
confirmation readings are still to come.
