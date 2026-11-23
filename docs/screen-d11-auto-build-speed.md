# Development screen D11: extra links and the refine pass in Epzero's automatic build

Recorded on 24 November 2026, before it was run. Product question, development data only.

## Why

In VIBE, Epzero's automatic build (`epzero.build`) took 2–3× Glass's build time (single-threaded) and was
behind the leaders at lower recall on two of three datasets. The automatic build ends with a refine pass
over every node. Screens D3–D6 showed that `extra_backlinks` raises recall and, at high recall,
throughput, for 0–5% build time. Two questions: does `extra_backlinks` 4 improve the automatic build's
speed at equal recall, and can it replace the refine pass to cut build time without losing speed?

## Design

`epzero.build` on the development branch (all other choices automatic: 8-bit codes, building on the
codes, principal directions if it picks them, re-rank, calibration), one thread for building and for
search, one process at a time on the laptop, on the 300K SIFT (L2), GloVe (inner product) and LongEval
(inner product, 2,560-d) samples, inserted in sample order. Arms:

| Arm | refine | extra_backlinks |
| --- | --- | --- |
| A (today's default) | on | 0 |
| B | on | 4 |
| C | off | 4 |
| D | off | 0 |
| E | off | 8 |

Measures: build time (the whole `epzero.build` call); then a sweep of 17 search widths at k = 10 and k = 100
with the automatic re-rank: recall, single-thread queries per second, distance computations. Compared
with arm A at recall@10 0.90, 0.95, 0.98 and 0.99 by interpolation.

## Prediction

- **B against A:** throughput at recall@10 ≥ 0.95 at least equal (within 3%), and higher by at least
  5% at 0.98 or 0.99 on at least two datasets; build time at most 10% longer.
- **C against A:** build time at most 75% of A's, and throughput at recall@10 0.95 within 5% of A's on
  all three datasets.
- **D** is slower than A at recall@10 ≥ 0.95 (the refine pass matters without extra links).

A setting is proposed as a default only if it also holds on PACE against Glass on the VIBE datasets.
