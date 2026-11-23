# Development screen D5: is `extra_backlinks` a better recall/work trade-off overall?

Recorded on 23 November 2026, before it was run. Development data only.

## Question

Screens D3–D4 showed that `extra_backlinks` raises recall at a fixed search width for slightly more
work. Benchmarks compare indexes at matched work, along the whole recall–throughput curve. If the
option moves Epzero's curve up across the range, it is a product improvement and a candidate default,
not only a fix for arrival bias.

## Design

Epzero development build, uncompressed, M 16, efConstruction 100, one thread, seed 11, on the 300K
SIFT, GloVe and LongEval samples, as arrived and shuffled, `extra_backlinks` 0 and 8. Each build is
searched at ef 10, 20, 40, 80, 120, 200, 300, 400, 600 and 800, k = 10 and k = 100. Measures per
width: recall@10, recall@100, distance computations per query (from traces) and single-thread
queries per second. Comparison: recall at matched distance computations, by linear interpolation
between widths, at recall@10 targets 0.90, 0.95 and 0.98 (the work needed to reach each target).

## Prediction

With `extra_backlinks` 8, the work needed to reach recall@10 of 0.95 and 0.98 is at most that of
the baseline on all three datasets, as arrived and shuffled (the curve does not get worse anywhere
above recall 0.9), and is lower by at least 3% on at least two of the three datasets as arrived.
