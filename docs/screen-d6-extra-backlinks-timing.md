# Development screen D6: clean timing of `extra_backlinks`

Recorded on 23 November 2026, before it was run. Development data only.

## Why

Screen D5 measured work (distance computations) reliably but not throughput: builds ran concurrently
and the width grid was coarse. A throughput claim, or making `extra_backlinks` a default, needs
sequential runs on an otherwise idle machine and a finer grid.

## Design

`research/longeval/pareto_screen.py`, one process at a time, single thread, on the 300K SIFT, GloVe
and LongEval samples, as arrived, `extra_backlinks` 0, 4 and 8, seed 11; widths 10, 15, 20, 30, 40,
60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800. Measures: recall@10 and @100, distance
computations, queries per second. Compared at recall@10 0.90, 0.95, 0.98 and 0.99 by interpolation.
The machine is a laptop under WSL; timings are indicative, and any claim about throughput is
repeated on a PACE node before it is made.

## Prediction

At recall@10 0.95 and above, `extra_backlinks` 4 and 8 need no more distance computations than the
baseline and reach at least the baseline's queries per second (within 3%) on all three datasets; at
0.98 and above, LongEval and SIFT need at least 5% less work with 8.
