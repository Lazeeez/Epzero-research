# Screen M3: arrival bias in a Vamana (FreshDiskANN-style) graph

Recorded on 26 November 2026, before any code for it was written or run. Exploratory, development data and
public data only. Section 7.4 of the research document asks whether the arrival gradient is common to
incremental graph indexes or specific to HNSW; three HNSW implementations agree (X1). Vamana, the graph
of DiskANN and FreshDiskANN, is the main streaming alternative: no layer hierarchy, a fixed entry point,
and RobustPrune with α > 1, which keeps long-range edges.

## Implementation

DiskANN's Python package could not be installed in this environment, so the screen uses a small,
independent C++ implementation of the published FreshDiskANN insertion (Singh et al. 2021, Algorithms 1–3),
`research/longeval/vamana_driver.cpp`, compiled with g++ 13 (-O3 -march=native) in WSL. It is our code, not
DiskANN's, and is reported as such:

- **GreedySearch(s, x, L):** beam search from the entry point s with a candidate list of size L; returns the
  expanded nodes V and the best k.
- **Insert p:** V = GreedySearch(s, p, L_build); N_out(p) = RobustPrune(p, V, α, R); for each j in N_out(p),
  add p to N_out(j) and, if |N_out(j)| > R, N_out(j) = RobustPrune(j, N_out(j), α, R).
- **RobustPrune(p, V, α, R):** repeatedly keep the closest remaining candidate p* and drop every p' with
  α·d(p*, p') ≤ d(p, p'), until R are kept.
- **Entry point:** the first inserted vector, fixed (an incremental build from empty).
- **Parameters:** R 32 (HNSW M 16's base-layer degree), L_build 100 (efConstruction 100), α 1.2, squared
  Euclidean distance (the LongEval vectors and queries are unit-normalised, so the ranking equals the inner
  product's). Search with L 100, 200, 400, 800, k 100. One thread.

A test checks the driver against brute force on a small random set (recall@10 at L 200 above 0.99) and
that its graph has no node with more than R out-links.

## Data and orders

As screen X1: the LongEval 300K development sample with the 381 test queries, and the SIFT 300K public
sample with random cohorts and 1,000 queries; orders `appended`, `reversed` (deterministic: one build
each) and `shuffled` (seeds 11 and 29), from `insertion_pilot.insertion_order`.

## Measures and predictions (fixed now; L 200, seeds averaged within query, query bootstrap 5,000 draws)

From `insertion_strata.orders`: the within-query Mantel–Haenszel miss ratio of snapshot-3 (or random
cohort 3) against snapshot-1 targets in each order; and mean in-degree by insertion decile.

**Predictions** (as X1's for hnswlib), on both datasets (SIFT's cohorts are random, so there position is
the only difference between them): as arrived above 1.10, shuffled within [0.90, 1.10], reversed below
0.95; and in shuffled builds, mean in-degree of the first insertion decile exceeds the last decile's by at
least 30%. If the ratios fail, the arrival gradient is specific to HNSW's insertion rule, and that is
reported.
