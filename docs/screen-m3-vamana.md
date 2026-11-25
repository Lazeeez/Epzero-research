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

## Addendum: DiskANN's own library (26 November 2026, before any run of it)

With the author's approval, DiskANN's Python package was installed in its own WSL environment
(`~/diskann-venv`, Python 3.11): `diskannpy` 0.7.0, wheel `diskannpy-0.7.0-cp311-cp311-manylinux_2_28_x86_64.whl`,
PyPI SHA-256 cf9af7b23453e3c20551cfc7949324d1ee309bd5a8f6aee20853084299ad837d, with numpy 1.25.0. The screen
adds a second Vamana backend, `diskann`, beside the independent driver: `DynamicMemoryIndex` (FreshDiskANN's
in-memory dynamic index) with distance `l2`, graph degree 32, build complexity 100, α 1.2, one frozen start
point (the library's default), one thread, one `insert` call per vector in the insertion order (tag = sample
position + 1); searched with complexity 100, 200, 400 and 800, k 100. In-degree is read from the saved
graph file (each location's neighbour list, the frozen point excluded) through the saved tags. Same data,
orders, measures and predictions as above; the two backends' ratios are reported side by side. No DiskANN
build or search had been run on study data when this was written.

## Result (26 November 2026)

Aggregates `research/longeval/m3-vamana.json`. Within-query miss ratio of snapshot-3 (or random cohort 3)
against snapshot-1 targets at L 200, k 100, with 95% query-bootstrap intervals; mean in-degree of the first
and last insertion tenths in shuffled builds.

| Backend | Data | As arrived | Shuffled | Reversed | In-degree, first / last tenth | Recall@100 at L 200 |
| --- | --- | --- | --- | --- | --- | --- |
| independent driver | LongEval 300K | 1.32 [1.18, 1.47] | 0.94 [0.83, 1.04] | 0.65 [0.56, 0.74] | 51.9 / 17.4 | 0.937–0.943 |
| independent driver | SIFT 300K | 1.99 [1.73, 2.28] | 0.99 [0.87, 1.12] | 0.42 [0.36, 0.49] | 44.4 / 18.6 | 0.987–0.988 |
| DiskANN 0.7.0 | LongEval 300K | 0.99 [0.84, 1.16] | 0.97 [0.85, 1.11] | 0.89 [0.76, 1.03] | 47.0 / 28.1 | 0.962–0.965 |
| DiskANN 0.7.0 | SIFT 300K | 1.43 [1.16, 1.76] | 0.90 [0.74, 1.08] | 0.47 [0.35, 0.60] | 45.4 / 26.4 | 0.994–0.995 |

The driver's LongEval run has one shuffled seed (11): its second shuffled build was dropped to free the node
for screen M2, as it would have taken over two hours (the driver builds at about a fifth of DiskANN's speed).
Each saved DiskANN graph left out its first inserted vector (one in 300,000 in-degree counts).

**Predictions.** Independent driver: **all held** on both datasets (as arrived above 1.10, shuffled within
[0.90, 1.10], reversed below 0.95, first-tenth in-degree at least 1.3 times the last's: 2.4–3.0). DiskANN:
**as arrived on LongEval failed** (0.99 against above 1.10; as arrived minus reversed +0.10 [−0.04, +0.25]);
**shuffled on SIFT failed narrowly** (0.898 against at least 0.90; its interval contains 1); the other four
ratio predictions and the in-degree prediction (1.68–1.72) held.

**Reading.** The arrival gradient is not specific to HNSW: an independent implementation of FreshDiskANN's
published insertion shows it as strongly as HNSW on both datasets. DiskANN's own library shows it on SIFT but
hardly at all on LongEval. The visible difference is the floor of incoming links: DiskANN's newest tenth keeps
26–28 of them (HNSW's about 12, the driver's 17–19), and DiskANN's recall is higher at the same L. One candidate
cause, not tested here, is that DiskANN lets a neighbour list grow past R (the saved graphs report a maximum
degree above 32) before pruning it, so a newcomer's reverse links are kept more often; the frozen start point
is another difference from the driver. Which choice protects late arrivals is a question for a follow-up screen
that varies them one at a time.
