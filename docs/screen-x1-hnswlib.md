# Cross-implementation screen X1: hnswlib

Recorded on 24 November 2026, before any code for it was written or run. Exploratory, development
data only. It is the first of the cross-implementation readings listed in section 11 of
`docs/arrival-bias-research.md` ("common to incremental graphs, or an artefact of one pruning rule").

## Why

The arrival gradient has been measured in two implementations, Epzero and FAISS HNSW. Both re-run the
pruning heuristic over a full neighbour's links when a new node asks to join them. hnswlib, the
reference HNSW library, follows the same rule (`getNeighborsByHeuristic2` on the neighbour's links plus
the new node). A third implementation with its own code base and the same rule tests whether the
gradient belongs to the rule and not to either code base. Lucene, with a different reverse-link rule,
is a later screen.

## Design

As the 300K development pilot (`Oryx/build/longeval-insertion-pilot-03-m16`, the same sample in WSL
`~/epzero-longeval/pilot/sample`): M 16, efConstruction 100, one thread, one vector at a time in each
order (`appended`, `reversed`, `shuffled`; `insertion_pilot.insertion_order`), seeds 11, 29 and 47
(hnswlib's `random_seed`), inner-product space, full-precision vectors. Searched at ef 100, 200, 400 and
800, k 100, with the sample's 381 test queries.

hnswlib comes from PyPI, the newest release, in its own virtual environment in WSL. Its version and
the archive's SHA-256 are recorded with the results. `insertion_pilot.py` is frozen (extension protocol
V1), so a new module, `research/longeval/cross_pilot.py`, builds and scores. It writes the same files
(`report.json`, `per-query-ef{w}.npy`, `per-query-position-ef{w}.npy`, `labels-ef{w}.npy`) and uses the
same scoring. A test checks that it reproduces `insertion_pilot.run` exactly on a FAISS build of a small
sample.

## Measures (fixed now, ef 200, seeds averaged within query, query bootstrap 5,000 draws)

From `insertion_strata.orders` and `insertion_strata.between` on the per-query tables:

1. The within-query (Mantel–Haenszel) miss ratio of snapshot-3 against snapshot-1 targets, as
   arrived, shuffled and reversed, and as arrived minus reversed.
2. The same ratio in hnswlib minus the mean of FAISS's and Epzero's on the same queries
   (`between`, as arrived).

## Predictions

1. As arrived, hnswlib's within-query ratio is above 1 with an interval excluding 1, and as arrived
   minus reversed is positive with an interval excluding 0.
2. Shuffled, its interval includes 1.
3. As arrived, its ratio is within ±0.15 of the mean of FAISS's and Epzero's on the same sample. The
   pruning rule is the same, so the size should be too.

If prediction 1 fails, the gradient depends on the code base and not only on the rule, and the paper's
claim is narrowed to the implementations where it holds.

## Result

Run on 24 November 2026, after the design above was committed. hnswlib 0.8.0, built from the PyPI
source archive (SHA-256 `cb6d037e…c44c`) in its own environment (numpy 2.4.6, pybind11 3.1.0). The
module's tests passed there, including the exact reproduction of `insertion_pilot run` on FAISS. Nine
builds, 339–352 s each, single-threaded, recall@100 at ef 200 of 0.941–0.946 (FAISS on the same sample:
0.946 as arrived). FAISS and Epzero are the pilot's own builds (`Oryx/build/longeval-insertion-pilot-03-m16`).
Aggregates: `research/longeval/cross-pilot-x1-hnswlib.json`; raw builds in `Oryx/build/screen-x1/`.

Within-query (Mantel–Haenszel) miss ratio of snapshot-3 against snapshot-1 targets, ef 200, seeds
averaged within query, 95% query-bootstrap intervals (5,000 draws):

| Implementation | As arrived | Shuffled | Reversed | As arrived minus reversed |
| --- | --- | --- | --- | --- |
| **hnswlib** | **1.34** [1.20, 1.51] | 1.00 [0.90, 1.10] | 0.77 [0.68, 0.86] | **0.57** [0.44, 0.72] |
| FAISS HNSW | 1.28 [1.14, 1.43] | 0.98 [0.89, 1.09] | 0.77 [0.67, 0.86] | 0.51 [0.40, 0.65] |
| Epzero | 1.32 [1.18, 1.49] | 1.00 [0.90, 1.10] | 0.76 [0.66, 0.85] | 0.57 [0.44, 0.71] |

As a within-query risk difference, hnswlib as arrived is +1.78 points [+1.07, +2.52] (FAISS +1.44,
Epzero +1.71), and reversed −1.54 [−2.23, −0.89]. hnswlib minus the mean of FAISS and Epzero, as
arrived: +0.04 [−0.00, +0.09]; shuffled +0.01 [−0.02, +0.03]; reversed +0.01 [−0.02, +0.04].

**Predictions:**

1. "As arrived above 1 with an interval excluding 1, and as arrived minus reversed positive with an
   interval excluding 0": **held** (1.34 [1.20, 1.51]; 0.57 [0.44, 0.72]).
2. "Shuffled, the interval includes 1": **held** (1.00 [0.90, 1.10]).
3. "As arrived within ±0.15 of the mean of FAISS's and Epzero's": **held** (+0.04, with the interval
   [−0.00, +0.09] inside the band).

**Reading:** a third code base with the same pruning rule shows the same arrival gradient, of the same
size, and the same reversal under the reversed order. The gradient belongs to the incremental HNSW
construction rule, not to one implementation. Lucene, whose reverse-link rule differs, is the reading
that can separate the rule from HNSW in general.
