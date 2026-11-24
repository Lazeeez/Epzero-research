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
