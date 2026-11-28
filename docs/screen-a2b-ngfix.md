# Screen A2b: NGFix against the arrival bias

Recorded on 29 November 2026, before any code for it was written or run. For paper A (`plan-paper-a.md`, A2).
Exploratory.

## Why

NGFix (arXiv:2510.22316; code `github.com/yuhuifishash/NGFix`, commit `9acd987`) repairs an HNSW graph after
it is built, using training queries and their exact neighbours: it adds edges where those queries' searches fail
(NGFix) or reach their neighbours late (RFix). It is query-driven, unlike the insertion-time repair of this study
and CleanANN's bridges. This project integrated it before (`research/longeval/ngfix_driver.cpp`, PACE job
7141656) on an 8,192-vector sample and measured overall recall only. The question here is whether it removes
the arrival bias when it repairs a graph grown by appending.

## Design

**Implementation.** The existing adapter, `research/longeval/ngfix_driver.cpp`, unchanged: ngfixlib's
`HNSW_NGFix` (inner product, M 16, its second width 48 as in the adapter), serial insertion in the order the base
file lists the vectors, then four variants of the same graph (`base`, `ngfix`, `rfix`, `ngfix_rfix`; NGFix
applied at (100, 100) and (10, 10) per training query, RFix at 10, as in job 7141656), each searched at ef 100,
200, 400 and 800, k 100. Compiled on a Gold 6226 node with GCC 13.3.0 and `-march=skylake-avx512` from the
pinned source in the verified package `longeval-ngfix-screen-02` (its kernels need AVX-512).

**Data.** The two inner-product samples of screens M1 and M3: LongEval (300,000 documents, the 381 test
queries) and GloVe (300,000 vectors, 1,000 queries, arrival cohorts assigned at random). Each sample's queries are
split by seed 2029 into a training half, used only to fit NGFix and RFix, and a test half, used only to score:
190 / 191 on LongEval, 500 / 500 on GloVe. The training queries' exact top 200 over the sample (float64
inner product, ties by position) is what NGFix is given.

**Orders.** As arrived (the sample's cohorts in order) and shuffled (seed 11), as in the order study. The base
file lists the vectors in insertion order; the adapter's IDs are mapped back to sample positions.

**Measures** (test queries; ef 200 unless stated; 5,000-draw query bootstrap): the within-query miss ratio of
cohort 3 against cohort 1 and the newest cohort's within-query excess (Mantel–Haenszel, strata query × exact-rank
band), in `base` and in each repaired variant, as paired differences; recall@100; the repair's time.

## Predictions

1. In the as-arrived `base` graph the ratio is above 1.10 on both samples (an HNSW graph grown by appending).
2. `ngfix_rfix` removes less than half of the as-arrived `base` graph's newest-cohort excess on both samples: it
   repairs the graph where the training queries fail, not where the newest vectors are.
3. `ngfix_rfix` raises test recall@100 at ef 200 by at most 1 point on both samples (job 7141656 found modest
   gains).

LongEval's 191 test queries give wide intervals; the result there is read as a direction. A result against a
prediction is reported as such. Aggregates only, in `research/longeval/`.
