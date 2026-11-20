# Exploratory analyses after the held-out results (2026-11-20)

These analyses are **not** part of FINAL_PROTOCOL_V1. They were designed after the held-out
results were seen, and they describe the frozen evidence without changing it. The code is
`heldout_exploratory.py`. The outputs are in `build/longeval-heldout-exploratory-01/`
(`report.json`, `cohort-ratios.json`) and `build/longeval-development-order-01/`.

## 1. Newly inserted documents are under-retrieved by HNSW

Each document is assigned to the snapshot in which it first appears:

| Inserted at | Documents |
| --- | --- |
| Snapshot 1 | 869,902 |
| Snapshot 2 | 452,818 |
| Snapshot 3 | 339,180 |

For every exact top-100 neighbour of a test query, we record whether the ANN run returned it.
Newer documents could sit at different depths of the ranking, so misses are compared within
exact-rank bands: 1–10, 11–50 and 51–100. The *rank-standardized ratio* compares the misses
observed among documents inserted after snapshot 1 with the misses those documents would have
at the snapshot-1 cohort's rates in each band.

The intervals below come from a query bootstrap with 2,000 draws; seeds are pooled within each
query.

| Arm | Snapshot 2 | Snapshot 3 |
| --- | --- | --- |
| Epzero HNSW, incremental, ef 200 | 1.60 [1.45, 1.78] | 1.63 [1.49, 1.80] |
| FAISS HNSW, incremental, ef 200 | 1.54 [1.40, 1.70] | 1.54 [1.40, 1.69] |
| FAISS HNSW, incremental, calibrated | 1.52 [1.37, 1.71] | 1.56 [1.42, 1.72] |
| IVF, frozen training, calibrated | 1.17 [1.06, 1.30] | 1.17 [1.05, 1.29] |
| IVF, refreshed training, calibrated | 1.24 [1.13, 1.35] | 1.14 [1.03, 1.26] |

The raw rates show the same pattern. For Epzero at snapshot 3, ef 200, the miss rates in the
top-10 band are 2.7% for documents inserted at snapshot 1, 5.7% for snapshot 2 and 6.9% for
snapshot 3. For FAISS they are 2.4%, 6.0% and 7.0%.

### Insertion order or document property?

IVF has no insertion-order structure, so its ratio of about 1.15–1.25 estimates how much harder
the newer documents themselves are to retrieve. HNSW's excess is well above that.

The development evidence includes shuffled-order rebuilds, in which all documents are inserted
in a random order. The table gives the ratio on 50 development queries × 3 seeds at ef 200,
pooling every document inserted after snapshot 1, as in the held-out table:

| Arm | Snapshot 2, appended | Snapshot 2, shuffled | Snapshot 3, appended | Snapshot 3, shuffled |
| --- | --- | --- | --- | --- |
| Epzero | 1.66 | 1.08 | 1.21 | 1.11 |
| FAISS | 1.32 | 1.07 | 1.21 | 1.05 |

Shuffling lowers the ratio in all four comparisons, to 1.05–1.11, which is about the IVF level.
Most of the HNSW excess is therefore consistent with an **insertion-order effect**: in a graph
built by appending, documents inserted later are less reachable. The remainder, about 1.1, is a
property of the documents.

On the 50 development queries the appended-order excess at snapshot 3 (1.21) is smaller than on
the held-out queries (1.54–1.63). The development comparison is small, so a held-out
shuffled-order run is needed to confirm the mechanism (see "Improvements").

This is an aging mechanism that *maintenance* cannot remove. Incremental maintenance and an
identical-history rebuild are bit-identical (RQ1). Only a rebuild in a different order changes
it.

**Caveats.**
- This analysis is exploratory and was chosen after seeing the results.
- The development comparison is small: 50 queries, with 126–237 top-10 neighbours in the later
  cohorts.
- The ratio counts neighbour misses, not relevance.
- How much of the nDCG gap comes from new documents has not yet been measured.

## 2. What replaces a missed top-10 neighbour

These counts are pooled over seeds, on the primary panel of 374 queries with GPT-OSS-120B grades.
Unjudged documents count as grade 0.

| Arm (snapshot 3) | Dropped top-10 neighbours | Mean grade of dropped | Mean grade of replacement | Replacements that were judged | Mean grade of kept |
| --- | --- | --- | --- | --- | --- |
| Epzero, ef 200 | 399 | 1.01 | 0.22 | 18% | 1.54 |
| FAISS HNSW, ef 200 | 374 | 0.77 | 0.21 | 20% | 1.55 |
| FAISS HNSW, calibrated | 280 | 0.77 | 0.25 | 24% | 1.54 |
| IVF frozen, calibrated | 303 | 1.32 | 0.56 | 39% | 1.53 |

1. **Dropped neighbours are less relevant than the ones kept**, with mean grade 0.77–1.01
   against 1.54 for HNSW. Together with misses being rarer at the top of the ranking, this
   explains why the nDCG@10 gap (about 1%) is much smaller than the neighbour-recall shortfall
   (about 6%).
2. **Replacements are mostly unjudged.** Only 18–24% of HNSW replacements carry a judgment, and
   unjudged documents count as non-relevant. The released judgments were presumably pooled from
   exact-search-style runs, so the measured ANN gap is partly a pooling artifact and is likely an
   upper bound on the true loss. The paper should state this. Judging a sample of the
   replacement documents would test it (see "Improvements").

## 3. Condensed-list check of the pooling bias

The condensed-list check removes unjudged documents from every ranking, exact and approximate,
before computing nDCG@10. Any unjudged substitute then counts neither for nor against a run. The
setup is the same as the primary analysis: 374 queries, seeds averaged within each query, and
the same inference settings.

| Arm (calibrated) | Gap at s1 | Gap at s2 | Gap at s3 | D [95% interval] | p |
| --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0008 | −0.0043 | −0.0056 | −0.0048 [−0.0116, +0.0003] | 0.112 |
| FAISS HNSW | −0.0012 | −0.0006 | −0.0007 | +0.0004 [−0.0031, +0.0039] | 0.824 |
| IVF, frozen | −0.0020 | −0.0020 | −0.0017 | +0.0003 [−0.0011, +0.0017] | 0.647 |

For comparison, the standard gaps at s1 are −0.0057, −0.0048 and −0.0065.

Removing unjudged documents shrinks the gap by roughly half to three-quarters (at snapshot 1:
from −0.005 to −0.006, down to −0.001 to −0.002). The ANN loss on judged documents is about
0.001–0.006 nDCG@10.

The pattern over time is unchanged:

- FAISS HNSW and IVF stay flat.
- Epzero's gap still widens, by −0.0048. Its interval just crosses −0.01, as in the primary
  analysis.

The computation is `heldout_exploratory.condensed_gaps`; its output is in
`build/longeval-heldout-exploratory-01/condensed.json`.

## 4. Does the new-document blind spot reach effectiveness?

The 374 primary-panel queries are split by the share of their exact top 10 at snapshot 3 that
was inserted after snapshot 1:

- **fresh:** 50% or more new, 77 queries;
- **old:** less than 50% new, 297 queries.

The table gives the snapshot-3 ANN-minus-exact nDCG@10 at the calibrated settings, with seeds
averaged within each query and a query bootstrap of 10,000 draws. The computation is
`heldout_exploratory.freshness_gaps`; the output is
`build/longeval-heldout-exploratory-01/freshness.json`.

| Arm | Fresh-content queries | Old-content queries |
| --- | --- | --- |
| Epzero HNSW (ef stayed at 200) | −0.0257 [−0.0549, −0.0056] | −0.0080 [−0.0131, −0.0035] |
| FAISS HNSW (ef rose to 400 for seeds 29 and 47) | −0.0081 [−0.0171, −0.0015] | −0.0048 [−0.0095, −0.0012] |
| IVF, frozen | −0.0097 [−0.0185, −0.0027] | −0.0076 [−0.0112, −0.0045] |

For the HNSW arm held at a fixed width, the average gap of about 0.01 hides a loss about three
times larger on queries whose answers are new content. A wider search (FAISS) and IVF shrink the
difference.

**Caveats.**
- The split is post hoc.
- The fresh group is small (77 queries), so its intervals are wide.
- New-content share is correlated with other query properties.

## 5. Held-out extension: shuffled order and ef sweep (jobs 7179544–7179549)

The six extension jobs all completed and their evidence checksums verify. The rebuilt incremental
arm reproduced every analysed primary run, and the incremental ef-200 sweep runs equal the
primary fixed-ef runs. All 216 sweep runs were recounted. The analysis is
`heldout_exploratory.extension`, with output in `build/longeval-heldout-extension-analysis-01/`.

**The insertion-order mechanism is confirmed on held-out queries.** The table gives the
rank-standardized miss ratio at ef 200 (2,000-draw query bootstrap):

| Arm | Snapshot 2, appended | Snapshot 2, shuffled | Snapshot 3, appended | Snapshot 3, shuffled |
| --- | --- | --- | --- | --- |
| Epzero | 1.60 [1.45, 1.78] | 1.13 [1.04, 1.23] | 1.63 [1.49, 1.80] | 1.25 [1.15, 1.35] |
| FAISS | 1.54 [1.40, 1.70] | 1.16 [1.07, 1.26] | 1.54 [1.40, 1.69] | 1.21 [1.11, 1.31] |

The shuffled ratios are close to IVF's (1.14–1.24).

**Reordering does not improve effectiveness.**
- Shuffled recall at ef 200 at snapshot 3 is 0.933, against 0.936 incremental.
- At ef 200, the shuffled gaps (s1/s2/s3) are −0.008/−0.011/−0.013 for Epzero and
  −0.007/−0.010/−0.011 for FAISS.
- With calibration, shuffled indexes needed ef 400 more often, and their calibrated D is about 0
  (Epzero +0.0011, FAISS +0.0009).

On the freshness split at ef 200 and snapshot 3:

| Arm | Fresh-content queries, incremental | Fresh-content queries, shuffled | Old-content queries, incremental | Old-content queries, shuffled |
| --- | --- | --- | --- | --- |
| Epzero | −0.026 | −0.024 | −0.008 | −0.011 |
| FAISS | −0.009 | −0.017 | −0.006 | −0.010 |

Reordering spreads the misses evenly across document ages; it does not remove them.

**ef sweep (incremental).** Mean recall and gap, s1 → s3:

| ef | Epzero recall | Epzero gap | FAISS recall | FAISS gap |
| --- | --- | --- | --- | --- |
| 100 | 0.900 → 0.885 | −0.010 → −0.023 | 0.902 → 0.892 | −0.011 → −0.015 |
| 200 | 0.947 → 0.936 | −0.006 → −0.012 | 0.946 → 0.938 | −0.005 → −0.007 |
| 400 | 0.969 → 0.964 | −0.003 → −0.005 | 0.969 → 0.964 | −0.003 → −0.005 |
| 800 | 0.982 → 0.979 | −0.002 → −0.003 | 0.982 → 0.978 | −0.002 → −0.003 |
| 1,600 | 0.988 → 0.986 | −0.001 → −0.001 | 0.988 → 0.985 | −0.001 → −0.001 |
| 3,200 | 0.994 → 0.991 | −0.000 → −0.000 | 0.993 → 0.991 | 0.000 → −0.000 |

Above about 98% recall the gap is small, and growth no longer changes it.

**Timing (jobs 7178503–7178505; secondary outcome of the protocol).** These are native
single-query results on the 381 test queries, as the mean of per-query medians:

| Index and setting | Latency | Distance work |
| --- | --- | --- |
| Epzero, ef 200 | 2.53 → 2.66 ms (p95 3.5–3.6 ms) | 2,672 → 2,754 computations |
| FAISS, ef 200 | 2.34 → 2.44 ms | 2,720 → 2,806 (native ndis) |
| FAISS, calibrated ef 400 (snapshots 2–3) | 3.9 ms | 4,300 |
| HNSW, ef 3,200 | 31–32 ms | about 30,000 |
| IVF at the calibrated nprobe 128 (primary jobs) | 129 → 184 ms | 160k → 228k comparisons |
| IVF at the frozen snapshot-1 nprobe | up to 243 ms | 302k comparisons |
