# Does the Index Matter Over Time? Approximate Search, Index History and Collection Growth in Longitudinal Dense Retrieval

**Draft for the CLEF 2027 conference track (LNCS, long paper). Version 2026-11-20.**
This draft supersedes [the development-only draft](../docs/history/manuscript-development-draft.md), which is kept unchanged.
Placeholders marked **[TIMING]** wait for the held-out native timing jobs
(`scripts/pace/LONGEVAL_HELDOUT_COST.md`). Every other number comes from verified evidence:
see `../HELDOUT_RESULTS.md` and the development result files.

## Abstract

Longitudinal evaluation campaigns such as LongEval measure how retrieval effectiveness changes as
a collection grows. Their dense baselines use exact search. Deployed dense retrieval instead uses
an approximate nearest-neighbour (ANN) index that is built once and then updated in place. We ask
whether exact-search conclusions transfer to such deployments.

We use three cumulative snapshots of LongEval-Sci 2026 (0.87M–1.66M documents, Qwen3-Embedding-4B
vectors) and compare exact search with two HNSW implementations and an IVF index. Each index is
maintained incrementally and also rebuilt from the identical insertion history, with three
construction seeds. The evaluation uses a pre-registered, hash-frozen protocol on 381 held-out
test queries.

We find the following:

- **History.** Incremental maintenance is exactly equivalent to rebuilding from the same history:
  the index states, rankings and scores are identical in all 27 comparisons.
- **Gap size.** At about 95% neighbour recall, the nDCG@10 lost to approximate search is below
  0.012 at every snapshot.
- **Gap over time.** From snapshot 1 to snapshot 3, the gap's change is negligible within ±0.01
  for two of three arms.
- **The third arm.** Its calibration set never triggered a wider search while its recall on test
  queries drifted from 94.5% to 93.5%, and its gap widened from −0.006 to −0.012.
- **Newer documents.** In exploratory analyses, HNSW misses documents inserted at later
  snapshots about 1.6 times as often as earlier ones at the same exact rank; IVF shows about 1.15
  times. On fixed-width HNSW, queries about new content lose about three times the average. A
  held-out rebuild in random insertion order removes most of the excess misses, but it does not
  improve effectiveness.
- **Search width is the lever.** An ef sweep shows the gap shrinking to about 0.002–0.004 at
  every snapshot once recall reaches about 98%.
- **Repair.** Four pre-registered graph-repair strategies fail to beat an ordinary wider search at
  equal distance work.

The practical rule is that exact-search longitudinal conclusions hold for ANN deployments to
within about 0.01 nDCG@10, provided neighbour recall is re-checked on the evaluated queries at
every snapshot. A target of about 98% recall makes the gap negligible and stable over time.

## 1 Introduction

LongEval [5, 6] evaluates retrieval systems over successive snapshots of a growing collection,
to ask whether effectiveness is robust to temporal change. Its dense baselines score every
document for every query, at every snapshot.

Production dense retrieval does not work that way. At this scale it runs on an ANN index, usually
an HNSW graph [19] or an IVF partition. The index is built once and then updated as documents
arrive, and operators rebuild it on a schedule, largely because they cannot tell whether a
long-lived index has drifted.

This leaves two questions open that matter both for evaluation methodology and for practice:

1. **Approximation over time.** Does the effectiveness lost to approximate search stay constant
   as the collection grows? If it does not, conclusions drawn with exact search do not transfer
   to deployed systems.
2. **Index history.** Does an index updated in place behave differently from one rebuilt from
   scratch?

The systems and IR literatures each stop short of the join. Systems work tracks index quality
over time in neighbour-recall units, on synthetic or unjudged workloads [11–15]. IR work tracks
effectiveness over time with exact search or a fixed index [4–7, 16]. Static studies show that
an HNSW index costs about 0.01–0.02 nDCG@10 on a fixed collection [1, 2]. Nobody has followed
that quantity across snapshots of a real, relevance-judged test collection.

We follow Stahlberg and Byrne's separation of search errors from model errors [3]. We run the
same embeddings with exact and approximate search at every snapshot, so the signed ANN-minus-exact
difference isolates the index. This adds a system-internal term to Keller et al.'s
environment-side account of temporal change [4].

Our contributions are:

- the first measurement of the ANN-minus-exact effectiveness gap over time on a relevance-judged
  collection, under a pre-registered, hash-frozen held-out protocol;
- an exact equivalence result for incremental maintenance versus rebuild under append-only growth;
- a diagnosis that ties the gap's change to neighbour-recall drift and to calibration;
- pre-registered negative results for four graph-repair strategies at equal search work;
- a fully checksummed artifact.

## 2 Related Work

**Longitudinal IR evaluation.**
- LongEval [5, 6] provides snapshot collections with time-specific judgments.
- Keller et al. [4] decompose temporal change into environment components.
- DS@GT [7] and Still Fresh? [16] study temporal robustness of retrieval pipelines.
- All of them use exact search or a single fixed index.

**Streaming dense retrieval.** PLAID SHIRTTT [8] and MURR [9] address encoder and centroid drift
in document streams, and AURORA [10] studies temporal reranking. These works change the model
or the representation. We hold the representation fixed and isolate the index.

**Dynamic ANN indexes.**
- FreshDiskANN [11] and SPFresh [12] maintain graph and partition indexes under updates.
- Quake [13] adapts partitions to workload.
- DeDrift [14] repairs IVF under content drift.
- When to Repair [15] studies repair scheduling.

They evaluate neighbour recall and throughput, not judged effectiveness.

**Graph repair.** NGFix/RFix [17] and RoarGraph [18] add query-driven edges to improve hard
queries. We screen this family against equal-work controls.

**ANN effectiveness in IR.** Lin [1] and Tu et al. [2] measure the ANN-versus-flat nDCG loss on
static collections. Insertion-order sensitivity of HNSW is known [20]; we treat it as a control,
not as a contribution.

## 3 Experimental Setup

### 3.1 Data and representation

We use LongEval-Sci 2026, CORE scientific search, which has three cumulative snapshots:

| Snapshot | Documents |
| --- | --- |
| 1 | 869,902 |
| 2 | 1,322,720 |
| 3 | 1,661,900 |

There are no deletions. Shared documents keep bitwise-identical vectors, which we audited. We use
the released Qwen3-Embedding-4B document vectors (2,560 dimensions, inner product) unchanged.

Queries are encoded with the organiser-specified model revision and prompt, on CUDA in FP32 with
TF32 off and batch size 1. A re-encoding check on the training queries reached a minimum cosine
of 0.9999999999959.

There are 100 training queries, split deterministically into 50 calibration and 50 development
queries. The 381 test queries were used only once, after the protocol was frozen (§3.4).

### 3.2 Arms

All indexes are built single-threaded on Intel Xeon Gold 6226 nodes, with construction seeds
11, 29 and 47.

| Arm | Construction | Search settings |
| --- | --- | --- |
| Exact | Float64 exhaustive top-100 | — |
| Epzero HNSW | M=16, efConstruction=100 | ef 200 fixed; calibrated ef |
| FAISS HNSW | Same as Epzero | Same as Epzero |
| FAISS IVF-Flat | nlist 1,024; 65,536 training vectors; training frozen at snapshot 1, or refreshed at each snapshot | nprobe fixed at its snapshot-1 value; calibrated nprobe; all lists |

**Calibrated settings.** A calibrated setting is the smallest ef in {100, 200, 400, 800}, or
nprobe in {1, 2, …, 1,024}, that reaches 95% mean neighbour recall@100 on the 50 calibration
queries at that snapshot.

**Maintenance modes.** Every HNSW and frozen IVF index is built in two ways:

- **incremental**, where each snapshot's new documents are inserted into the live index in
  lexical ID order;
- **identical-history rebuild**, where a fresh index receives the same chronological insertion
  sequence.

### 3.3 Measures

- **Effectiveness:** nDCG@10 with linear gains, deterministic tie-breaking, and unjudged documents
  counted as non-relevant. The implementation agrees with pytrec_eval to 1e-12.
- **Other effectiveness measures:** relevant-document R@100 and judged@10.
- **Index fidelity:** neighbour recall@100 against exact search, including the worst-decile mean
  and counts of zero-recall queries.

### 3.4 Held-out protocol

The protocol was approved and then frozen on 19 November 2026, before any test-query retrieval
ran. Freezing hashed the protocol, the configurations, the analysis code and the job scripts:
55 files. It also recorded that no held-out outcome existed.

**Primary labels.** The released GPT-OSS-120B judgments. The panel is fixed: the 374 test queries
with a positive judgment at all three snapshots.

**Primary contrast.** For each arm A among Epzero HNSW (incremental, calibrated), FAISS HNSW
(incremental, calibrated) and IVF (frozen training, incremental, calibrated):

  D_A = [nDCG(A, s3) − nDCG(exact, s3)] − [nDCG(A, s1) − nDCG(exact, s1)]

It is averaged over seeds within each query, then over queries.

**Inference.** Two-sided paired sign-randomization tests (100,000 draws), Holm-adjusted across
the three arms, and 95% paired query-bootstrap intervals (10,000 samples).

**Decision rule.** A change is *negligible* only if its whole interval lies within ±0.01. It is
*changed* if it is not negligible and its Holm-adjusted p is below 0.05. Otherwise it is
*inconclusive*.

**Sensitivity labels.** GPT-OSS-20B, raw clicks, DCTR and human labels are reported separately,
each on its own panel.

**Verification.** All 15 held-out jobs ran the frozen package with one native binary. The
analysis recounted every run's neighbour recall against the exact references, and it was run
once.

## 4 Results

### 4.1 Index history has no effect (RQ1)

The incremental index and its identical-history rebuild are identical in all 27 held-out
comparisons: both HNSW implementations and IVF, three seeds, three snapshots. They match in state
fingerprints and in every ranking and score at every search setting. The same holds in
development.

Under append-only growth, "rebuild or keep updating" is therefore not a question of accumulated
drift. It is a question of insertion order and construction randomness, which a rebuild changes
only if it changes the history.

### 4.2 Approximation over time (RQ2)

**Table 1.** Primary contrasts on 374 queries with GPT-OSS-120B labels. Figure 1(a) plots the gaps.

| Arm | Gap at s1 | Gap at s2 | Gap at s3 | D (s3 − s1) | 95% interval | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0057 | −0.0102 | −0.0117 | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| FAISS HNSW | −0.0048 | −0.0049 | −0.0055 | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| IVF, frozen training | −0.0065 | −0.0079 | −0.0080 | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

At calibrated settings, the gap is below 0.012 nDCG@10 at every snapshot for every arm. This is
consistent with the static estimates of 0.01–0.02 [1, 2].

For FAISS HNSW and frozen IVF, the gap's change across the collection's growth is negligible.
For Epzero HNSW, the gap doubles from −0.006 to −0.012. The interval for that change crosses
−0.01, so it is not negligible, and the Holm-adjusted p of 0.117 means it is not significant
either.

GPT-OSS-20B labels (377 queries) give the same picture:

- Epzero D = −0.0074 [−0.0142, −0.0019];
- FAISS D = −0.0017 [−0.0049, +0.0017];
- IVF D = −0.0014 [−0.0035, +0.0007].

On the 29 human-judged queries at snapshot 3, every gap lies within ±0.0005.

**The two HNSW implementations diverge because of calibration.** At fixed ef 200, both lose
neighbour recall at the same rate on the test queries: Epzero from 94.5% to 93.5%, FAISS from
94.4% to 93.6% (Table 2). FAISS's secondary fixed-ef D is −0.0022 [−0.0053, +0.0013].

What differs is calibration:

- For FAISS seeds 29 and 47, calibration recall at ef 200 fell to 94.4–94.6% at snapshots 2 and 3.
  Calibration therefore selected ef 400, which restored test recall to 95.3–95.7% and held the
  gap flat.
- Epzero's calibration recall stayed at 95.1–95.6% for every seed and snapshot, just above the
  target, so it never changed ef.

The change in the gap thus tracks the change in neighbour recall on the evaluated queries
(Figure 1). A calibrated setting prevents drift only if a small calibration set detects it.

**Figure 1** (`figures/fig-gap-recall.pdf`). (a) ANN-minus-exact nDCG@10 by snapshot at
calibrated settings, mean over 374 queries and three seeds, with 95% query-bootstrap intervals.
(b) Mean neighbour recall@100 on the test queries. The dashed line is FAISS HNSW held at ef 200.
Epzero's calibrated setting stayed at ef 200 at every snapshot.

### 4.3 Where approximate search fails (RQ3)

**Table 2.** Neighbour recall@100 on held-out queries, mean over three seeds.

| Arm, setting | s1 | s2 | s3 | Worst-decile mean at s3 | Zero-recall queries at s3 |
| --- | --- | --- | --- | --- | --- |
| Epzero, ef 200 | 0.945 | 0.939 | 0.935 | 0.646 | 1 |
| FAISS HNSW, ef 200 | 0.944 | 0.942 | 0.936 | 0.654 | 0 |
| FAISS HNSW, calibrated | 0.944 | 0.957 | 0.953 | 0.724 | 0 |
| IVF frozen, calibrated | 0.964 | 0.956 | 0.958 | 0.793 | 0 |
| IVF frozen, fixed nprobe | 0.964 | 0.965 | 0.966 | 0.829 | 0 |

Mean recall declines slowly with growth at a fixed setting. Most of the loss is in the tail: the
worst decile of queries recovers only 65% of their true neighbours.

In development, the zero-recall queries recover on the unchanged graph with a wider search. All
523 misses that remain at a 4,000-distance budget are true neighbours that the search never
reached, for queries that had already found other true neighbours. Failures are therefore
search-coverage failures, not unreachable regions. They depend on construction randomness: one
seed of one implementation produced the two development zero-recall queries, and other seeds did
not.

Refreshing IVF training does not dominate keeping it frozen. On held-out queries the refreshed
arm's D is −0.0017 [−0.0051, +0.0014]. With nprobe held fixed, frozen IVF recall does not decline
at all (0.964–0.966), while the scanned volume grows with the collection.

### 4.4 Newer documents are under-retrieved by HNSW (exploratory)

This analysis was designed after the held-out results were seen, so it is exploratory.

**Method.** Each document is assigned to the snapshot in which it first appears. We then compare
how often approximate search misses an exact top-100 neighbour, within exact-rank bands (1–10,
11–50, 51–100), because newer documents can sit at different depths of the ranking. The
rank-standardized miss ratio compares the misses among documents inserted after snapshot 1 with
the misses they would have at the snapshot-1 cohort's rates. Its intervals come from a query
bootstrap with 2,000 draws.

**Result.** At snapshot 3 the ratio is:

| Arm | Ratio [95% interval] |
| --- | --- |
| Epzero HNSW | 1.63 [1.49, 1.80] |
| FAISS HNSW | 1.54 [1.40, 1.69] |
| IVF, frozen training | 1.17 [1.05, 1.29] |
| IVF, refreshed training | 1.14 [1.03, 1.26] |

The raw rates show the same thing. In the top-10 band at ef 200, Epzero misses 2.7% of neighbours
inserted at snapshot 1, against 5.7% and 6.9% of those inserted at snapshots 2 and 3.

**Insertion order or the documents themselves?** IVF has no insertion-order structure, so its
ratio of about 1.15 estimates how much harder the newer documents themselves are to retrieve.
The development runs include rebuilds in a random insertion order. On the 50 development
queries, shuffling lowered the ratio in all four backend × snapshot comparisons: from 1.21–1.66
to 1.05–1.11 (Figure 2b).

A follow-up job confirmed this on the held-out queries. It rebuilt each held-out HNSW index at
every snapshot in a random insertion order (three seeds; the incremental arm reproduced the
analysed runs exactly). Shuffling lowered the snapshot-3 ratio:

- Epzero, from 1.63 to 1.25 [1.15, 1.35];
- FAISS, from 1.54 to 1.21 [1.11, 1.31].

Both are close to IVF's level. Most of the HNSW excess is therefore an insertion-order effect: in
a graph built by appending, later insertions are less reachable.

**Effect on effectiveness.** We split the 374 queries by the share of their exact snapshot-3
top 10 that is new content (inserted after snapshot 1).

- On the 77 fresh-content queries (50% or more new), Epzero's calibrated arm, which stayed at
  ef 200, loses 0.026 nDCG@10 [0.006, 0.055].
- On the 297 old-content queries it loses 0.008.
- FAISS, whose calibration widened the search, loses 0.008 and 0.005 respectively.
- IVF loses 0.010 and 0.008.

The average gap of about 0.01 therefore hides a loss about three times larger on exactly the
queries about new content.

This is an aging mechanism that maintenance cannot remove. An identical-history rebuild
reproduces it exactly (§4.1); only a rebuild in a different order changes it.

**Reordering does not fix effectiveness.** The shuffled rebuild spreads misses evenly across
document ages without reducing them. At ef 200 its snapshot-3 recall is 0.933, against 0.936
incremental.

| Arm, ef 200, snapshot 3 | Fresh-content queries, incremental | Fresh-content queries, shuffled | Old-content queries, incremental | Old-content queries, shuffled |
| --- | --- | --- | --- | --- |
| Epzero | −0.026 | −0.024 | −0.008 | −0.011 |
| FAISS | −0.009 | −0.017 | −0.006 | −0.010 |

The fresh-content loss barely moves, and old-content queries get worse. The remedy for the
new-document blind spot is therefore a wider search (§4.5), not a periodic reordered rebuild.

**Figure 2** (`figures/fig-cohort.pdf`). (a) Rank-standardized miss ratio of documents inserted
after snapshot 1, relative to snapshot-1 documents, on held-out queries at snapshot 3, with 95%
query-bootstrap intervals. (b) The same ratio on development queries: filled markers show
appended (incremental) insertion order, hollow markers a shuffled-order rebuild.

### 4.5 How wide a search, and at what cost (exploratory)

**The recall–gap curve.** A follow-up job searched every held-out HNSW index at ef 100, 200, 400,
800, 1,600 and 3,200 (Figure 3). The gap is a steep function of neighbour recall, and the slope
steepens as the collection grows.

| Epzero HNSW, incremental | ef 200 | ef 800 | ef 1,600 |
| --- | --- | --- | --- |
| Recall at snapshots 1 / 3 | 94.7% / 93.6% | 98.2% / 97.9% | about 98.6–98.8% |
| Gap at snapshots 1 / 3 | −0.006 / −0.012 | −0.002 / −0.003 | about −0.001 at every snapshot |

FAISS behaves alike: at ef 800 the gap is −0.002 to −0.003 at every snapshot. Above about 98%
recall the curves for the three snapshots coincide, and growth no longer matters.

**Cost.** We timed native single-query latency on the 381 test queries: single thread, one warmup
and three timed calls, median per query, then the mean over queries. The indexes reproduced the
analysed runs exactly.

| Index and setting | Latency | Search work |
| --- | --- | --- |
| HNSW at ef 200 | 2.3–2.7 ms (p95 3.2–3.6 ms) | about 2,700–2,800 distance computations; grows only 3% from snapshot 1 to snapshot 3 |
| FAISS HNSW at its calibrated ef 400 | 3.9 ms | about 4,300 computations |
| HNSW at ef 3,200 | 31–32 ms | about 30,000 computations |
| IVF at its calibrated nprobe (128) | 129–184 ms | 160,000–228,000 vectors scanned; grows 42% with the collection at the same nprobe |

Moving HNSW from about 95% to about 98% recall takes ef 800, which we did not time. The measured
neighbours bracket it: ef 400 costs 1.6 times the distance work of ef 200 (3.9 ms), and ef 3,200
costs about 11 times (31 ms). The 98% setting therefore stays at a few milliseconds per query.

**Figure 3** (`figures/fig-recall-gap.pdf`). The ANN-minus-exact nDCG@10 gap against neighbour
recall@100 across the ef sweep, for incrementally maintained (a) Epzero and (b) FAISS HNSW,
averaged over three seeds and 374 queries. Darker lines are later snapshots.

### 4.6 Graph repair versus wider search (RQ4, development)

We pre-registered four repair strategies, each against controls that receive the same distance
work. All were screened on development queries only.

1. **Frontier resumption** restarts an exhausted search from skipped candidates. In 200 paired
   comparisons it had 3 wins, 186 ties and 11 losses against a wider search at identical work.
2. **Witness edges** add edges learned from calibration queries. They gained +0.02 points on
   development queries at a 1,000-distance cap, which is one neighbour in 5,000.
3. **NGFix/RFix [17]** gained +0.04 to +0.18 points of recall at similar latency, with search
   effort tuned independently of the fitting queries.
4. **Pseudo-query bridging** joins answer-set fragments using encoder-generated queries. It
   lost 0.04–0.39 points at equal work, because the bridges add about 12% more edges and 11–13%
   more distance work.

Each method raises recall at a fixed search setting, but none beats an ordinary wider search at
the same number of distance computations. This matches §4.3: the failures come from limited
search coverage, and widening the search is the cheapest way to increase coverage.

## 5 Discussion

**Implications for longitudinal evaluation.** Exact-search conclusions transfer to ANN
deployments to within about 0.01 nDCG@10 under one condition: the index must be held at about
95% neighbour recall, measured on the evaluated queries, at every snapshot. Holding a search
setting fixed is not enough. Recall drifts down by about a point across the collection's growth,
and the gap follows it. A campaign that reports ANN runs should therefore report neighbour recall
per snapshot alongside nDCG.

**Implications for operators.** Rebuilding an append-only HNSW or IVF index from the same history
gains nothing, and rebuilding in a new order only redistributes misses. The decision that matters
is the search width.

- Re-calibrate it as the collection grows, with a calibration set large enough, or a target with
  enough margin, to detect a one-point drift. Our 50-query calibration with a grid that doubles at
  each step was too coarse to trigger for one implementation.
- Alternatively, target about 98% recall instead of 95%. The gap is then about 0.002–0.004 at
  every snapshot, at a cost of a few milliseconds per query.

**Why the effect is small.** Three things combine.

1. Misses are less common at the top of the ranking. At snapshot 3 at ef 200 (seed 11),
   approximate search misses about 6.3% of the exact top 100, but only 3.4–3.7% of the exact
   top 10.
2. The top-10 neighbours it misses are less relevant than those it keeps. Their mean
   GPT-OSS-120B grade is 0.77–1.01, against 1.54 for the kept neighbours (HNSW, snapshot 3, all
   seeds).
3. The replacement documents have a mean grade of only 0.21–0.25, but 76–82% of them are
   unjudged and therefore count as non-relevant.

The released judgments were presumably pooled from exact-search-style runs, so the measured ANN
gap is partly a pooling artifact.

A condensed-list analysis removes unjudged documents from every ranking before scoring. It
shrinks the snapshot-1 gaps from −0.005 to −0.006 down to −0.001 to −0.002. The pattern over time
is unchanged: FAISS and IVF stay flat (D = +0.0004 and +0.0003), and Epzero still drifts
(D = −0.0048 [−0.0116, +0.0003]). The loss on judged documents is therefore about 0.001–0.006
nDCG@10.

## 6 Limitations

- **Scope of data.** There is one collection and one embedding model, and the collection is
  append-only. The equivalence result does not cover deletions or rewritten documents. LongEval
  Web, which has deletions, is the natural next test.
- **Pooling.** Most documents that approximate search substitutes into the top 10 are
  unjudged, which biases the gap towards larger losses.
- **Labels.** The primary labels are LLM-generated. Human labels exist for only 29 queries, at
  snapshot 3 only. The click labels judge 6–38% of the top 10 results, so they cannot support
  an index comparison.
- **Calibration design.** Calibration uses 50 queries and a coarse grid. This is a design choice,
  and it is what separates the two HNSW arms.
- **Repair screens.** These are development screens, some on an 8,192-vector sample. They are
  claims about equal-work search on this data, not about every possible repair.

## 7 Conclusion

The index does matter over time, but only through neighbour recall, and only by about 0.01
nDCG@10 when recall is held near 95%. Incremental maintenance is equivalent to rebuilding.
Re-checking recall on the evaluated queries at each snapshot is what keeps exact-search
longitudinal conclusions valid for deployed ANN systems.

## Reproducibility

The artifact contains:

- the frozen protocol and its hash receipt;
- all job scripts and packagers;
- per-job evidence checksums;
- the analysis code, with tests on synthetic data and a recorded pipeline test on development
  queries.

LongEval-derived data are not redistributed. The scripts rebuild them from the official release.

## References

1. Lin. Operational advice for dense and sparse retrievers: HNSW, flat, or inverted indexes? 2024.
2. Tu, Jiang, Lin et al. Approximate nearest neighbor search and lightweight dense vector reranking in multi-stage retrieval. ICTIR 2020.
3. Stahlberg and Byrne. On NMT search errors and model errors: Cat got your tongue? EMNLP 2019.
4. Keller, Breuer and Schaer. Evaluation of temporal change in IR test collections. ICTIR 2024.
5. LongEval at CLEF 2025: Longitudinal evaluation of IR systems on web and scientific data.
6. Extended overview of the CLEF 2026 LongEval lab. CEUR Vol-4283.
7. DS@GT at LongEval: Evaluating temporal performance in web search systems and topics with two-stage retrieval.
8. Lawrie et al. PLAID SHIRTTT for large-scale streaming dense retrieval. SIGIR 2024.
9. Yang et al. MURR: Model updating with regularized replay for searching a document stream. 2025.
10. Saravanan, Salla and Amancherla. AURORA. Findings of ACL 2026.
11. Singh et al. FreshDiskANN. 2021.
12. Xu et al. SPFresh. SOSP 2023.
13. Mohoney et al. Quake: Adaptive indexing for vector search. OSDI 2025.
14. Baranchuk et al. DeDrift: Robust similarity search under content drift. ICCV 2023.
15. Mandarapu and Kunkunuru. When to repair. 2026.
16. Kuissi et al. Still fresh? 2026.
17. NGFix/RFix: Dynamically detect and fix hardness for efficient approximate nearest neighbor search. 2025.
18. Chen et al. RoarGraph. VLDB 2024.
19. Malkov and Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI 2020.
20. Elliott and Clark. The impacts of data, ordering, and intrinsic dimensionality on recall in hierarchical navigable small worlds. 2024.
