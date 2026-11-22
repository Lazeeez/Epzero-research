# Does the Index Matter Over Time? Approximate Search, Index History and Collection Growth in Longitudinal Dense Retrieval

**Draft for the CLEF 2027 conference track (LNCS, long paper). Version 2026-11-21.**
This draft supersedes [the development-only draft](../docs/history/manuscript-development-draft.md), which is kept unchanged.
Every number comes from verified evidence: see `../HELDOUT_RESULTS.md`, `../HELDOUT_EXPLORATORY.md`
and the development result files. Results marked *exploratory* were designed after the held-out
results were seen.

## Abstract

Longitudinal evaluation campaigns such as LongEval measure how retrieval effectiveness changes as
a collection grows. Their dense baselines use exact search. Deployed dense retrieval instead uses
an approximate nearest-neighbour (ANN) index that is built once and then updated in place. We ask
how much approximation changes what such a campaign measures.

We use three cumulative snapshots of LongEval-Sci 2026 (0.87M–1.66M documents, Qwen3-Embedding-4B
vectors) and compare exact search with two HNSW implementations and an IVF index, with three
construction seeds, under a pre-registered, hash-frozen protocol on 381 held-out test queries.
Our exact search reproduces the lab's official dense baseline to within 0.0014 nDCG@10. An index
maintained incrementally is identical to one rebuilt from the same insertion history in all 27
comparisons, so the differences below come from approximation and insertion order, not from
accumulated drift.

**The frozen longitudinal measurement.** Under the organisers' LLM relevance labels, and at about
95% neighbour recall, approximate search loses 0.005–0.012 nDCG@10 at each snapshot. From
snapshot 1 to snapshot 3, the change in that loss met the pre-specified ±0.01 margin for two of
the three calibrated arms: FAISS HNSW (−0.0007 [−0.0038, +0.0027]) and IVF (−0.0015 [−0.0037,
+0.0005]). For Epzero HNSW the change was −0.0059 [−0.0129, −0.0006], inconclusive under the frozen
rule. Its calibration never widened the search; at the next wider setting its change lies within
the margin (exploratory). For scale, exact search's own change over the same snapshots is
+0.0070 [−0.0000, +0.0139].

**The insertion-order result** (exploratory). Compared with first arrivals of the same query at
the same exact rank, HNSW misses documents inserted at later snapshots 1.25–1.33 times as often;
IVF 1.08–1.10 times. The excess follows the insertion position. On the full collection, the
latest arrivals are missed 1.28–1.42 times as often as the first when the index is built as they
arrived, as often (1.00–1.03) when it is built in random order, and 0.82 times as often when they
are inserted first. In shuffled builds the miss rate rises steadily with insertion position, from
5% in the first tenth of insertions to 7.4–7.8% in the last. The effect of position replicates
on two public benchmarks with randomly assigned arrival groups. Pooled over queries the ratios
are higher, about 1.6 as arrived, because queries whose neighbours are recent documents are
harder in every order. The contrast recorded before the full-collection run, the pooled
as-arrived ratio minus the reversed one, is +0.62 [+0.46, +0.80] and +0.77 [+0.57, +0.98] for the
two implementations.

Further exploratory findings:

- **Recall drift and calibration.** At a fixed search width, neighbour recall falls by about one
  point while the collection nearly doubles (94.5% to 93.5%). In a resampling study at
  snapshot 3, a 50-query calibration set keeps the too-narrow width in about one draw in five and
  a 200-query set in about one in 25. A separate paired test detects the decline between
  snapshots in 89–90% of draws with 200 queries.
- **Concentration.** At snapshot 3, 82–89% of queries lose nothing and the worst 5% carry 71–85%
  of the loss. On fixed-width HNSW, queries about new content lose about three times the average.
- **Search width.** The loss shrinks to about 0.002–0.004 at every snapshot at ef 800, where
  recall is about 98%, at about 3.5 times the latency of ef 200 (8.4–9.2 ms against 2.3–2.7 ms
  per query on one thread). Four pre-registered
  graph-repair strategies fail to beat an ordinary wider search at equal distance work.

**What the labels leave open.** The labels are lenient: on the 29 queries that also have human
grades, they agree with them at a kappa of 0.24. They cover a pool of the submitted systems, so
most documents that an HNSW index substitutes into a top 10 have no grade and count as
non-relevant: at snapshot 3 and ef 200, 18% of Epzero's replacement documents had one. With
ungraded documents removed from every ranking, its measured loss changes from −0.0117 to
−0.0056. The relevance of the ungraded substitutes is unknown, so neither figure bounds the true
loss; a follow-up grades the whole pool with one assessor.

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
  collection, under a pre-registered, hash-frozen held-out protocol, with an external check
  against the lab's official baseline;
- an exact equivalence result for incremental maintenance versus rebuild under append-only growth;
- a diagnosis that ties the gap's change to neighbour-recall drift, and a measurement of how many
  calibration queries are needed to detect that drift;
- evidence that an HNSW index built by appending under-retrieves its latest arrivals, that
  insertion position accounts for this within a query, and that a pooled comparison overstates it
  because queries about recent documents are harder;
- an account of what the released labels can support, including how much of the measured gap is
  judgment coverage;
- pre-registered negative results for four graph-repair strategies at equal search work;
- a fully checksummed artifact.

## 2 Related Work

**Longitudinal IR evaluation.**
- LongEval [5, 6] provides snapshot collections with time-specific judgments.
- Keller et al. [4] decompose temporal change into environment components.
- DS@GT [7] and Still Fresh? [16] study temporal robustness of retrieval pipelines.
- All of them use exact search or a single fixed index.

**Streaming dense retrieval.** PLAID SHIRTTT [8] and MURR [9] address encoder and centroid drift
in document streams. AURORA [10] keeps an HNSW graph and refits its quantizer when drift appears,
on synthetic drift. These works change the model or the representation. We hold the representation
fixed and isolate the index.

**Dynamic ANN indexes.**
- FreshDiskANN [11] and SPFresh [12] maintain graph and partition indexes under updates.
- Quake [13] adapts partitions to workload.
- DeDrift [14] repairs IVF under content drift.
- When to Repair [15] studies repair scheduling.

They evaluate neighbour recall and throughput, not judged effectiveness.

**Graph repair.** NGFix/RFix [17] and RoarGraph [18] add query-driven edges to improve hard
queries. We screen this family against equal-work controls.

**ANN effectiveness in IR.** Lin [1] and Tu et al. [2] measure the ANN-versus-flat nDCG loss on
static collections. Zerhoudi et al. [21] compare flat, HNSW and IVF search on MS MARCO subsets of
increasing size. Their subsets are static and sampled independently, with fixed search settings
and no control of neighbour recall, so they have no time axis, no index maintenance and no
insertion order.

**Insertion order.** Elliott and Clark [20] show that HNSW recall depends on the order of
insertion. We add which documents bear the loss when a collection grows by appending: the latest
arrivals do, and inserting them first reverses it (§4.4).

**LLM relevance labels.** The lab's overview [6] reports that its LLM assessors grade far more
documents as relevant than click-derived labels do. We quantify this for the labels we use
(§4.6), because it limits what a label-based gap can show.

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
  counted as non-relevant. The implementation agrees with pytrec_eval to 1e-12. Throughout,
  "effectiveness" means effectiveness under the stated label set; §4.6 examines how far the
  labels carry.
- **Other effectiveness measures:** relevant-document R@100 and judged@10.
- **Index fidelity:** neighbour recall@100 against exact search, including the worst-decile mean
  and counts of zero-recall queries.

### 3.4 Held-out protocol

The protocol was approved and then frozen on 19 November 2026, before any test-query retrieval
ran. Freezing hashed the protocol, the configurations, the analysis code and the job scripts:
55 files. It also recorded that no held-out outcome existed.

**Primary labels.** The released GPT-OSS-120B judgments. The organisers graded a depth-10 pool
of the 24 systems submitted to the lab, from the query alone, with a prompt adapted from UMBRELA
[6, 22]; the grade of a query–document
pair is the same at every snapshot where it appears. The panel is fixed: the 374 test queries
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

**External check.** Over every query that each label set judged, our exact search scores an
nDCG@10 of 0.7896 / 0.7953 / 0.7989 at the three snapshots under the GPT-OSS-120B labels. The
lab's overview reports 0.791 / 0.796 / 0.800 for its dense baseline, which uses the same vectors
[6]. Under the click-derived DCTR labels the values are 0.2334 / 0.2015 / 0.1330 against
0.234 / 0.201 / 0.133. The largest difference is 0.0014.

**Table 1.** Primary contrasts on 374 queries with GPT-OSS-120B labels. Figure 1(a) plots the gaps.

| Arm | Gap at s1 | Gap at s2 | Gap at s3 | D (s3 − s1) | 95% interval | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0057 | −0.0102 | −0.0117 | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| FAISS HNSW | −0.0048 | −0.0049 | −0.0055 | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| IVF, frozen training | −0.0065 | −0.0079 | −0.0080 | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

At calibrated settings, the gap is below 0.012 nDCG@10 at every snapshot for every arm. This is
consistent with the static estimates of 0.01–0.02 [1, 2].

**The margin against the trend it protects.** *(Exploratory.)* Exact search's own nDCG@10 on the
panel is 0.8043 / 0.8102 / 0.8113, a change of +0.0070 [−0.0000, +0.0139] from snapshot 1 to
snapshot 3 (p = 0.051). The ±0.01 margin is therefore as large as the trend itself, and the trend
is at the edge of what 374 queries can detect. The arms' own changes are +0.0063 [−0.0014,
+0.0139] for FAISS HNSW, +0.0054 [−0.0016, +0.0125] for IVF and +0.0010 [−0.0087, +0.0100] for
Epzero HNSW. This collection can show that the loss to approximation did not move by more than the
margin. It cannot show that a longitudinal trend is preserved.

For FAISS HNSW and frozen IVF, the gap's change across the collection's growth is negligible.
For Epzero HNSW, the gap doubles from −0.006 to −0.012. The interval for that change crosses
−0.01, so it is not negligible, and the Holm-adjusted p of 0.117 means it is not significant
either.

GPT-OSS-20B labels (377 queries) give the same picture:

- Epzero D = −0.0074 [−0.0142, −0.0019];
- FAISS D = −0.0017 [−0.0049, +0.0017];
- IVF D = −0.0014 [−0.0035, +0.0007].

On the 29 human-judged queries at snapshot 3, every gap lies within ±0.0005.

**Would the index have changed the lab's results?** *(Exploratory.)* We placed the official dense
baseline, served through each index, among the other 21 official systems, which keep their
published scores. Under the LLM labels the baseline keeps rank 5 at every snapshot with every
index. Under DCTR it keeps ranks 16 and 20 at snapshots 1 and 2; at snapshot 3, where four
systems lie within 0.002 of each other, it drops from rank 18 to 19 for two of the four index
settings. The lab's robustness measure, the relative change in nDCG@10 from snapshot 1, moves by
at most 0.02 on DCTR, while it ranges from 0.17 to 0.56 across all but two of the official
systems at snapshot 3.

**The loss is concentrated.** *(Exploratory.)* The mean gap is not a small loss spread over all
queries. At snapshot 3, 82–89% of the 374 queries have the same nDCG@10 as exact search,
depending on the arm, and the 18 worst queries (5%) carry 71–85% of the total loss. Their mean
loss is 0.12–0.22, and the largest single loss is 0.87.

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

**Fifty queries cannot detect a one-point drift.** *(Exploratory.)* Per-query neighbour recall at
ef 200 has a standard deviation of 0.11–0.12 on the test queries, so the mean of 50 queries has a
standard error of about 1.6 points. We drew calibration sets from the test queries and applied
the protocol's rule at snapshot 3, where the true recall at ef 200 is 93.5–93.6%:

| Calibration queries | 25 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- |
| Rule keeps ef 200 (Epzero / FAISS) | 32% / 32% | 21% / 22% | 11% / 12% | 4% / 4% | 1% / 1% |
| Paired test detects the fall from snapshot 1 (Epzero / FAISS) | 35% / 33% | 51% / 51% | 73% / 73% | 89% / 90% | 96% / 98% |

The second row tests the per-query change between snapshots instead of the level (one-sided
paired t-test at 5%). It removes the variation between queries: recall at snapshots 1 and 3
correlates at 0.92–0.95 within a query, which halves the standard error.

The two rows answer different questions. The first is about choosing a sufficient width at one
snapshot: with 200 queries the rule rarely keeps the too-narrow width. The second is about
detecting a decline between snapshots: with 200 queries the paired test finds it in 89–90% of
draws. Both resample the 381 test queries; the protocol's own calibration set is 50 training
queries. Sampling noise is enough to explain why Epzero's calibration stayed at ef 200.

A safety margin on the target is a second remedy, and it has a price. Asking for 96% instead of
95% on 50 queries cuts the risk of keeping a too-narrow width from 21% to 5.5%. It also raises
the mean chosen width from 416 to 522, and the share of draws in which no width up to 800
qualifies from 1.0% to 4.8%.

**At the same width the two implementations behave alike.** *(Exploratory.)* The held-out sweep
(§4.5) lets us hold the width fixed. At ef 400, the width that FAISS's calibration moved to,
Epzero's change is −0.0026 [−0.0059, +0.0002], inside the ±0.01 margin, against
−0.0022 [−0.0053, +0.0005] for FAISS. At ef 200 the two changes do not differ
reliably (−0.0038 [−0.0101, +0.0010], p = 0.20). The change shrinks
steadily as the search widens, from −0.0125 at ef 100 to −0.0006 at ef 800 for Epzero.
The inconclusive result of the third arm therefore belongs to the width its calibration kept.

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
the misses they would have at the snapshot-1 cohort's rates. This ratio pools all queries. We
also report a within-query ratio: the Mantel–Haenszel risk ratio stratified by query and rank
band [23], which compares a later document only with snapshot-1 documents retrieved for the same
query at the same depth. We added it after the full-collection results below were seen, to
separate the documents from the queries that retrieve them. Intervals come from a query
bootstrap.

**Result.** At snapshot 3 the two ratios are:

| Arm | Pooled ratio | Within-query ratio |
| --- | --- | --- |
| Epzero HNSW | 1.63 [1.49, 1.80] | 1.33 [1.21, 1.47] |
| FAISS HNSW | 1.54 [1.40, 1.69] | 1.25 [1.14, 1.38] |
| IVF, frozen training | 1.17 [1.05, 1.29] | 1.10 [1.01, 1.20] |
| IVF, refreshed training | 1.14 [1.03, 1.26] | 1.08 [0.99, 1.19] |

The raw rates show the same thing. In the top-10 band at ef 200, Epzero misses 2.7% of neighbours
inserted at snapshot 1, against 5.7% and 6.9% of those inserted at snapshots 2 and 3.

**Queries about recent documents are harder.** The pooled ratio is higher than the within-query
one because the queries differ. Split by the share of their exact top 100 that arrived after
snapshot 1, the third of queries with the fewest later neighbours misses 2.6–2.7% of its
snapshot-1 neighbours in a shuffled build, and the other two thirds miss 7.4–8.8%. Later
documents are concentrated in the harder queries, and the pooled ratio counts that difficulty as
an excess of the documents. On the public benchmarks below, where arrival groups are random, the
two ratios agree to within 0.04.

**Insertion order or the documents themselves?** The development runs include rebuilds in a
random insertion order. On the 50 development queries, shuffling lowered the pooled ratio in all
four backend × snapshot comparisons: from 1.21–1.66 to 1.05–1.11 (Figure 2b).

A follow-up job confirmed this on the held-out queries. It rebuilt each held-out HNSW index at
every snapshot in a random insertion order (three seeds; the incremental arm reproduced the
analysed runs exactly). Shuffling lowered the snapshot-3 ratio:

- Epzero, from 1.63 to 1.25 [1.15, 1.35];
- FAISS, from 1.54 to 1.21 [1.11, 1.31].

Most of the pooled HNSW excess is therefore an insertion-order effect: in a graph built by
appending, later insertions are less reachable. What remains after shuffling is the query
difficulty described above.

**Inserting the latest arrivals first reverses their excess.** Shuffling shows that the order
matters. It does not show which position is the bad one. We therefore built both HNSW
implementations on all 1,661,900 snapshot-3 documents (study settings, three seeds) in three
orders: as the documents arrived, shuffled, and with the three arrival blocks reversed. The
design and its pooled contrast were recorded before the output was seen. The as-arrived builds
reproduce the held-out runs query for query. The table gives both ratios of snapshot-3 arrivals
against snapshot-1 arrivals at ef 200, as the mean over seeds with 95% intervals (Figure 4):

| Implementation | Ratio | As arrived | Shuffled | Reversed | As arrived − reversed, paired |
| --- | --- | --- | --- | --- | --- |
| FAISS HNSW | pooled | 1.60 [1.40, 1.81] | 1.23 [1.08, 1.39] | 0.97 [0.84, 1.11] | +0.62 [+0.46, +0.80] |
| FAISS HNSW | within query | 1.28 [1.12, 1.47] | 1.03 [0.93, 1.14] | 0.82 [0.72, 0.92] | +0.46 [+0.33, +0.62] |
| Epzero HNSW | pooled | 1.75 [1.54, 1.98] | 1.20 [1.05, 1.35] | 0.98 [0.85, 1.12] | +0.77 [+0.57, +0.98] |
| Epzero HNSW | within query | 1.42 [1.25, 1.63] | 1.00 [0.90, 1.10] | 0.82 [0.72, 0.92] | +0.60 [+0.44, +0.79] |

The recorded contrast is the pooled paired difference. It is above zero in all 24 comparisons of
implementation, seed and search width from 100 to 800 (lowest interval bound +0.38). The pooled
reversed ratio is about 1, not below it, because the pooled level includes the query difficulty:
it is 1.2 even in a shuffled build.

Within a query the three orders are symmetric. A later document is missed as often as a first
arrival when its position is random, 1.28–1.42 times as often when it is inserted last, and 0.82
times as often when it is inserted first. The reversed interval lies below 1 at all four search
widths (ratios of 0.66–0.82). Collection size does not change this. Three stratified draws of
869,902 documents and a 300,000-document sample give within-query ratios of 1.23–1.42 as arrived,
0.96–1.07 shuffled and 0.76–0.83 reversed, and the full collection differs from the mean of the
draws by at most 0.04 in any order.

The graph shows why position matters. Base-layer incoming links fall with insertion position in
every order and in both implementations, from 28.0–29.6 for the first tenth of insertions to
11.5–13.7 for the last tenth. At most 0.06% of documents have no incoming link, so late insertions are
weakly connected, not unreachable. Recall@100 is 0.929–0.938 in every order: the order decides
who bears the misses far more than how many there are.

**A denser graph sharpens the effect.** We repeated the full-collection design with a denser graph
(M 32, efConstruction 200), recording the design and its measure before the jobs ran; a pilot on a
300,000-document sample had pointed the same way. The denser graph finds more neighbours (recall@100
at ef 200 of 0.961–0.969 against 0.929–0.938), but the misses that remain fall more heavily on the
latest arrivals. Within a query at ef 200 they are missed 1.65–1.66 times as often as first
arrivals when inserted last (1.28–1.42 at M 16), 0.99 times at a random position, and 0.67–0.69
times when inserted first (0.82). At every width the as-arrived interval lies above 1, the
reversed interval below 1 and the shuffled interval includes 1. Tuning the graph for recall does
not remove the effect of insertion position; it sharpens it.

**Position alone is enough.** In a shuffled build the position of a document in the insertion
sequence is independent of the document and of the query. The miss rate still rises steadily with position: on the
full collection at ef 200 it goes from 4.9–5.0% in the first tenth of insertions to 7.4–7.8% in
the last, a rank-standardized ratio of 1.49 [1.33, 1.72] for FAISS and 1.60 [1.42, 1.85] for
Epzero (Figure 5a). The interval lies above 1 for every later decile.

**The effect is not specific to this collection.** We ran the same design on two public
benchmarks, GloVe (cosine) and SIFT (Euclidean), with arrival groups assigned at random so that
they differ only in where they are inserted. The shuffled ratio is then 0.92–1.03, as it
should be. The as-arrived ratio is 1.09–1.30 on GloVe and 1.31–2.25 on SIFT across the four
search widths, and the reversed ratio 0.72–0.85 and 0.33–0.63; the reversed interval lies
below 1 and the paired difference above zero in all 16 cases. The decile curves have the same
shape (Figure 5b, c). The under-retrieval of late insertions is a property of building an HNSW
graph by insertion, in both implementations.

**Effect on effectiveness.** We split the 374 queries by the share of their exact snapshot-3
top 10 that is new content (inserted after snapshot 1).

- On the 77 fresh-content queries (50% or more new), Epzero's calibrated arm, which stayed at
  ef 200, loses 0.026 nDCG@10 [0.006, 0.055].
- On the 297 old-content queries it loses 0.008.
- FAISS, whose calibration widened the search, loses 0.008 and 0.005 respectively.
- IVF loses 0.010 and 0.008.

The average gap of about 0.01 therefore hides a loss about three times larger on exactly the
queries about new content. Insertion position is only part of the reason: these are the queries
that are harder in every insertion order, which is why reordering does not help them (below).

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

**Figure 5** (`figures/fig-position-full.pdf`). Rank-standardized miss ratio of each insertion
decile against the first, in shuffled builds at ef 200, for (a) all of LongEval snapshot 3,
(b) GloVe and (c) SIFT. Means over three seeds with 95% query-bootstrap intervals.

**Figure 4** (`figures/fig-insertion-order.pdf`). Miss ratio of snapshot-3 arrivals against
snapshot-1 arrivals by insertion order, on all 1,661,900 snapshot-3 documents at ef 200:
(a) pooled over queries, (b) within the same query and rank band. Markers are means over three
seeds with 95% query-bootstrap intervals.

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
| HNSW at ef 100 | 1.3–1.4 ms | about 1,500–1,600 computations |
| HNSW at ef 200 | 2.3–2.7 ms (p95 3.2–3.6 ms) | about 2,700–2,800 distance computations; grows only 3% from snapshot 1 to snapshot 3 |
| FAISS HNSW at its calibrated ef 400 | 3.9 ms | about 4,300 computations |
| HNSW at ef 400 | 4.5–5.0 ms | about 4,800–5,000 computations |
| HNSW at ef 800 | 8.4–9.2 ms (p95 11.0–12.0 ms) | about 8,800–9,200 computations |
| HNSW at ef 1,600 | 15.9–17.1 ms | about 16,200–16,800 computations |
| HNSW at ef 3,200 | 31–32 ms | about 30,000 computations |
| IVF at its calibrated nprobe (128) | 129–184 ms | 160,000–228,000 vectors scanned; grows 42% with the collection at the same nprobe |

Moving HNSW from about 95% to about 98% recall takes ef 800, at about 3.5 times the latency of
ef 200 (8.4–9.2 ms against 2.3–2.7 ms at snapshots 1 to 3). Latency follows the distance work,
and growth adds 2–7% at a fixed width. The widths other than ef 200, the calibrated one and
3,200 were timed in a second run (exploratory), which repeated those three to within 0.5%.

**Figure 3** (`figures/fig-recall-gap.pdf`). The ANN-minus-exact nDCG@10 gap against neighbour
recall@100 across the ef sweep, for incrementally maintained (a) Epzero and (b) FAISS HNSW,
averaged over three seeds and 374 queries. Darker lines are later snapshots.

### 4.6 What the labels can support (exploratory)

The gaps above are measured under the organisers' LLM labels. Two properties of those labels
limit what the gaps show.

**Coverage.** The labels cover a depth-10 pool of the submitted systems, which includes the
exact-search baseline. Exact search therefore has 99.5–99.6% of its top 10 graded at every
snapshot. Epzero's share falls from 97.6% to 96.6%, because 76–82% of the documents that an HNSW
index substitutes into the top 10 are ungraded and count as non-relevant. Over all rankings and
snapshots, the top 10 of exact search and of every primary run contain 5,383 distinct
query–document pairs on the panel, of which 426 were never graded.

Two rescoring rules bracket the effect without any new judgment. The *carried* rule gives an
ungraded occurrence the grade that the organisers gave the same pair at another snapshot. The
*condensed* rule removes ungraded documents from every ranking before scoring.

| Arm (calibrated) | Gap at s3: official | carried | condensed | D: official | carried | condensed |
| --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0117 | −0.0089 | −0.0056 | −0.0059 | −0.0034 | −0.0048 |
| FAISS HNSW | −0.0055 | −0.0033 | −0.0007 | −0.0007 | +0.0015 | +0.0004 |
| IVF, frozen | −0.0080 | −0.0046 | −0.0017 | −0.0015 | +0.0019 | +0.0003 |

With ungraded documents removed, the gap is 0.001–0.006 nDCG@10, about half of the official gap or
less. Removing them also promotes lower-ranked documents, so this is a different quantity and not
the loss on a fixed top 10. The pattern over time is the same under all three rules: FAISS and IVF
stay flat, and only Epzero, whose search was never widened, drifts. The carried rule adds 131
grades at snapshot 3 but only 2 at snapshot 1, so its D is biased upwards and is shown for
completeness only. Deciding how much of the gap is real needs grades for the 426 pairs, which is
the planned follow-up.

**Leniency.** For 29 test queries the release also holds human grades at snapshot 3 (1,581
pairs). The GPT-OSS-120B labels grade 87% of these pairs as relevant, the human grades 60%, and
71% of the pairs that the human grades call non-relevant are relevant under the LLM labels.
Binary agreement is 0.70 [0.62, 0.78] and the linear-weighted kappa is 0.24 [0.14, 0.36]. Exact
search scores 0.93 nDCG@10 under the LLM labels and 0.56 under the human grades on these queries,
and the per-query values are uncorrelated. The comparison is not clean: the human grades assess a
formalized topic and the LLM labels the query alone. It still shows that nDCG@10 under these
labels sits near its ceiling and separates rankings poorly.

Under lenient labels a substitute costs little when it is graded and its full gain when it is
not, so coverage has a large part in the measured gap. None of these figures bounds the true
loss: the grades of the ungraded substitutes are unknown, and new grades would also change the
ideal ranking that every score is divided by. The results on neighbour recall and on arrival
groups (§4.1, §4.3, §4.4) do not depend on any label set. The effectiveness results are
measurements under the organisers' labels and their pool.

### 4.7 Graph repair versus wider search (RQ4, development)

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

**Implications for longitudinal evaluation.** Two of the three calibrated arms met the ±0.01
margin for the change in the gap, and the third was inconclusive. The loss at each snapshot, its
change over time and the recall that was actually reached on the test queries are three
different quantities, and only the second was the frozen decision. Holding a search setting fixed
is not enough to keep them stable: recall drifts down by about a point across the collection's
growth, and the gap follows it. A campaign that reports ANN runs should therefore report
neighbour recall per snapshot alongside nDCG.

**Implications for operators.** Rebuilding an append-only HNSW or IVF index from the same history
gains nothing, and rebuilding in a new order only redistributes misses. The decision that matters
is the search width.

- Re-calibrate it as the collection grows, with a calibration set large enough, or a target with
  enough margin, to detect a one-point drift. Our 50-query calibration kept a too-narrow width
  for one implementation, which sampling noise alone explains. In the resampling study a
  200-query set rarely did, and a one-point margin on 50 queries cut the risk to about 6% at the
  price of a wider search (§4.2).
- Expect the misses of an appended HNSW index to fall on its newest documents. A system whose
  users mostly look for new content should measure recall on those documents, not only on
  average.
- Alternatively, target about 98% recall instead of 95%. The gap is then about 0.002–0.004 at
  every snapshot, at about 3.5 times the latency of a 95% setting (about 9 ms per query on one
  thread for these 2,560-dimensional vectors, against 2.5 ms).

**Why the effect is small.** Three things combine.

1. Misses are less common at the top of the ranking. At snapshot 3 at ef 200 (seed 11),
   approximate search misses about 6.3% of the exact top 100, but only 3.4–3.7% of the exact
   top 10.
2. The top-10 neighbours it misses are less relevant than those it keeps. Their mean
   GPT-OSS-120B grade is 0.77–1.01, against 1.54 for the kept neighbours (HNSW, snapshot 3, all
   seeds).
3. The replacement documents have a mean grade of only 0.21–0.25, but 76–82% of them are
   ungraded and therefore count as non-relevant. The gap is thus partly an artifact of judgment
   coverage (§4.6).

## 6 Limitations

- **Scope of data.** There is one collection and one embedding model, and the collection is
  append-only. Only the insertion-position effect was replicated elsewhere, on two public
  benchmarks. The equivalence result does not cover deletions or rewritten documents. LongEval
  Web, which has deletions, is the natural next test.
- **Pooling.** Most documents that approximate search substitutes into the top 10 are
  ungraded, which biases the gap towards larger losses. The carried and condensed rescoring
  bound the bias but do not remove it; 426 pairs on the panel have no grade at any snapshot.
- **Labels.** The primary labels are LLM-generated and lenient. On the 29 queries with human
  grades at snapshot 3 they agree with them at a kappa of 0.24. The human grades cover one
  snapshot, so they cannot confirm a change over time. The click labels judge 6–38% of the
  top 10 results, so they cannot support an index comparison.
- **Exploratory analyses.** The test queries were used once for the frozen analysis. Every
  analysis marked exploratory reuses them and was designed after those results were seen.
- **Insertion order.** The full-collection experiment used two graph settings, and the
  half-size draws one. The within-query ratio was added after the pooled
  full-collection result was seen. It assumes one ratio common to all queries, and we did not
  examine why queries about recent documents are harder.
- **Calibration design.** Calibration uses 50 queries and a coarse grid. This is a design choice,
  and it is what separates the two HNSW arms.
- **Repair screens.** These are development screens, some on an 8,192-vector sample. They are
  claims about equal-work search on this data, not about every possible repair.

## 7 Conclusion

At about 95% neighbour recall, approximate search lost 0.005–0.012 nDCG@10 under the organisers'
labels, and the change in that loss as the collection nearly doubled stayed within ±0.01 for two
of three calibrated arms; the third was inconclusive. Incremental maintenance was equivalent to
rebuilding. At a fixed search width, recall fell by about a point, and an index built by
appending missed its newest documents most; within a query, inserting the same documents first
reversed that. How much of the measured loss is real depends on documents that the release never
graded, and that is what the follow-up measures.

## Reproducibility

The artifact contains:

- the frozen protocol and its hash receipt;
- all job scripts and packagers;
- per-job evidence checksums;
- the analysis code, with tests on synthetic data and a recorded pipeline test on development
  queries.

LongEval-derived data are not redistributed. The scripts rebuild them from the official release.

## References

1. Lin. Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes? ACL 2025 (Industry Track), pp. 865–872, 2025. https://doi.org/10.18653/v1/2025.acl-industry.61
2. Tu et al. Approximate Nearest Neighbor Search and Lightweight Dense Vector Reranking in Multi-Stage Retrieval Architectures. ICTIR 2020, pp. 97–100, 2020. https://doi.org/10.1145/3409256.3409818
3. Stahlberg and Byrne. On NMT Search Errors and Model Errors: Cat Got Your Tongue? EMNLP 2019, 2019. https://doi.org/10.18653/v1/D19-1331
4. Keller, Breuer and Schaer. Evaluation of Temporal Change in IR Test Collections. ICTIR 2024, pp. 3–13, 2024. https://doi.org/10.1145/3664190.3672530
5. Cancellieri et al. LongEval at CLEF 2025: Longitudinal Evaluation of IR Systems on Web and Scientific Data. CLEF 2025, LNCS, pp. 363–387, 2025. https://doi.org/10.1007/978-3-032-04354-2_20
6. Bouaraba et al. Extended Overview of the CLEF 2026 LongEval Lab on Longitudinal Evaluation of Model Performance. CLEF 2026 Working Notes, CEUR vol. 4283, 2026. https://ceur-ws.org/Vol-4283/paper374.pdf
7. Miyaguchi, Afrulbasha and Pramov. DS@GT at LongEval: Evaluating Temporal Performance in Web Search Systems and Topics with Two-Stage Retrieval. arXiv:2507.08360, 2025.
8. Lawrie et al. PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval. SIGIR 2024, pp. 2574–2578, 2024. https://doi.org/10.1145/3626772.3657964
9. Yang et al. MURR: Model Updating with Regularized Replay for Searching a Document Stream. LNCS, pp. 85–100, 2025. https://doi.org/10.1007/978-3-031-88708-6_6
10. Saravanan, Salla and Amancherla. AURORA: Neuro-Symbolic Continual Indexing for Evolving RAG Systems. Findings of ACL 2026, pp. 10179–10195, 2026. https://aclanthology.org/2026.findings-acl.495/
11. Singh et al. FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search. arXiv:2105.09613, 2021.
12. Xu et al. SPFresh: Incremental In-Place Update for Billion-Scale Vector Search. SOSP 2023, pp. 545–561, 2023. https://doi.org/10.1145/3600006.3613166
13. Mohoney et al. Quake: Adaptive Indexing for Vector Search. arXiv:2506.03437, 2025.
14. Baranchuk et al. DeDrift: Robust Similarity Search under Content Drift. ICCV 2023, pp. 10992–11001, 2023. https://doi.org/10.1109/ICCV51070.2023.01012
15. Mandarapu and Kunkunuru. When to Repair a Graph ANN Index: A Matched-Budget Negative Result, and the Interpolated-Baseline Trap That Hid It. arXiv:2607.00728, 2026.
16. Kuissi et al. Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks. arXiv:2603.04532, 2026.
17. Hua et al. Dynamically Detect and Fix Hardness for Efficient Approximate Nearest Neighbor Search. Proc. ACM Manag. Data 3, pp. 1–28, 2025. https://doi.org/10.1145/3769783
18. Chen et al. RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search. PVLDB 17, pp. 2735–2749, 2024. https://doi.org/10.14778/3681954.3681959
19. Malkov and Yashunin. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs. IEEE TPAMI, 2020. https://doi.org/10.1109/TPAMI.2018.2889473
20. Elliott and Clark. The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds. ICTIR 2024, pp. 25–33, 2024. https://doi.org/10.1145/3664190.3672512
21. Zerhoudi et al. As We May Search. ICTIR 2026, pp. 425–435, 2026. https://doi.org/10.1145/3805713.3820402
22. Upadhyay et al. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing RELevance Assessor. arXiv:2406.06519, 2024.
23. Greenland and Robins. Estimation of a Common Effect Parameter from Sparse Follow-Up Data. Biometrics 41(1), pp. 55–68, 1985. https://doi.org/10.2307/2530643
