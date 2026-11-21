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
whether exact-search conclusions transfer to such deployments.

We use three cumulative snapshots of LongEval-Sci 2026 (0.87M–1.66M documents, Qwen3-Embedding-4B
vectors) and compare exact search with two HNSW implementations and an IVF index. Each index is
maintained incrementally and also rebuilt from the identical insertion history, with three
construction seeds. The evaluation uses a pre-registered, hash-frozen protocol on 381 held-out
test queries. Our exact search reproduces the lab's official dense baseline to within 0.0014
nDCG@10.

Three findings need no relevance labels:

- **History.** Incremental maintenance is exactly equivalent to rebuilding from the same history:
  the index states, rankings and scores are identical in all 27 comparisons.
- **Recall drift.** At a fixed search width, neighbour recall falls by about one point while the
  collection nearly doubles (94.5% to 93.5%). A 50-query calibration set keeps a width that is too
  narrow in about one draw in five; about 200 queries are needed to detect the fall reliably.
- **Newer documents.** In exploratory analyses, HNSW misses documents inserted at later snapshots
  about 1.6 times as often as earlier ones at the same exact rank; IVF shows about 1.15 times. A
  held-out rebuild in random order removes most of the excess. On a 300,000-document sample,
  reversing the insertion order moves the excess onto the oldest documents (ratio 0.82–0.84).
  The cause is the position in the insertion sequence, not the age of the document.

Four findings are stated under the organisers' LLM relevance labels:

- **Gap size.** At about 95% neighbour recall, the nDCG@10 lost to approximate search is below
  0.012 at every snapshot. Served through any tested index, the official baseline would have kept
  its leaderboard rank on these labels.
- **Gap over time.** From snapshot 1 to snapshot 3, the gap's change is negligible within ±0.01
  for two of three arms. It is inconclusive for the third, whose calibration never triggered a
  wider search while its gap widened from −0.006 to −0.012. Searched at the next wider setting,
  its change lies within ±0.01 as well.
- **Concentration.** At snapshot 3, 82–89% of queries lose nothing and the worst 5% carry 71–85%
  of the loss. On fixed-width HNSW, queries about new content lose about three times the average.
- **Search width is the lever.** The gap shrinks to about 0.002–0.004 at every snapshot once
  recall reaches about 98%. Four pre-registered graph-repair strategies fail to beat an ordinary
  wider search at equal distance work.

These labels bound what can be claimed. They are lenient: on the 29 queries that also have human
grades, they agree with them at a kappa of 0.24. They were pooled from the submitted systems, so
76–82% of the documents that an HNSW index substitutes into the top 10 are ungraded and count as
non-relevant. The measured gap is therefore an upper bound that judgment coverage inflates:
restricted to graded documents it is 0.001–0.006.

The practical rule is that exact-search longitudinal conclusions hold for ANN deployments to
within about 0.01 nDCG@10, provided neighbour recall is re-checked on enough queries at every
snapshot. A target of about 98% recall makes the gap negligible and stable over time.

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
- evidence that an HNSW index built by appending under-retrieves its latest arrivals, and that
  the insertion position causes it;
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
on synthetic drift. These works change the model or the representation. We hold the representation fixed and isolate the index.

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
arrivals do, and reversing the order reverses it (§4.4).

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
correlates at 0.92–0.95 within a query, which halves the standard error. Either way, about 200
queries are needed to detect a one-point fall in nine cases out of ten. Sampling noise is
therefore enough to explain why Epzero's calibration stayed at ef 200.

A safety margin is the cheaper remedy. Asking for 96% instead of 95% on the same 50 queries cuts
the risk of keeping a too-narrow width from 21% to 5.5%, and with 200 queries a
half-point margin brings it to 0.6%.

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

**Reversing the order moves the blind spot.** Shuffling shows that the order matters. It does not
show which position is the bad one. We therefore built both HNSW implementations on a fixed
sample of 300,000 snapshot-3 documents (study settings, ef 200, three seeds, exact neighbours
within the sample) in three orders: as the documents arrived, shuffled, and with the three
arrival blocks reversed. The table gives the miss ratio of snapshot-3 arrivals against snapshot-1
arrivals, as the mean over seeds with 95% intervals (Figure 4):

| Implementation | As arrived | Shuffled | Reversed | As arrived − reversed, paired |
| --- | --- | --- | --- | --- |
| FAISS HNSW | 1.41 [1.24, 1.59] | 1.08 [0.96, 1.21] | 0.84 [0.72, 0.96] | +0.57 [+0.45, +0.71] |
| Epzero HNSW | 1.48 [1.32, 1.65] | 1.10 [0.99, 1.21] | 0.82 [0.71, 0.94] | +0.65 [+0.53, +0.80] |

The same documents are missed more when they are inserted last and less when they are inserted
first. The paired difference is above zero in each of the six builds. The shuffled ratio of about
1.1 is the part that belongs to the documents themselves, and the insertion position multiplies
or divides it by about 1.3.

The graph shows why. Base-layer incoming links fall with insertion position in every order and in
both implementations, from 28–29 for the first tenth of insertions to 12–15 for the last tenth.
At most 0.02% of documents have no incoming link, so late insertions are weakly connected, not
unreachable. Recall@100 is 0.942–0.947 in every order: the order decides who bears the misses,
not how many there are.

The result does not rest on one setting. We repeated the builds at ef 100, 200, 400 and 800, and
with a denser graph (M 32, efConstruction 200). In all 16 combinations of implementation, graph
and width the as-arrived ratio lies above 1, the reversed ratio is below 1, and the paired
difference is above zero. A wider search or a denser graph removes misses but concentrates the
rest: the as-arrived ratio is 1.30–1.58 at M 16 and 1.55–2.22 at M 32, and it is higher at ef 800
than at ef 200 in every case.

This is a pilot on 18% of snapshot 3. The reversed order has not been run on the full collection.

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

**Figure 4** (`figures/fig-insertion-pilot.pdf`). Miss ratio of snapshot-3 arrivals against
snapshot-1 arrivals by insertion order, on a 300,000-document sample at ef 200. Markers are means
over three seeds with 95% query-bootstrap intervals; short ticks are the single seeds.

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

The loss on graded documents is 0.001–0.006 nDCG@10, about half of the official gap or less.
The pattern over time survives: FAISS and IVF stay flat, and only Epzero, whose search was never
widened, drifts. The carried rule adds 131 grades at snapshot 3 but only 2 at snapshot 1, so its
D is biased upwards and is shown for completeness only. Deciding how much of the gap is real
needs grades for the 426 pairs, which is the planned follow-up.

**Leniency.** For 29 test queries the release also holds human grades at snapshot 3 (1,581
pairs). The GPT-OSS-120B labels grade 87% of these pairs as relevant, the human grades 60%, and
71% of the pairs that the human grades call non-relevant are relevant under the LLM labels.
Binary agreement is 0.70 [0.62, 0.78] and the linear-weighted kappa is 0.24 [0.14, 0.36]. Exact
search scores 0.93 nDCG@10 under the LLM labels and 0.56 under the human grades on these queries,
and the per-query values are uncorrelated. The comparison is not clean: the human grades assess a
formalized topic and the LLM labels the query alone. It still shows that nDCG@10 under these
labels sits near its ceiling and separates rankings poorly.

Under lenient labels a substitute costs little when it is graded and its full gain when it is
not, so coverage drives the measured gap. The results on neighbour recall and on arrival groups
(§4.1, §4.3, §4.4) do not depend on any label set. The effectiveness results should be read as
bounds under the organisers' labels: the gap at 95% recall is at most about 0.012, and at most
about 0.006 on graded documents.

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
  enough margin, to detect a one-point drift. Our 50-query calibration kept a too-narrow width
  for one implementation, which sampling noise alone explains: about 200 queries are needed, or
  a one-point margin on the target (§4.2).
- Expect the misses of an appended HNSW index to fall on its newest documents. A system whose
  users mostly look for new content should measure recall on those documents, not only on
  average.
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
   ungraded and therefore count as non-relevant. The gap is thus partly an artifact of judgment
   coverage (§4.6).

## 6 Limitations

- **Scope of data.** There is one collection and one embedding model, and the collection is
  append-only. The equivalence result does not cover deletions or rewritten documents. LongEval
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
- **Insertion-order reversal.** The reversed order was run on a 300,000-document sample, at four
  search widths and two graph settings, not on the full collection.
- **Calibration design.** Calibration uses 50 queries and a coarse grid. This is a design choice,
  and it is what separates the two HNSW arms.
- **Repair screens.** These are development screens, some on an 8,192-vector sample. They are
  claims about equal-work search on this data, not about every possible repair.

## 7 Conclusion

The index does matter over time, but only through neighbour recall. Incremental maintenance is
equivalent to rebuilding. At a fixed search width, recall falls by about a point as the
collection nearly doubles, and an index built by appending loses its newest documents first.
Under the organisers' labels this costs at most about 0.01 nDCG@10 when recall is held near 95%,
and at most about 0.006 on graded documents. Re-checking recall at each snapshot, on enough
queries to see a one-point fall, is what keeps exact-search longitudinal conclusions valid for
deployed ANN systems.

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
