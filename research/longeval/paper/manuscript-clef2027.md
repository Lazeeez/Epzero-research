# Does the Index Matter Over Time? Approximate Search, Index History and Collection Growth in Longitudinal Dense Retrieval

**Draft for the CLEF 2027 conference track (LNCS, long paper: 12 pages plus references). Version 2026-11-25.**
Passages marked **[E4]** wait for the second-encoder (BGE) runs, which were submitted on 25 November.
In the LNCS layout (`md2lncs.py`, then XeLaTeX) the body ends on page 11 of 13, references after it: about
10.5 of the 12 pages allowed, leaving room for the [E4] results.
This version restructures and shortens [the full draft of 21 November](../docs/history/manuscript-full-draft-2026-11-23.md),
which is kept unchanged and holds every analysis in more detail. Every number comes from verified
evidence: `../HELDOUT_RESULTS.md`, `../HELDOUT_EXPLORATORY.md` and the development result files.
Results marked *exploratory* were designed after the held-out results were seen.

## Abstract

Longitudinal evaluation campaigns such as LongEval measure how retrieval effectiveness changes as a
collection grows, but their dense baselines use exact search, while deployed systems use an
approximate nearest-neighbour index that is built once and updated in place. On three snapshots of
LongEval-Sci 2026 (0.87M to 1.66M documents), we compare exact search with two HNSW implementations
and an IVF index under a pre-registered, hash-frozen protocol on 381 held-out queries. An index
updated in place is identical to one rebuilt from the same history in all 27 comparisons. At about
95% neighbour recall, approximate search loses 0.005–0.012 nDCG@10 per snapshot; the change in that
loss as the collection nearly doubled met a ±0.01 margin for two arms and was inconclusive for the
third, whose 50-query calibration missed a one-point drift in recall. In exploratory follow-ups, an
HNSW index built by appending misses later arrivals 1.25–1.33 times as often as first arrivals of
the same query; rebuilding the full collection in three insertion orders shows that
insertion position causes this, and a denser graph sharpens it. The effect appears in three HNSW
implementations and, twice as strongly, on the Big-ANN streaming benchmark. Giving each insertion
extra incoming links and refining each new batch removes it at a fraction of a rebuild's cost. The labels limit what the
effectiveness numbers can show: they agree weakly with human grades, and most documents that an
approximate index substitutes into the top 10 were never graded.

## 1 Introduction

LongEval [5, 6] evaluates retrieval systems over successive snapshots of a growing collection. Its
dense baselines score every document for every query. Deployed dense retrieval instead runs on an
approximate nearest-neighbour (ANN) index, usually an HNSW graph [19] or an IVF partition, which is
built once and then updated as documents arrive. Operators rebuild such indexes on a schedule,
largely because they cannot tell whether a long-lived index has drifted.

Three questions follow, for evaluation methodology and for practice. Does the effectiveness lost to
approximate search stay constant as the collection grows, so that conclusions drawn with exact
search transfer to deployed systems? Does an index updated in place behave differently from one
rebuilt from scratch? And when a collection grows by appending, are its newest documents, the ones a
longitudinal evaluation is about, retrieved as reliably as its oldest?

Systems work tracks index quality over time in neighbour-recall units on unjudged workloads
[11–15]; IR work tracks effectiveness over time with exact search or a fixed index [4–7, 16]. We
join the two by following Stahlberg and Byrne's separation of search errors from model errors [3]:
the same embeddings are searched exactly and approximately at every snapshot, so the signed
ANN-minus-exact difference isolates the index.

Our contributions are:

- the first measurement of the ANN-minus-exact effectiveness gap over time on a relevance-judged
  collection, under a pre-registered, hash-frozen protocol, with an exact equivalence result for
  incremental maintenance under append-only growth;
- a diagnosis that ties the change in the gap to neighbour-recall drift, and a measurement of how
  often a small calibration set misses that drift;
- evidence at full collection scale, with a controlled reversal of the insertion order, three
  implementations and the Big-ANN streaming benchmark, that an HNSW index built by appending
  under-retrieves its latest arrivals because of where they were inserted, and a repair that removes it;
- an account of what pooled LLM labels can support when systems differ only in their index;
- pre-registered negative results for four graph-repair strategies at equal search work, and a
  checksummed artifact.

## 2 Related Work

**Longitudinal evaluation.** LongEval [5, 6] provides snapshot collections with time-specific
judgments; Keller et al. [4] attribute temporal change to the components of a test collection;
DS@GT [7] and Still Fresh? [16] study the temporal robustness of retrieval pipelines. All use exact
search or a single fixed index.

**ANN effectiveness.** Lin [1] and Tu et al. [2] measure the effectiveness lost to an HNSW index on
static collections (about 0.01–0.02 nDCG@10). Zerhoudi et al. [21] compare flat, HNSW and IVF search
on subsets of increasing size; their subsets are sampled independently, so there is no time axis,
no index maintenance and no insertion order.

**Indexes under change.** FreshDiskANN [11] and SPFresh [12] maintain graph and partition indexes
under updates, Quake [13] adapts partitions to the workload, DeDrift [14] repairs IVF under content
drift and When to Repair [15] studies repair scheduling. They report neighbour recall and
throughput, not judged effectiveness. The NeurIPS'23 Big-ANN streaming track [24] ranks indexes by
recall averaged over a runbook's search steps, and CANDOR-Bench [25] extends it with drift, noise and
concurrent updates; we break recall down by when each target arrived. PLAID SHIRTTT [8], MURR [9] and AURORA [10] keep retrieval
current by changing the model or the representation; we hold the representation fixed.

**Insertion order and repair.** Elliott and Clark [20] show that HNSW recall depends on the order of
insertion; we show which documents bear the loss when a collection grows by appending. NGFix/RFix
[17] and RoarGraph [18] add query-driven edges; we screen this family against equal-work controls.

**LLM labels.** The lab's overview [6] reports that its LLM assessors grade far more documents
relevant than click-derived labels do. We quantify what this means for an index comparison (§6).

## 3 Experimental Setup

**Data.** LongEval-Sci 2026 (CORE scientific search) has three cumulative snapshots of 869,902,
1,322,720 and 1,661,900 documents, with no deletions. We use the released Qwen3-Embedding-4B
document vectors (2,560 dimensions, inner product) unchanged, after auditing that shared documents
keep bitwise-identical vectors, and encode queries with the organiser-specified model revision and
prompt. The 100 training queries are split into 50 calibration and 50 development queries; the 381
test queries were used once, after the protocol was frozen.

**Arms.** Every index is built single-threaded on Intel Xeon Gold 6226 nodes with construction seeds
11, 29 and 47.

| Arm | Construction | Search settings |
| --- | --- | --- |
| Exact | Float64 exhaustive top 100 | — |
| Epzero HNSW, FAISS HNSW | M 16, efConstruction 100 | ef 200 fixed; calibrated ef |
| FAISS IVF-Flat | nlist 1,024; training frozen at snapshot 1, or refreshed | nprobe fixed at its snapshot-1 value; calibrated |

A calibrated setting is the smallest ef in {100, 200, 400, 800}, or nprobe in {1, …, 1,024}, that
reaches 95% mean neighbour recall@100 on the 50 calibration queries at that snapshot. Each HNSW and
frozen-IVF index is maintained *incrementally* (each snapshot's new documents inserted into the
live index in lexical ID order) and also *rebuilt* from the identical insertion history.

**Measures.** nDCG@10 with linear gains and unjudged documents counted as non-relevant (agreeing
with pytrec_eval to 1e-12), and neighbour recall@100 against exact search.

**Protocol.** The protocol was frozen on 19 November 2026, before any test-query retrieval, by
hashing the protocol, configurations, analysis code and job scripts (55 files). The primary labels
are the released GPT-OSS-120B judgments, which the organisers produced for a depth-10 pool of the 24
submitted systems with a prompt adapted from UMBRELA [6, 22]. The panel is the 374 test queries with
a positive judgment at all three snapshots. For each calibrated arm A (Epzero HNSW, FAISS HNSW, IVF
with frozen training),

  D_A = [nDCG(A, s3) − nDCG(exact, s3)] − [nDCG(A, s1) − nDCG(exact, s1)],

averaged over seeds within a query, then over queries. Inference uses paired sign-randomization
tests (Holm-adjusted across the three arms) and 95% paired query-bootstrap intervals. A change is
*negligible* if its whole interval lies within ±0.01, *changed* if it is not negligible and its
Holm-adjusted p is below 0.05, and *inconclusive* otherwise. The analysis was run once.

## 4 Approximation Over Time

### 4.1 Index history has no effect

The incremental index and its identical-history rebuild are identical in all 27 held-out comparisons
(both HNSW implementations and IVF, three seeds, three snapshots): the same state fingerprints and
every ranking and score at every search setting. Under append-only growth, "rebuild or keep
updating" is not a question of accumulated drift; a rebuild changes results only if it changes the
insertion history.

### 4.2 The gap and its change

Our exact search reproduces the lab's official dense baseline to within 0.0014 nDCG@10 at every
snapshot, under both the LLM and the click-derived DCTR labels.

**Table 1.** Primary contrasts, 374 queries, GPT-OSS-120B labels.

| Arm (calibrated) | Gap s1 | Gap s2 | Gap s3 | D (s3 − s1) | 95% interval | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0057 | −0.0102 | −0.0117 | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| FAISS HNSW | −0.0048 | −0.0049 | −0.0055 | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| IVF, frozen training | −0.0065 | −0.0079 | −0.0080 | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

The gap stays below 0.012 at every snapshot, in line with the static estimates [1, 2]. For FAISS and
IVF the intervals are much narrower than the margin: they rule out a change larger than about 0.004
(two one-sided tests at ±0.004 pass), although in relative terms the gap could still have grown by
60-80%. *(Exploratory.)* The pattern holds under GPT-OSS-20B labels (D of −0.0074, −0.0017 and
−0.0014), when only grade 2 or any positive grade counts as relevant, and when every grade is
redrawn with the repeat behaviour of an LLM assessor (D moves by a standard deviation of at most
0.0007). Construction randomness matters more: Epzero's change comes mostly from one of its three
seeds (−0.011 against −0.003 and −0.004), and with seeds resampled as well as queries its interval
is [−0.0146, +0.0011]. *(Exploratory.)* Exact search's
own nDCG@10 changes by +0.0070 [−0.0000, +0.0139] over the same snapshots, so the ±0.01 margin is as
large as the trend it protects: the collection can show that the loss to approximation did not move
by more than the margin, not that a longitudinal trend is preserved. The loss is also concentrated:
82–89% of queries lose nothing at snapshot 3, and the worst 5% carry 71–85% of it.

### 4.3 Why the two HNSW arms differ: calibration

At fixed ef 200 both implementations lose neighbour recall at the same rate as the collection grows
(Epzero 94.5% to 93.5%, FAISS 94.4% to 93.6%; Figure 1b), and the worst tenth of queries recovers
only 65% of its true neighbours at snapshot 3. What differs is calibration. For two FAISS seeds the
calibration recall at ef 200 dipped to 94.4–94.6% at snapshots 2 and 3, so calibration chose ef 400,
which restored test recall to 95.3–95.7% and held the gap flat. Epzero's calibration recall stayed
at 95.1–95.6%, just above the target, so its ef never changed. The change in the gap tracks the
change in recall on the evaluated queries. *(Exploratory.)* At the same width the implementations
behave alike: at ef 400 Epzero's D is −0.0026 [−0.0059, +0.0002], inside the margin, against −0.0022
for FAISS, and at ef 200 their difference is not reliable (p = 0.20).

**Fifty queries cannot detect a one-point drift** *(exploratory)*. Per-query recall at ef 200 has a
standard deviation of 0.11–0.12, so a 50-query mean has a standard error of about 1.6 points. We
drew calibration sets from the test queries and applied the protocol's rule at snapshot 3, where
true recall at ef 200 is 93.5–93.6%:

**Table 2.** How often calibration keeps the too-narrow width (Epzero / FAISS), and how often a paired
test detects the fall in recall from snapshot 1.

| Calibration queries | 25 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- |
| Rule keeps ef 200 | 32% / 32% | 21% / 22% | 11% / 12% | 4% / 4% | 1% / 1% |
| Paired test detects the fall | 35% / 33% | 51% / 51% | 73% / 73% | 89% / 90% | 96% / 98% |

Sampling noise alone explains why Epzero's calibration stayed at ef 200. A one-point safety margin
(96% instead of 95%) on 50 queries cuts the risk from 21% to 5.5%, at a wider mean width (416 to 522).

### 4.4 Search width, cost and repair

*(Exploratory.)* Searching every held-out HNSW index at ef 100 to 3,200 shows that the gap is a
steep function of neighbour recall (Figure 2), and that above about 98% recall the three snapshots
coincide. Timed on one thread (median of three calls per query), Epzero at ef 200 reaches 94.5% and
93.5% recall@100 at snapshots 1 and 3, with gaps of −0.006 and −0.012, in 2.5–2.7 ms; ef 400 reaches
96.8% and 96.3% (−0.003, −0.005) in 4.8–5.0 ms; ef 800 98.1% and 97.8% (−0.002, −0.003) in 8.9–9.2 ms;
and ef 3,200 99.4% and 99.1%, with no measurable gap, in 31–32 ms. FAISS is within 0.005 in recall and
5–8% faster at equal width. About 98% recall costs about 3.5 times the latency of a 95% setting;
latency follows the distance work, and growth adds 2–7% at a fixed width. IVF at its calibrated nprobe
needs 129–184 ms.

**Repair does not beat a wider search** (RQ4, development). Four pre-registered strategies were
screened against controls given the same distance work: resuming an exhausted search from skipped
candidates (3 wins, 186 ties and 11 losses in 200 paired comparisons), witness edges learned from
calibration queries (+0.02 points), the published NGFix/RFix repair [17] (+0.04 to +0.18 points at
similar latency) and bridges from encoder-generated pseudo-queries (−0.04 to −0.39 points). Each
raises recall at a fixed setting; none beats an ordinary wider search at equal work. In development,
the queries with zero recall recover on the unchanged graph with a wider search: failures are
search-coverage failures, not unreachable regions.

## 5 The Latest Arrivals Are Under-Retrieved

*(Exploratory.)* Each document is assigned to the snapshot in which it first appears. For every
exact top-100 neighbour we record whether the index returned it, and compare later arrivals with
first arrivals of the same query in the same exact-rank band (1–10, 11–50, 51–100), as a
Mantel–Haenszel risk ratio stratified by query and band [23]. In the held-out runs at snapshot 3,
HNSW misses documents that arrived after snapshot 1 1.33 [1.21, 1.47] (Epzero) and 1.25 [1.14,
1.38] (FAISS) times as often; IVF, which has no insertion order, 1.08–1.10 times. A ratio pooled over
all queries is higher (1.54–1.63 for HNSW) because queries whose neighbours are recent documents are
harder in every insertion order: the third of queries with the fewest recent neighbours misses
2.6–2.7% of its first-arrival neighbours, the other two thirds 7.4–8.8%. We chose the within-query
ratio after seeing the pooled one; the pooled contrast recorded in advance is reported beside it.

**Inserting the same documents in another order.** We rebuilt both HNSW implementations on all
1,661,900 snapshot-3 documents in three orders (as the documents arrived, shuffled, and with the
three arrival blocks reversed), three seeds each. The design and its pooled contrast were recorded
before any output was seen, and the as-arrived builds reproduce the held-out runs query for query.

**Table 3.** Miss ratio of snapshot-3 arrivals against snapshot-1 arrivals at ef 200, mean over
three seeds with 95% paired query-bootstrap intervals.

| Graph | Implementation | Ratio | As arrived | Shuffled | Reversed |
| --- | --- | --- | --- | --- | --- |
| M 16 | FAISS | within query | 1.28 [1.12, 1.47] | 1.03 [0.93, 1.14] | 0.82 [0.72, 0.92] |
| M 16 | Epzero | within query | 1.42 [1.25, 1.63] | 1.00 [0.90, 1.10] | 0.82 [0.72, 0.92] |
| M 16 | FAISS / Epzero | pooled | 1.60 / 1.75 | 1.23 / 1.20 | 0.97 / 0.98 |
| M 32 | FAISS | within query | 1.66 [1.40, 2.01] | 0.99 [0.87, 1.12] | 0.67 [0.54, 0.81] |
| M 32 | Epzero | within query | 1.65 [1.38, 2.02] | 0.99 [0.87, 1.11] | 0.69 [0.54, 0.82] |

The same documents are missed more when inserted last, as often as any others at a random position,
and less when inserted first (Figure 3). The recorded pooled contrast, as arrived minus reversed, is
+0.62 [+0.46, +0.80] for FAISS and +0.77 [+0.57, +0.98] for Epzero, and is above zero for every
seed and search width from 100 to 800. Within a query the reversed interval lies below 1 at every
width. Three stratified draws of half the collection give the same within-query ratios as the full
collection to within 0.04, and a 300,000-document sample the same pattern, so the effect does not
depend on collection size.

**A denser graph sharpens it.** With M 32 and efConstruction 200, also recorded before the jobs
ran, recall@100 at ef 200 rises from 0.93 to 0.96–0.97, but the misses that remain fall more heavily
on the latest arrivals (Table 3). Tuning the graph for recall does not remove the effect of
insertion position.

**Position alone.** In a shuffled build a document's position is independent of the document and
the query, yet the miss rate still rises steadily with position, from 4.9–5.0% in the first tenth
of insertions to 7.4–7.8% in the last (Figure 4a), while base-layer incoming links fall from about
29 to about 12. Late insertions are weakly connected, not unreachable: at most 0.06% of documents
have no incoming link. On GloVe and SIFT, with arrival groups assigned at random, the same design
gives shuffled ratios of 0.92–1.03, reversed ratios whose intervals lie below 1, and the same decile
curves (Figure 4b, c): the effect belongs to building an HNSW graph by insertion.

**Where it lands matters.** The within-query ratio is not one common ratio. Across rank bands it
agrees, but across queries it does not: a late arrival is missed 2.0–2.7 times as often as first
arrivals when it lands among mostly older neighbours, and only 1.06–1.33 times when its
neighbourhood is itself mostly new (in shuffled builds all groups are near 1).

**Three implementations and a public stream.** hnswlib, the reference HNSW library, gives the same
within-query ratios on a 300,000-document LongEval sample as the other two (as arrived 1.34 [1.20,
1.51], shuffled 1.00, reversed 0.77; within +0.04 of their mean). On Big-ANN's clustered MSTuring-10M
runbook [24], 10 million vectors arrive as 32 k-means clusters, each together with its own
neighbours. Reversing that order changes the gap between the newest and the oldest thirds' miss rates
by −11.2 [−11.5, −10.9] points for Epzero and −9.1 [−9.4, −8.8] for FAISS (−4.0 for Epzero on
LongEval), while a vector's position within its own cluster adds only 0.7–1.1 points. Arriving with its
neighbours does not protect a cluster: what counts is the global insertion position.

**Effect on effectiveness.** On the 77 queries whose exact top 10 is at least half new content,
Epzero's calibrated arm, which stayed at ef 200, loses 0.026 nDCG@10 [0.006, 0.055], against 0.008
on the other 297; FAISS, whose search widened, loses 0.008 and 0.005. These queries were already
harder at snapshot 1, before any new content existed, and within strata of that difficulty the extra
loss is −0.017 [−0.045, +0.004] for Epzero and −0.003 for FAISS: most of it is query difficulty. Rebuilding in a random order
does not help: it spreads the misses across document ages without reducing them (snapshot-3 recall
0.933 against 0.936), does not reduce the fresh-content loss (Epzero −0.026 to −0.024, FAISS −0.009
to −0.017) and makes the other queries worse. A reordered rebuild is no remedy; a wider search lowers
every miss rate, and the repair below removes the gap between arrivals.

**A repair.** Late insertions lack the incoming links that earlier documents keep receiving from later
arrivals. We give each insertion eight more, from its closest construction candidates that it did not
choose, and refine each new batch once, with a search width of 100, at the batch's end. In Epzero's
production path (8-bit codes; snapshot 1 bulk-built, snapshots 2 and 3 inserted) on all 1,661,900
documents, the within-query excess miss rate of snapshot-3 over snapshot-1 arrivals at ef 200 falls from
3.36 to 0.57 points on the test queries (change −2.79 [−3.41, −2.21]) and from 1.12 to −0.37 on the
100 training queries; a full rebuild gives 0.43 and −0.80. The oldest arrivals' miss rate changes by
+0.20 [−0.02, +0.42] points, less than a rebuild's +0.40, recall@100 rises by 0.56 points, and the cost
is 9% more insertion time and 54 s of refining on 24 cores, against 182 s for a rebuild. On the Big-ANN
stream the excess falls from 5.63 to −0.65 points and recall@100 at ef 200 rises from 74.7% to 80.9%,
above a rebuild's 79.2%. The repair was chosen on a 300,000-document development sample through
screens whose designs and predictions were recorded before they ran; refining the older documents
around each batch, or two narrow refine passes, made the gap worse or no better. **[E4]** The
second-encoder reading is pending.

## 6 What the Labels Can Support

*(Exploratory.)* **Coverage.** The labels cover a depth-10 pool that includes the exact baseline, so
99.5–99.6% of exact search's top 10 is graded, but 76–82% of the documents that an HNSW index
substitutes into the top 10 are ungraded and count as non-relevant. Two rescoring rules bracket the
effect without new judgments: *carried* gives an ungraded occurrence the grade the same pair has at
another snapshot, and *condensed* removes ungraded documents from every ranking.

**Table 4.** Gap at snapshot 3 and D under three scoring rules.

| Arm (calibrated) | Gap: official | carried | condensed | D: official | carried | condensed |
| --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0117 | −0.0089 | −0.0056 | −0.0059 | −0.0034 | −0.0048 |
| FAISS HNSW | −0.0055 | −0.0033 | −0.0007 | −0.0007 | +0.0015 | +0.0004 |
| IVF, frozen | −0.0080 | −0.0046 | −0.0017 | −0.0015 | +0.0019 | +0.0003 |

With ungraded documents removed the gap halves or more, and the pattern over time is unchanged
(only Epzero, whose search never widened, drifts). The carried rule adds 131 grades at snapshot 3 but
only 2 at snapshot 1, so its D is biased upwards. **Leniency.** On the 29 test queries that also
have human grades (1,581 pairs), the LLM labels call 87% of the pairs relevant against 60%, with a
linear-weighted kappa of 0.24 [0.14, 0.36]; exact search scores 0.93 nDCG@10 under the LLM labels
and 0.56 under the human grades, uncorrelated per query.

Under lenient labels a substitute costs little when it is graded and its full gain when it is not,
so judgment coverage has a large part in the measured gap. Neither rescoring bounds the true loss.
The results on neighbour recall, index history and arrival (§4.1, §4.3, §5) use no labels at all.

## 7 Discussion

**For longitudinal evaluation.** Two of three calibrated arms met the margin, and the third was
inconclusive because its search width was never widened while recall drifted down by about a point.
The loss at each snapshot, its change over time and the recall actually reached on the test queries
are different quantities, and a fixed search setting keeps none of them stable. A campaign that
reports ANN runs should report neighbour recall per snapshot beside nDCG, and should know that an
appended graph serves its newest documents worst, which is what a longitudinal evaluation measures.

**For operators.** Rebuilding an append-only index from the same history gains nothing, and
rebuilding in a new order only moves the misses; extra incoming links for each insertion and a refine of
each new batch remove the gap between arrivals for 9% more insertion work (§5). What matters is the search width: re-check it as
the collection grows, with a calibration set large enough to detect a one-point drift or a margin on
the target (§4.3), or target about 98% recall at about 3.5 times the latency. A system whose users mostly look for new
content should measure recall on those documents, not only on average.

**Why the loss is small.** Approximate search misses about 6.3% of the exact top 100 at ef 200 but
only 3.4–3.7% of the top 10; the top-10 neighbours it misses are less relevant than those it keeps
(mean grade 0.77–1.01 against 1.54); and most replacements are ungraded (§6).

## 8 Limitations

There is one collection and, until the second encoder's runs return **[E4]**, one embedding model,
and the collection only grows, so the equivalence result does not cover deletions or rewritten
documents; only the insertion-position effect was reproduced elsewhere (GloVe, SIFT, a third
implementation and the Big-ANN stream). The effectiveness results rest on lenient LLM labels whose pool leaves most
substitutes ungraded; human grades cover 29 queries at one snapshot. The test queries were used
once for the frozen analysis, and every exploratory analysis reuses them. The within-query ratio was
chosen after the pooled result was seen and averages over query groups whose ratios differ. Three
construction seeds are too few to separate seed variation from Epzero's change. Calibration used 50 queries and a coarse
grid, a design choice that separates the two HNSW arms. The query-driven repair results come from
development screens; the arrival repair was confirmed at full scale and on the stream, by neighbour
recall, not by judged effectiveness.

## 9 Conclusion

At about 95% neighbour recall, approximate search lost 0.005–0.012 nDCG@10 under the organisers'
labels, and the change in that loss as the collection nearly doubled stayed within ±0.01 for two of
three calibrated arms. Incremental maintenance was equivalent to rebuilding. At a fixed width recall
fell by about a point, and an index built by appending missed its newest documents most; inserting
the same documents first reversed that, and a denser graph made it stronger. Extra incoming links for
each insertion and a refine of each new batch removed it, on LongEval and on the Big-ANN stream. How much of the
measured loss is real depends on documents the release never graded, which our follow-up measures.

**Figure 1** (`figures/fig-gap-recall.pdf`). (a) ANN-minus-exact nDCG@10 by snapshot at calibrated
settings, mean over 374 queries and three seeds, with 95% query-bootstrap intervals. (b) Mean
neighbour recall@100 on the test queries; the dashed line is FAISS HNSW held at ef 200.

**Figure 2** (`figures/fig-recall-gap.pdf`). The gap against neighbour recall@100 across ef 100 to
3,200, for incrementally maintained (a) Epzero and (b) FAISS HNSW; darker lines are later snapshots.

**Figure 3** (`figures/fig-insertion-order.pdf`). Miss ratio of snapshot-3 arrivals against
snapshot-1 arrivals by insertion order, on all 1,661,900 snapshot-3 documents at ef 200, M 16:
(a) pooled over queries, (b) within the same query and rank band.

**Figure 4** (`figures/fig-position-full.pdf`). Miss ratio of each insertion decile against the
first, in shuffled builds at ef 200: (a) LongEval snapshot 3, (b) GloVe, (c) SIFT.

## Reproducibility

The artifact holds the frozen protocol and its hash receipt, every job script and packager, per-job
evidence checksums, and the analysis code with its tests. LongEval-derived data are not
redistributed; the scripts rebuild them from the official release.

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
24. Simhadri et al. Results of the Big ANN: NeurIPS'23 Competition. NeurIPS 2025 Datasets and Benchmarks Track, 2025. arXiv:2409.17424.
25. Wang, Dong, Liao and Jin. CANDOR-Bench: Benchmarking In-Memory Continuous ANNS under Dynamic Open-World Streams. SIGMOD 2026. https://doi.org/10.1145/3786630
