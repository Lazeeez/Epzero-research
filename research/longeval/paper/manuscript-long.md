# Does the Index Matter Over Time? The Long Version

**A working document, without a page limit. Version 2026-11-26 (second revision).** It explains the whole study: the
background a reader needs, every experiment and its design, the results, the analyses that failed,
and what remains open. The submission is the 12-page `manuscript-clef2027.md`; every number here comes
from the same verified sources (`../HELDOUT_RESULTS.md`, `../HELDOUT_EXPLORATORY.md`,
`../../../docs/arrival-bias-research.md`, the runbooks' result sections under `scripts/pace/`, and the
aggregate files in `research/longeval/`). Where a result was designed after the held-out results were
seen it is marked *exploratory*; where its design and prediction were recorded before the run, that is
said.

## Contents

1. What this study is about, in plain terms
2. Background: dense retrieval, approximate search, HNSW and IVF
3. The collection, the queries and the labels
4. The frozen study: design and protocol
5. Result 1: index history has no effect
6. Result 2: the effectiveness gap and its change over time
7. Why the two HNSW arms differ: calibration
8. Search width, cost, and repairs that did not help
9. Result 3: the latest arrivals are under-retrieved
10. Why it happens: the mechanism
11. How general it is: other datasets, a third implementation, a public stream, a second encoder
12. What it costs in ranked relevance
13. A repair, and how it was found
14. What the labels can support
15. The extension protocol and its amendments
16. Discussion: for evaluation campaigns and for operators
17. Limitations
18. What remains open
19. Reproducibility
20. Glossary
21. References

---

## 1. What this study is about, in plain terms

Search engines increasingly retrieve documents by meaning rather than by words. Each document and each
query is turned into a vector (an *embedding*), and the documents whose vectors lie closest to the
query's are returned. Finding the closest vectors exactly means comparing the query with every document,
which is too slow for large collections, so deployed systems use an *approximate nearest-neighbour*
(ANN) index that finds most of the closest documents in a small fraction of the time.

Evaluation campaigns such as CLEF LongEval study how well retrieval systems keep working as a
collection changes over time: they release the same collection at successive moments (*snapshots*),
each larger than the last, with relevance judgments at each. Their dense baselines use *exact* search.
Deployed systems do not: they build an ANN index once and keep adding documents to it.

So three questions matter, for evaluation and for practice:

1. **Does the loss caused by approximate search stay constant as the collection grows?** If it does,
   conclusions drawn with exact search carry over to deployed systems.
2. **Does an index that is updated in place drift away from one rebuilt from scratch?** Operators
   rebuild indexes on a schedule largely because they cannot tell.
3. **When a collection grows by appending, are its newest documents retrieved as reliably as its
   oldest?** A longitudinal evaluation is about exactly those newest documents.

The short answers, which the rest of the document supports in detail:

1. **Yes, within a tight margin.** At about 95% neighbour recall, approximate search loses 0.005–0.012
   nDCG@10 per snapshot under the official labels, and that loss changed by less than ±0.01 for two of
   three indexes as the collection nearly doubled; the third was inconclusive because its search width
   was never widened while recall drifted down by about a point. Graded completely by a validated local
   judge, the loss is 30–60% smaller still, with the same trend.
2. **No.** An index updated in place is *bit-for-bit identical* to one rebuilt from the same insertion
   history, in all 27 comparisons.
3. **No, and this is the main new finding.** In an HNSW graph built by appending, documents inserted
   later are missed 1.25–1.42 times as often as earlier ones by the same query. Rebuilding the same
   documents in other orders shows that the *insertion position* causes it: inserted first, the same
   documents are missed less. It appears in three HNSW implementations, with a second embedding model,
   and twice as strongly on the public Big-ANN streaming benchmark. A cheap repair removes it.

---

## 2. Background: dense retrieval, approximate search, HNSW and IVF

**Dense retrieval.** An encoder model maps text to a vector of a few hundred to a few thousand
numbers. Relevance is approximated by similarity: the inner product (or cosine) of the query's and a
document's vectors. Retrieval returns the k documents with the highest similarity.

**Exact and approximate search.** Exact search scores every document. An ANN index scores only a small
part of the collection and returns the best of those. Its quality is measured by *neighbour recall@k*:
the share of the exact top-k documents it returns. Neighbour recall says nothing about relevance; it
only says how close the approximation is to exact search. The *effectiveness gap* is the difference in
a relevance measure (here nDCG@10) between the approximate and the exact ranking of the same queries
over the same vectors. Because the vectors are the same, the gap isolates the index: it is a *search
error*, separate from the model's own errors (the distinction of Stahlberg and Byrne [3]).

**HNSW** (Hierarchical Navigable Small World [19]) is the most widely used ANN index. It is a graph: each
document is a node, linked to a few dozen of its near neighbours. A search starts at an entry node and
repeatedly moves to the neighbours closest to the query, keeping a candidate list (the *beam*) of size
*ef*. A larger ef explores more of the graph: higher recall, more time. The graph is built by
*inserting* documents one at a time: each new document searches the existing graph for its nearest
neighbours, links to up to M of them (2M on the base layer), and asks each of those to link back. A
neighbour whose list is full keeps only its most useful links. Two parameters shape the graph: M (links
per node) and efConstruction (the beam used while inserting).

**IVF** (inverted file) partitions the collection with k-means into *nlist* clusters. A search scores
only the documents in the *nprobe* clusters nearest the query. New documents are assigned to their
nearest existing cluster; the cluster centres can be kept from the first training (*frozen*) or
re-trained (*refreshed*). IVF has no insertion order in the sense HNSW has: with fixed centres and full
scans of the chosen clusters, the order in which documents were added cannot change the results.

**Why insertion order could matter for HNSW.** When a document is inserted, it links to its neighbours
and they try to link back. Every *later* document that lands nearby may also link to it. An early
document has had the whole rest of the build to collect such incoming links; the last document
inserted has had none. Incoming links are how a search *reaches* a node, so late documents could be
harder to find. Whether this matters in practice, and for whom, is what §9–§13 measure.

---

## 3. The collection, the queries and the labels

**LongEval-Sci 2026** (scientific search over the CORE collection) has three cumulative snapshots of
869,902, 1,322,720 and 1,661,900 documents. Documents are only added, never deleted or changed. Each
document is assigned to its *arrival snapshot*, the first snapshot that contains it.

**Vectors.** We use the organisers' released Qwen3-Embedding-4B document vectors (2,560 dimensions,
inner product) unchanged, after auditing that every document shared between snapshots keeps
bitwise-identical vectors. Queries are encoded with the organiser-specified model revision and prompt.
Our exact search reproduces the lab's official dense baseline to within 0.0014 nDCG@10 at every
snapshot, under both the LLM labels and the click-derived DCTR labels.

**Queries.** 100 training queries, split into 50 calibration and 50 development queries, and 381 test
queries. The test queries were used once, for the frozen analysis, after the protocol was frozen.

**Labels.** The primary relevance labels are the organisers' GPT-OSS-120B judgments of a depth-10 pool
of the 24 submitted systems, made with a prompt adapted from UMBRELA [6, 22], on a 0–3 scale. The
*panel* is the 374 test queries with a positive judgment at all three snapshots. Two other label sets
exist: GPT-OSS-20B judgments and click-derived DCTR labels, and human grades for 29 test queries at
one snapshot (1,581 pairs). §14 shows what these labels can and cannot support.

---

## 4. The frozen study: design and protocol

**Arms.** Every index is built single-threaded on Intel Xeon Gold 6226 nodes, with construction seeds
11, 29 and 47.

| Arm | Construction | Search settings |
| --- | --- | --- |
| Exact | Float64 exhaustive top 100 | — |
| Epzero HNSW, FAISS HNSW | M 16, efConstruction 100 | ef 200 fixed; calibrated ef |
| FAISS IVF-Flat | nlist 1,024; training frozen at snapshot 1, or refreshed | nprobe fixed at its snapshot-1 value; calibrated |

Epzero is an open-source C++ HNSW library; FAISS is Meta's. Using two implementations separates
properties of HNSW from properties of one codebase.

**Calibration.** A *calibrated* setting is the smallest ef in {100, 200, 400, 800}, or nprobe in
{1, …, 1,024}, that reaches 95% mean neighbour recall@100 on the 50 calibration queries at that
snapshot. This mimics an operator who re-checks the search width as the collection grows.

**Maintenance.** Each HNSW and frozen-IVF index is maintained *incrementally*: each snapshot's new
documents are inserted into the live index, in lexical ID order. Each is also *rebuilt* from the
identical insertion history.

**Measures.** nDCG@10 with linear gains and unjudged documents counted as non-relevant (agreeing with
pytrec_eval to 1e-12), and neighbour recall@100 against exact search.

**The primary contrast.** For each calibrated arm A (Epzero HNSW, FAISS HNSW, IVF with frozen
training),

  D_A = [nDCG(A, s3) − nDCG(exact, s3)] − [nDCG(A, s1) − nDCG(exact, s1)],

averaged over seeds within a query, then over queries: the change in the effectiveness gap from the
first snapshot to the last.

**Decision rule.** Inference uses paired sign-randomization tests (Holm-adjusted across the three arms)
and 95% paired query-bootstrap intervals (10,000 draws). A change is *negligible* if its whole interval
lies within ±0.01, *changed* if it is not negligible and its Holm-adjusted p is below 0.05, and
*inconclusive* otherwise.

**Pre-registration.** The protocol was frozen on 19 November 2026, before any test-query retrieval, by
hashing the protocol, the configurations, the analysis code and the job scripts: 55 files, listed with
their hashes in `FINAL_PROTOCOL_V1_FREEZE.json`. The analysis was run once. Everything after it is either
exploratory (designed after seeing the results) or pre-registered under the extension protocol (§15).

---

## 5. Result 1: index history has no effect

The incremental index and its identical-history rebuild are identical in all 27 held-out comparisons
(both HNSW implementations and IVF, three seeds, three snapshots): the same state fingerprints and
every ranking and score at every search setting. The second encoder repeats this: again 27 of 27.

What this means: under append-only growth, "rebuild or keep updating" is not a question of accumulated
drift. A rebuild changes results only if it changes the insertion history: for example, by inserting
the same documents in a different order, which is the subject of §9.

---

## 6. Result 2: the effectiveness gap and its change over time

**Table 1.** Primary contrasts, 374 queries, GPT-OSS-120B labels.

| Arm (calibrated) | Gap s1 | Gap s2 | Gap s3 | D (s3 − s1) | 95% interval | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0057 | −0.0102 | −0.0117 | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| FAISS HNSW | −0.0048 | −0.0049 | −0.0055 | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| IVF, frozen training | −0.0065 | −0.0079 | −0.0080 | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

The gap stays below 0.012 at every snapshot, in line with the static estimates of Lin [1] and Tu et al.
[2] (about 0.01–0.02). For FAISS and IVF the intervals are much narrower than the margin: they rule out a
change larger than about 0.004 (two one-sided tests at ±0.004 pass), although in relative terms the gap
could still have grown by 60–80%.

**Robustness** *(exploratory)*. The pattern holds under the GPT-OSS-20B labels (D of −0.0074, −0.0017 and
−0.0014), when only grade 2 or any positive grade counts as relevant, and when every grade is redrawn
with the repeat behaviour of an LLM assessor (D moves by a standard deviation of at most 0.0007).
Construction randomness matters more: Epzero's change comes mostly from one of its three seeds (−0.011
against −0.003 and −0.004), and with seeds resampled as well as queries its interval is [−0.0146,
+0.0011].

**Ten construction seeds** *(exploratory; design and prediction recorded before submission)*. Seven
further seeds (101, 211, 307, 401, 503, 601, 701) through the frozen code, with queries and seeds both
resampled (`HELDOUT_EXPLORATORY.md` section 18):

| Arm | D, three frozen seeds | D, ten seeds | 95% interval (queries and seeds) | SD across seeds |
| --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0059 | −0.0034 | [−0.0092, +0.0015] | 0.0033 |
| FAISS HNSW | −0.0007 | −0.0021 | [−0.0060, +0.0015] | 0.0027 |
| IVF, frozen training | −0.0015 | −0.0009 | [−0.0035, +0.0014] | 0.0023 |

Over ten seeds all three intervals lie within ±0.01, Epzero's included: the frozen three-seed value was
pulled down by seed 29 (−0.0110; the other nine lie between −0.0047 and +0.0007). The prediction that
Epzero's ten-seed D is smaller in magnitude held; the prediction that FAISS stays within ±0.004 held for
its point estimate but not its interval, so with seed variation included the defensible bound for FAISS
is about 0.006 rather than 0.004 (IVF: 0.0035). FAISS's calibration moved from ef 200 to 400 on six of ten
seeds, with D −0.0005 on those six against −0.0045 on the four that stayed at 200; Epzero stayed at ef 200
on nine seeds; IVF never needed more probes. The frozen, confirmatory result for Epzero remains
"inconclusive"; the ten-seed result is exploratory. The same seven seeds on the BGE vectors are prepared
as a follow-up (amendment 7, §15).

**The margin is as large as the trend** *(exploratory)*. Exact search's own nDCG@10 changes by +0.0070
[−0.0000, +0.0139] over the same snapshots. So the collection can show that the loss to approximation did
not move by more than the margin, not that a longitudinal trend is preserved.

**The loss is concentrated.** 82–89% of queries lose nothing at snapshot 3, and the worst 5% carry
71–85% of the loss.

**With a second encoder** (pre-registered, §15). With BGE-base-en-v1.5 vectors of the same documents
(768 dimensions, a different model family) and the same design, all three arms are negligible: D is
+0.0016 [−0.0040, +0.0084] for Epzero, −0.0027 [−0.0093, +0.0028] for FAISS and +0.0005 [−0.0012, +0.0021]
for IVF. Paired with Qwen on the same 374 queries, D differs by −0.0020 [−0.0097, +0.0048] for FAISS and
+0.0020 [−0.0008, +0.0050] for IVF, within the margin; for Epzero by +0.0075 [−0.0017, +0.0205]. Epzero's
small loss under Qwen did not reproduce, and whether the two encoders agree for it is not settled. Exact
search with BGE *loses* 0.037 nDCG@10 [0.026, 0.048] over the same snapshots (Qwen: +0.007): the
approximation's change stays within the margin under either trend.

---

## 7. Why the two HNSW arms differ: calibration

At a fixed ef of 200, both implementations lose neighbour recall at the same rate as the collection
grows (Epzero 94.5% to 93.5%, FAISS 94.4% to 93.6%), and the worst tenth of queries recovers only 65% of
its true neighbours at snapshot 3. What differs is calibration. For two FAISS seeds the calibration
recall at ef 200 dipped to 94.4–94.6% at snapshots 2 and 3, so calibration chose ef 400, which restored
test recall to 95.3–95.7% and held the gap flat. Epzero's calibration recall stayed at 95.1–95.6%, just
above the target, so its ef never changed. The change in the gap tracks the change in recall on the
evaluated queries.

*(Exploratory.)* At the same width the implementations behave alike: at ef 400 Epzero's D is −0.0026
[−0.0059, +0.0002], inside the margin, against −0.0022 for FAISS; at ef 200 their difference is not
reliable (p = 0.20).

**Fifty queries cannot detect a one-point drift** *(exploratory)*. Per-query recall at ef 200 has a
standard deviation of 0.11–0.12, so a 50-query mean has a standard error of about 1.6 points. We drew
calibration sets from the test queries and applied the protocol's rule at snapshot 3, where true recall
at ef 200 is 93.5–93.6%:

**Table 2.** How often calibration keeps the too-narrow width (Epzero / FAISS), and how often a paired
test detects the fall in recall from snapshot 1.

| Calibration queries | 25 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- |
| Rule keeps ef 200 | 32% / 32% | 21% / 22% | 11% / 12% | 4% / 4% | 1% / 1% |
| Paired test detects the fall | 35% / 33% | 51% / 51% | 73% / 73% | 89% / 90% | 96% / 98% |

Sampling noise alone explains why Epzero's calibration stayed at ef 200. A one-point safety margin (96%
instead of 95%) on 50 queries cuts that risk from 21% to 5.5%, at a wider mean width (416 to 522).

**Calibration becomes more optimistic as the index ages** *(exploratory, ten seeds)*. At the calibrated
width the 50 calibration queries reach 0.012–0.017 higher recall than the 381 test queries for Epzero
(0.009–0.012 for FAISS), and the difference grows from snapshot 1 to 3: a measured case of search-width
calibration depending on the index's history.

**The calibration queries themselves are the problem, not only their number** *(analysis R1.2, design
recorded first)*. Repeating the study above by resampling the 50 *calibration* queries (the seven further
seeds recorded each one's recall at every width) shows how often the protocol's rule keeps ef 200 at
snapshot 3:

| Resampled calibration queries | 25 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- |
| Rule keeps ef 200 (Epzero / FAISS) | 58% / 52% | 58% / 49% | 60% / 47% | 63% / 45% | 66% / 42% |

Against 21–22% when the calibration set is drawn from the test queries, as predicted. Every seed's mean
calibration recall at ef 200 lies between 0.944 and 0.956, within about half a point of the 0.95 target, so
the rule's choice is close to a coin flip; and because these 50 queries are easier than the test queries,
more of them makes it worse for Epzero (66% at 400), not better. A larger calibration set helps only if it
is drawn like the queries it stands for.

**A margin, as an operator would run it** *(analysis R1.1)*. Applying the rule with a 0.96 target to the
frozen runs (the test rankings at the chosen widths come from the held-out sweep, which reproduced the
frozen rankings; a 0.95 target reproduces the frozen D exactly): both HNSW arms choose ef 400 at every
snapshot, for every seed, which was knowable from the calibration recalls beforehand.

| Arm, target 0.96 | Gap s1 / s2 / s3 | D | 95% interval | Test recall@100, s1 / s3 |
| --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0027 / −0.0046 / −0.0053 | −0.0026 | [−0.0059, +0.0002] | 96.8% / 96.3% |
| FAISS HNSW | −0.0028 / −0.0041 / −0.0050 | −0.0022 | [−0.0052, +0.0005] | 96.8% / 96.3% |

The loss is about halved at every snapshot and D is within ±0.01 for both, at about twice the latency of
ef 200 (§8). A 0.97 target chooses the same widths. FAISS's protocol arm had the smaller D (−0.0007) only
because its calibration widened part-way through, which masked the gap's growth; at a constant width the
two implementations change alike.

---

## 8. Search width, cost, and repairs that did not help

**The gap is a steep function of recall** *(exploratory)*. Searching every held-out HNSW index at ef 100
to 3,200 shows that above about 98% recall the three snapshots coincide. Timed on one thread (median of
three calls per query):

| Epzero width | Recall@100, s1 / s3 | Gap, s1 / s3 | Time per query |
| --- | --- | --- | --- |
| ef 200 | 94.5% / 93.5% | −0.006 / −0.012 | 2.5–2.7 ms |
| ef 400 | 96.8% / 96.3% | −0.003 / −0.005 | 4.8–5.0 ms |
| ef 800 | 98.1% / 97.8% | −0.002 / −0.003 | 8.9–9.2 ms |
| ef 3,200 | 99.4% / 99.1% | no measurable gap | 31–32 ms |

FAISS is within 0.005 in recall and 5–8% faster at equal width. About 98% recall costs about 3.5 times
the latency of a 95% setting; latency follows the distance work, and growth adds 2–7% at a fixed width.
IVF at its calibrated nprobe needs 129–184 ms.

**Repair does not beat a wider search** (pre-registered development screens, RQ4). Four strategies were
screened against controls given the same distance work:

| Strategy | Result at equal work |
| --- | --- |
| Resuming an exhausted search from skipped candidates | 3 wins, 186 ties, 11 losses in 200 paired comparisons |
| Witness edges learned from calibration queries | +0.02 recall points |
| NGFix/RFix, the published repair [17] | +0.04 to +0.18 points, at similar latency |
| Bridges from encoder-generated pseudo-queries | −0.04 to −0.39 points |

Each raises recall at a fixed setting; none beats an ordinary wider search at equal work. In
development, the queries with zero recall recover on the unchanged graph with a wider search: these are
search-coverage failures, not unreachable regions.

---

## 9. Result 3: the latest arrivals are under-retrieved

**The measurement** *(exploratory; chosen after the pooled result was seen)*. For every exact top-100
neighbour of a query we record whether the index returned it, and compare documents that arrived later
with documents that arrived in snapshot 1, *for the same query and in the same exact-rank band* (1–10,
11–50, 51–100), as a Mantel–Haenszel risk ratio stratified by query and band [23]. A ratio of 1.3 means
later arrivals are missed 1.3 times as often as first arrivals by the same query. Comparing within a
query matters: queries whose neighbours are mostly recent documents are harder in every insertion
order, so a ratio pooled over all queries confounds arrival with query difficulty. The pooled ratio
(recorded in advance) is reported beside it.

**In the held-out runs at snapshot 3**, HNSW misses documents that arrived after snapshot 1 1.33 [1.21,
1.47] times as often (Epzero) and 1.25 [1.14, 1.38] times (FAISS); IVF 1.08–1.10. The pooled ratio is
higher (1.54–1.63 for HNSW) because of query difficulty: the third of queries with the fewest recent
neighbours misses 2.6–2.7% of its first-arrival neighbours, the other two thirds 7.4–8.8%.

**Inserting the same documents in another order** (design and pooled contrast recorded before any output
was seen). We rebuilt both HNSW implementations on all 1,661,900 snapshot-3 documents in three orders:
as the documents arrived, shuffled, and with the three arrival blocks reversed, three seeds each. The
as-arrived builds reproduce the held-out runs query for query.

**Table 3.** Miss ratio of snapshot-3 arrivals against snapshot-1 arrivals at ef 200, mean over three
seeds with 95% paired query-bootstrap intervals.

| Graph | Implementation | Ratio | As arrived | Shuffled | Reversed |
| --- | --- | --- | --- | --- | --- |
| M 16 | FAISS | within query | 1.28 [1.12, 1.47] | 1.03 [0.93, 1.14] | 0.82 [0.72, 0.92] |
| M 16 | Epzero | within query | 1.42 [1.25, 1.63] | 1.00 [0.90, 1.10] | 0.82 [0.72, 0.92] |
| M 16 | FAISS / Epzero | pooled | 1.60 / 1.75 | 1.23 / 1.20 | 0.97 / 0.98 |
| M 32 | FAISS | within query | 1.66 [1.40, 2.01] | 0.99 [0.87, 1.12] | 0.67 [0.54, 0.81] |
| M 32 | Epzero | within query | 1.65 [1.38, 2.02] | 0.99 [0.87, 1.11] | 0.69 [0.54, 0.82] |

The same documents are missed more when inserted last, as often as any others at a random position, and
less when inserted first. The recorded pooled contrast, as arrived minus reversed, is +0.62 [+0.46,
+0.80] for FAISS and +0.77 [+0.57, +0.98] for Epzero, above zero for every seed and search width from 100
to 800. Within a query the reversed interval lies below 1 at every width. Three stratified draws of half
the collection give the same within-query ratios as the full collection to within 0.04, and a
300,000-document sample shows the same pattern: the effect does not depend on collection size.

**A denser graph sharpens it** (recorded before the jobs ran). With M 32 and efConstruction 200,
recall@100 at ef 200 rises from 0.93 to 0.96–0.97, but the misses that remain fall more heavily on the
latest arrivals (Table 3). Tuning the graph for recall does not remove the effect of insertion position.

**The order redistributes misses; it barely moves the mean.** Raw miss rates on the full collection (M 16,
ef 200, three seeds):

| Engine | Order | Recall@100 | Miss, snapshot 1 | Miss, snapshot 2 | Miss, snapshot 3 |
| --- | --- | --- | --- | --- | --- |
| Epzero | as arrived | 0.9352 | 5.17% | 8.36% | 9.41% |
| Epzero | shuffled | 0.9338 | 6.09% | 7.53% | 7.62% |
| Epzero | reversed blocks | 0.9308 | 6.72% | 7.52% | 6.92% |
| FAISS | as arrived | 0.9359 | 5.26% | 8.21% | 8.77% |
| FAISS | shuffled | 0.9353 | 5.85% | 7.55% | 7.58% |
| FAISS | reversed blocks | 0.9317 | 6.69% | 7.25% | 6.83% |

Overall recall differs by at most 0.44 points between orders, while the newest cohort's miss rate moves
by 2.5 points and the oldest's by 1.6 in the opposite direction. A metric that averages over the
collection, as recall usually is, cannot see the effect.

**Position alone.** In a shuffled build a document's position is independent of its content, yet the
miss rate still rises steadily with position, from 4.9–5.0% in the first tenth of insertions to 7.4–7.8%
in the last. A document inserted halfway through, with half the collection arriving after it, is still
missed more often (6.6%) than one from the first tenth (5.0%): the disadvantage fades only slowly as a
document ages. So an exact "fresh buffer" for the newest documents would rescue little: protecting the
newest tenth by exact scan would rescue only 11.6–12.1% of all misses.

**Where a document lands matters.** The within-query ratio is not one common ratio. Across rank bands it
agrees, but across queries it does not: a late arrival is missed 2.0–2.7 times as often as first
arrivals when it lands among mostly older neighbours, and only 1.06–1.33 times when its neighbourhood is
itself mostly new (in shuffled builds all groups are near 1).

---

## 10. Why it happens: the mechanism

**Incoming links fall with insertion position.** Mean base-layer in-degree (incoming links) by tenth of
insertion order (Epzero, M 16, efConstruction 100):

| Build | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LongEval 300K sample, as arrived | 28.6 | 25.4 | 23.3 | 20.6 | 17.2 | 17.1 | 16.7 | 14.7 | 14.2 | 13.3 |
| LongEval 300K sample, shuffled | 29.4 | 25.3 | 22.6 | 20.4 | 18.7 | 17.1 | 15.7 | 14.6 | 13.4 | 12.4 |
| LongEval full 1.66M, as arrived | 28.4 | 24.6 | 22.5 | 19.5 | 16.8 | 16.3 | 16.1 | 13.9 | 13.5 | 12.7 |
| LongEval full, M 32, as arrived | 47.8 | 40.9 | 36.8 | 31.5 | 26.9 | 25.8 | 25.0 | 21.6 | 20.5 | 19.2 |
| SIFT 300K, as arrived | 28.7 | 25.1 | 23.3 | 20.4 | 19.0 | 17.5 | 16.1 | 14.8 | 13.9 | 12.5 |

The gradient is the same whatever the content and whatever the order, so it is caused by position.

**It is passive accumulation, not rejection.** Two explanations were possible. In one, a new node's
requests for reverse links are *rejected* by full, established neighbours. In the other, old nodes
simply *accumulate* forward links from every later arrival nearby. Screen D2 measured it directly: right
after insertion a node already has on average 12.5 incoming links from the 12.6 neighbours it chose
(only 1.4% fewer), yet at the end of the build in-degree runs from 28.3 (first tenth) to 13.7 (last
tenth). Rejection is real but minor; the gradient is accumulation. Consistent with this, guaranteeing a
floor of incoming links at insertion (D2) had no effect, while giving late nodes extra incoming links
from nodes they had not chosen (D1, D3) removed most of their excess.

**A model that predicts it** (screen M1; design and predictions recorded first). Insert n nodes in a
random order; node j chooses k_j out-links among the j − 1 nodes present. If the choice ignores how long a
node has been there, node i is chosen by j with probability k_j / (j − 1), and the number of later arrivals
that choose it is E_i = Σ_{j>i} k_j / (j − 1) ≈ k̄ ln(n / i): *exposure*. Every link j → i with j later than i
comes from j's own choice (j's list otherwise gains only links from nodes newer than itself), so a node's
final in-degree splits exactly into links kept from later nodes (S) and from earlier ones (R). Reading
each node's list right after its own insertion, in eight 300,000-vector builds of LongEval and SIFT:

| Shuffled build, by insertion tenth | 1st | 2nd | 3rd | 5th | 8th | 10th |
| --- | --- | --- | --- | --- | --- | --- |
| LongEval: chosen by later arrivals, observed | 40.7 | 23.7 | 17.0 | 9.7 | 3.5 | 0.62 |
| LongEval: exposure prediction | 40.3 | 23.6 | 17.1 | 9.8 | 3.5 | 0.63 |
| LongEval: links from later nodes kept (S) | 26.7 | 20.2 | 15.8 | 9.6 | 3.5 | 0.62 |
| LongEval: links from earlier nodes (R) | 2.7 | 5.2 | 6.8 | 9.1 | 11.1 | 11.8 |
| SIFT: chosen, observed / predicted | 38.3 / 38.0 | 23.0 / 23.0 | 16.9 / 16.9 | 9.7 / 9.8 | 3.5 / 3.5 | 0.63 / 0.64 |

Exposure predicts the choices to within about 1% in every tenth, on both datasets: there is no
rich-get-richer term in construction, only time. Pruning removes a third of the oldest tenth's incoming
links and almost none of the rest; the kept links fall linearly in ln(n / i) (R² 0.95–0.96 in shuffled
builds). Against the author's prediction, R is not flat: a late node keeps nearly all of its ~12 reverse
links, an early one only a few, because the neighbours it chose later fill up with newer arrivals. So

  in(i) ≈ s(i) · k̄ · ln(n / i) + r(i):

about k̄ (12) for the newest nodes, about 2.4 k̄ (29) for the oldest, on any data and in any order. This is
also why the repair works: extra incoming links at insertion give a node now what exposure would give it
only over the rest of the build.

**The disadvantage fades, slowly** (screen M1, checkpoints after every tenth of each build). Against a
second prediction, the in-degree gap between the first and fifth twentieths of a build is not constant:
it shrinks by 26–37% from the half-way checkpoint to the end, as pruning trims the oldest nodes' incoming
links. In search, the newest tenth inserted is missed 1.3–1.9 times as often as the first tenth (LongEval)
and 2.3–4.5 times (SIFT) at every checkpoint; a fixed group, the fifth tenth of the build, goes from 1.53–1.56
at the half-way point to 1.17–1.47 at the end on LongEval. A document's disadvantage lasts most of the
collection's growth after it.

**Late nodes are weakly connected, not unreachable.** At most 0.06% of documents have no incoming link.
At ef 200 on the 300,000-document sample, no missed target was ever *scored* by the search: the search
simply never expanded any node that links to it, and every scored exact target was returned. The misses
are coverage failures of the traversal.

---

## 11. How general it is

**Public datasets with arrival assigned at random.** On GloVe and SIFT, with documents assigned to three
"arrival" groups at random and inserted in the same three orders, the shuffled ratios are 0.92–1.03,
the reversed ratios' intervals lie below 1, and the insertion-decile curves match LongEval's: the
effect belongs to building an HNSW graph by insertion, not to LongEval's content.

**A third implementation** (screen X1; design and predictions recorded before the run). hnswlib, the
reference HNSW library, gives the same within-query ratios on a 300,000-document LongEval sample as the
other two: as arrived 1.34 [1.20, 1.51], shuffled 1.00, reversed 0.77, within +0.04 of their mean. All
three predictions held.

**A public streaming workload** (design and predictions recorded first). Big-ANN's clustered
MSTuring-10M runbook [24] streams 10 million vectors as 32 k-means clusters, each arriving with its own
neighbours. This tests a competing explanation: perhaps only documents that arrive *among* older
neighbours are hurt (a "timing minority"), and a cluster that arrives together is protected. It is
not. Reversing the order changes the gap between the newest and the oldest thirds' miss rates by −11.22
[−11.52, −10.91] points for Epzero and −9.08 [−9.38, −8.79] for FAISS, more than twice LongEval's (−4.0 for
Epzero); shuffled builds show the global-position effect (+7.7 and +6.0 points, last tenth against
first), and a vector's position within its own cluster adds only 0.7–1.1 points. The
timing-minority prediction (|change| < 2.0) failed: what counts is the global insertion position.

**A second encoder** (pre-registered, §15). With BGE vectors, under predictions recorded on 23 November
before any BGE vector existed, the within-query ratio is 1.31 [1.21, 1.42] as arrived, 0.99 shuffled and
0.81 reversed for Epzero, and 1.24 [1.14, 1.35], 0.98 and 0.84 for FAISS. As arrived minus reversed, the
newest group's excess miss rate changes by +4.8 [+3.9, +5.7] and +4.0 [+3.1, +4.8] points (+4.3 and +3.4
with Qwen), while recall@100 differs by less than 1 point between orders (0.6 and 0.8 at ef 200; on the
confirmation queries 1.18 points for Epzero, a reading reported as it is). On the 100 training queries
the change is +4.1 [+2.0, +6.5] and +3.1 [+1.6, +4.9]. In the BGE held-out arms at snapshot 3 later
arrivals are missed 1.25 [1.16, 1.34] times as often for Epzero and 1.19 [1.12, 1.28] for FAISS (Qwen 1.33
and 1.25), and frozen IVF shows no excess (−0.02 [−0.32, +0.30] points; Qwen +0.44 [+0.04, +0.85]).

**IVF is the contrast.** IVF with centres frozen at snapshot 1 shows a small excess under Qwen (within-
query ratio 1.10 [1.01, 1.20], +0.44 points) and none under BGE. With a fixed router and complete list
scans, IVF's results cannot depend on insertion order, so its small Qwen excess must come from where the
new documents' content lands relative to the router.

---

## 12. What it costs in ranked relevance

The arrival effect is measured on neighbours. Does it cost relevance? Three analyses answer this, with
increasing completeness of the labels.

**Fresh-content queries** *(exploratory)*. On the 77 queries whose exact top 10 is at least half new
content, Epzero's calibrated arm, which stayed at ef 200, loses 0.026 nDCG@10 [0.006, 0.055], against
0.008 on the other 297; FAISS, whose search widened, loses 0.008 and 0.005. But these queries were
already harder at snapshot 1, before any new content existed, and within strata of that difficulty the
extra loss is −0.017 [−0.045, +0.004] for Epzero and −0.003 for FAISS: most of it is query difficulty.

**A shuffled rebuild is no remedy.** It spreads the misses across document ages without reducing them
(snapshot-3 recall 0.933 against 0.936), does not reduce the fresh-content loss (Epzero −0.026 to
−0.024, FAISS −0.009 to −0.017) and makes the other queries worse.

**With the pool graded in full** (Analysis C of the order rerun, designed before the rerun ran and
computed once the judging pool was graded, §14; `order-relevance-c.json`). At snapshot 3, ef 200, top 10,
374 queries, three seeds:

| | Epzero, as arrived | Epzero, reversed | FAISS, as arrived | FAISS, reversed |
| --- | --- | --- | --- | --- |
| nDCG@10 minus exact, full grades | −0.0089 | −0.0102 | −0.0032 | −0.0075 |
| Relevant exact top-10 documents dropped, arrivals of snapshot 1 / 2 / 3 | 2.2 / 4.1 / 4.0% | 2.2 / 3.7 / 3.3% | 1.5 / 3.0 / 2.8% | 2.2 / 3.3 / 1.9% |
| Substitutes per query; officially graded; relevant under full grades | 0.38; 23%; 37% | 0.36; 24%; 36% | 0.33; 20%; 37% | 0.34; 20%; 36% |

As arrived minus reversed, nDCG@10 differs by −0.0012 [−0.0068, +0.0041] (official) and +0.0013 [−0.0052,
+0.0079] (full grades) for Epzero, and +0.0027 [−0.0013, +0.0068] and +0.0043 [−0.0004, +0.0095] for FAISS.

What this shows: **at the top 10 the order effect is not detectable** with 374 queries; every interval
contains 0 and spans about ±0.006. Reversing the order does lower the newest arrivals' share of relevant
documents dropped (4.0 to 3.3% for Epzero, 2.8 to 1.9% for FAISS), the direction the neighbour results
predict, but in every order later arrivals' relevant documents are dropped more than first arrivals',
which is query difficulty again. The honest summary: the arrival effect is a fact about what the index
retrieves; on these queries its cost in ranked relevance is within ±0.01. A collection with more
fresh-content queries, or a top-100 metric, could show it; this one cannot.

---

## 13. A repair, and how it was found

**The idea.** Late insertions lack the incoming links that earlier documents keep receiving from later
arrivals. Stand in for those missing later arrivals: give each insertion extra incoming links, from its
closest construction candidates that it did not choose, and reconnect each new batch once at the end.

**How it was found** (development screens on a 300,000-document LongEval sample; every design and
prediction committed before its run; records `docs/screen-d1-*.md` to `docs/screen-d17-*.md`):

| Screen | What | Result |
| --- | --- | --- |
| D1 | +4 incoming links to every newest-cohort document after the build; donors closest, random or best-connected (FAISS) | Excess at ef 200 1.36 → 0.35–0.42 points; donor choice made no difference (prediction against); oldest cohort unharmed |
| D2 | A guaranteed floor of incoming links at insertion, 2 / 4 / 8 | No effect (prediction against): the gradient is passive accumulation (§10). Option removed |
| D3 | `extra_backlinks` 4 / 8 at insertion | Excess 1.62 → 0.78 / 0.41 points; oldest +0.08 / +0.16; work +0.4 / +0.9%; build +2 / +4% |
| D4 | 8 links on SIFT, GloVe and two more LongEval seeds | Excess cut 67–87% everywhere; oldest +0.05 to +0.21 points; work +0.6–5.2% |
| D7 | The default build path: 8-bit codes, refined bulk build of snapshot 1, later snapshots inserted | The bias reaches users there too: excess 2.27 points grown, 0.22 rebuilt; 4 links cut it only 37% (prediction against) |
| D8 | Why the default build path is worse | The refine pass of the bulk build: without it the grown excess is 0.91 points smaller |
| D9 | Refine each snapshot's insertions, alone and with 4 links | Alone −31% (prediction against); together −77% (2.27 → 0.53 points) |
| D13 | 8 links with each batch refined | Excess 0.63 → 0.28 points (rebuild 0.12) |
| D14 | Also refine the older nodes the new ones link to | Widens the gap (prediction against): the oldest cohort gains most |
| D15 | Refine each new batch harder: width 100 instead of 50, or two passes | Width 100: excess 0.28 → −0.08 points, below a rebuild's 0.12; two passes no gain |
| D17 | A search width calibrated once and kept, grown and repaired | Recall falls 1.1 points but ends at its 0.95 target, so the recall leg is not shown (prediction against); the drift is borne by later arrivals (excess 2.05 points); repaired: 0.20 |

**The full-scale confirmation** (`LONGEVAL_ENGINE_ARM.md`; three runs, each with its design and decision
margins recorded first). In Epzero's default build path on all 1,661,900 documents, ef 200, three seeds. The
decision margins (decision 3 of 23 November): the newest cohort's within-query excess must fall, with
the interval of the change excluding 0, to at most 1.0 point; the oldest cohort's miss rate may rise by
at most 0.75 points and recall@100 fall by at most 0.5 (upper 95% bounds).

| Run | Repair | Development excess (381 queries) | Confirmation excess (100 queries) | Outcome |
| --- | --- | --- | --- | --- |
| 1 | 4 links + refine new at 50 | 3.41 → 1.54 | 1.46 → −0.32 | Development missed the 1.0 target |
| 2 | 8 links + refine new at 50 | 3.43 → 1.17 | 1.52 → −0.36 | Development missed narrowly |
| 3 | **8 links + refine new at 100** | **3.36 → 0.57** (−2.79 [−3.41, −2.21]) | **1.12 → −0.37** | **Both pass; every prediction held** |

The third run is indistinguishable from a full rebuild: excess +0.14 [−0.48, +0.79] against it, the oldest
cohort no worse (+0.20 [−0.02, +0.42] points, less than a rebuild's +0.40), recall@100 +0.56 points. Cost:
1.09× single-threaded insertion time and 54 s of refining on 24 cores per batch, against a 182 s
rebuild; search work +2.5%.

**On the public stream** (Big-ANN MSTuring-10M). The repair takes the newest cohort's excess from 5.63 to
−0.65 points, lowers the oldest cohort's misses (−1.36), and lifts recall@100 at ef 200 from 74.7% to 80.9%,
above a rebuild's 79.2% (whose bulk build also inserts in arrival order).

**On the second encoder** (BGE). Excess 3.31 → 0.36 points on the test queries (−2.95 [−3.64, −2.27];
rebuild 0.38), 1.01 → −1.03 on the training queries; the oldest arrivals' miss rate changes by +0.08
[−0.29, +0.42] points and recall@100 rises by 0.90.

**What it does to ranked relevance** (analysis R2; design and predictions recorded first). The engine
arm's returned IDs at snapshot 3 can be scored almost completely: of the 3,973 distinct (query, document)
pairs in the three arms' top 10, only 11 have neither an official nor a local grade (bounded by setting
them to grade 3). At ef 200, 374 queries, three seeds:

| Arm | nDCG@10 minus exact: official / full grades | Relevant exact top-10 documents dropped, snapshot 1 / 2 / 3 arrivals |
| --- | --- | --- |
| grown | −0.0090 / −0.0067 | 1.7 / 4.6 / 4.3% |
| repaired (8 links, refine at 100) | −0.0082 / −0.0064 | 1.6 / 3.4 / 2.1% |
| rebuilt | −0.0065 / −0.0051 | 1.4 / 4.0 / 2.4% |

Repaired minus grown: +0.0008 [−0.0028, +0.0041] official, +0.0003 [−0.0033, +0.0035] with full grades. All
three predictions held: the repair halves the share of the newest arrivals' relevant documents that the
index drops (4.3% to 2.1%, below a rebuild's 2.4%), leaves the oldest unchanged, and costs nothing in mean
nDCG@10, which it does not measurably raise either. The neighbour-level effect reaches the relevant
documents; at these queries' size, mean nDCG@10 cannot show it.

---

## 14. What the labels can support

**Coverage.** The labels cover a depth-10 pool that includes the exact baseline, so 99.5–99.6% of exact
search's top 10 is graded, but 76–82% of the documents that an HNSW index substitutes into the top 10 are
ungraded and count as non-relevant. Under lenient labels a substitute costs little when it is graded
and its full gain when it is not, so the measured gap partly measures missing judgments.

**A local judge.** We graded every pooled pair (6,298, with every arm's top 10 at every snapshot, plus 22
substitutions from the order rerun) with a local GPT-OSS-120B judge on an H200 GPU: the organisers'
prompt and parser, temperature 0, every unparsed answer asked again at most twice. Its settings were
chosen in a pilot: on 270 prompt-development pairs graded four times, repeat agreement was 0.874 without
a sampling parameter and 0.915 at temperature 0, and agreement with the official grades was 0.87–0.90
exact, binary 0.92–0.96 and kappa 0.85–0.88. In the grading job all 7,135 pairs got a usable answer on the
first attempt.

**Validation** (thresholds fixed before any study pair was graded). On a 1,000-pair validation sample
drawn before grading, the judge met all three: binary agreement with the official grades 0.956 [0.941,
0.971] (required 0.90), linear-weighted kappa 0.884 [0.853, 0.912] (0.80), and nDCG@10 of complete lists
within 0.0026 of the official one at every snapshot (0.02).

**Table 4.** Gap at snapshot 3 and D under three scoring rules (*condensed* removes ungraded documents
from every ranking; *full grades* uses the local judge's grades for every pooled document).

| Arm (calibrated) | Gap: official | condensed | full grades | D: official | condensed | full grades |
| --- | --- | --- | --- | --- | --- | --- |
| Epzero HNSW | −0.0117 | −0.0056 | −0.0082 | −0.0059 | −0.0048 | −0.0050 [−0.0117, +0.0008] |
| FAISS HNSW | −0.0055 | −0.0007 | −0.0021 | −0.0007 | +0.0004 | −0.0002 [−0.0040, +0.0039] |
| IVF, frozen | −0.0080 | −0.0017 | −0.0030 | −0.0015 | +0.0003 | +0.0001 [−0.0021, +0.0025] |

Graded in full, the gap is 0.002–0.008 at every snapshot, against 0.005–0.012 with the official labels:
30% smaller for Epzero, about 60% for FAISS and IVF. Under the local grades alone, scoring only the
officially graded documents instead of all of them widens the gap by 0.002–0.007 for every arm and
snapshot. The pattern over time is unchanged: coverage moves D by at most +0.0018 [−0.0004, +0.0041], so
the equivalence results of §6 stand, with Epzero's arm still the inconclusive one.

**Leniency.** On the 29 test queries that also have human grades (1,581 pairs), the LLM labels call 87% of
the pairs relevant against 60%, with a linear-weighted kappa of 0.24 [0.14, 0.36]; exact search scores
0.93 nDCG@10 under the LLM labels and 0.56 under the human grades, uncorrelated per query. The local
judge, validated against the same lenient labels, shares their leniency: the full grades measure the
loss under these labels, not under human ones.

**What uses no labels.** The results on neighbour recall, index history and arrival (§5, §7, §9–§11, §13)
use no relevance labels at all.

---

## 15. The extension protocol and its amendments

After the frozen study, a second protocol, `EXTENSION_PROTOCOL_V1`, fixed the follow-up work before it
ran. Six open decisions were settled on 23 November 2026 (the advisor being unavailable):

| # | Point | Decision | Outcome known when decided? |
| --- | --- | --- | --- |
| 1 | Judge sampling | Temperature 0 | Judge pilot only |
| 2 | Judge thresholds | The three thresholds of §14 as the gate | No |
| 3 | Judging pool | Primary arms, shuffled-order runs, the ef sweep (6,298 pairs) and the order rerun's substitutions | No |
| 4 | Reversal test | Snapshot 3 against snapshot 1; confirmation on the 100 training queries | Yes for the test queries; no for the confirmation queries |
| 5 | Human audit | The arrival-order audit replaces an earlier sample; a second annotator sought | No |
| 6 | Second encoder | BGE-base-en-v1.5, revision pinned before encoding; margins ±0.01 | No |

Intervals are 95% paired query-bootstrap intervals with 10,000 draws, seeds averaged within queries. The
second encoder's three within-encoder arm tests form one Holm family at 0.05.

**Amendments** (`EXTENSION_AMENDMENTS.md`), each recorded before the analysis it affects:

- **3.** The BGE insertion sample is written by a new module, `bge_sample.py`, from the BGE vectors, as the
  frozen code wrote the Qwen sample.
- **4.** The BGE held-out inputs are prepared by `bge_heldout.py`, through a stand-in loader, so that the
  frozen cores run unchanged.
- **5.** The order rerun's substitutions are graded as a second grading folder, so that they do not change
  the primary scoring universe.
- **6.** The BGE held-out inputs are checked by content, not by file bytes: the 15 jobs prepared identical
  inputs but wrote different gzip timestamps and temporary paths, which the byte-level check would have
  refused. Checked before any analysis ran.
- **7.** Seven further construction seeds on the BGE vectors (`LONGEVAL_BGE_SEEDS.md`), as an exploratory
  follow-up beside E4's confirmatory three-seed results, which it does not change. Package
  `longeval-bge-seeds-01` is built; the jobs have not run.

**Predictions for the next release** (`NEXT_RELEASE_PREDICTIONS.md`, fixed on 26 November 2026, before any
2027 data or schedule existed): seven predictions (index history, the arrival ratio and its reversal, IVF's
small excess, D within ±0.01, the loss's concentration, the repair, calibration optimism) to be tested
unchanged on the next LongEval-Sci release's new snapshot and new test queries. It is the only test of
these findings on queries that no analysis here has seen.

**Pre-registered predictions for the second encoder** (fixed on 23 November, before any BGE vector
existed), and what happened:

| Prediction | Outcome |
| --- | --- |
| 1. As arrived above 1.10, shuffled within [0.90, 1.10], reversed below 0.95, and the newest cohort's excess change (as arrived minus reversed) above +1 point with its interval excluding 0, for both HNSW implementations | Held for both |
| 2. Overall recall@100 differs by less than 1 point between orders | Held on the test queries (0.6 and 0.8 points); 1.18 for Epzero on the confirmation queries |
| 3. Frozen IVF's neighbour-level excess below 1 point | Held (−0.02 points) |
| 4. D within ±0.01 for FAISS and IVF; Epzero's |D| below 0.006 | Held |
| Engine arm: both readings pass, as on Qwen and the stream | Held |

---

## 16. Discussion

**For longitudinal evaluation campaigns.** The loss to approximate search is small and, for two of three
calibrated arms, stable within ±0.01 as the collection nearly doubled. But the loss at each snapshot, its
change over time and the recall actually reached on the test queries are different quantities, and a
fixed search setting keeps none of them stable. A campaign that reports ANN runs should report neighbour
recall per snapshot beside nDCG, and should know that an appended graph serves its newest documents
worst, which is the content a longitudinal evaluation is about. Pooled LLM labels also make the gap look
larger than it is, because the documents an approximate index substitutes are mostly ungraded.

**For operators.**

- Rebuilding an append-only index from the same history gains nothing (§5).
- Rebuilding in a new order only moves the misses (§9, §12).
- Extra incoming links for each insertion and a refine of each new batch remove the gap between arrivals
  for 9% more insertion work (§13).
- What matters most is the search width: re-check it as the collection grows, with a margin on the
  target (a 0.96 target halved the loss and kept D within ±0.01 for both HNSW arms, at about twice the
  latency, §7), and with calibration queries drawn like the queries they stand for, since a larger set of
  easier queries makes the choice worse, not better; or target about 98% recall at about 3.5 times the
  latency (§8).
- A system whose users mostly look for new content should measure recall on those documents, not only on
  average: averages hide the effect (§9).

**Why the loss is small.** Approximate search misses about 6.3% of the exact top 100 at ef 200 but only
3.4–3.7% of the top 10; the top-10 neighbours it misses are less relevant than those it keeps (mean grade
0.77–1.01 against 1.54); and many of its replacements are relevant documents the release never graded.

**Relation to prior work.** Elliott and Clark [20] showed that HNSW recall depends on insertion order, and
CleanANN [26] that FreshVamana's insertion is not robust to the order of insertion (recall per round
differs between inserting whole clusters together and a random order, about 10 points below a rebuild),
which it repairs for every insertion with bridge edges. We show *which documents* bear the loss when a
collection grows by appending, that the mechanism is exposure (each later arrival chooses an existing
node in proportion to its age, with no preferential term), and that a targeted repair removes it, at
neighbour level and in the relevant documents dropped. Streaming ANN benchmarks (Big-ANN [24],
CANDOR-Bench [25]) average recall over the collection and so cannot see it. GRAB-ANNS names newcomer
isolation qualitatively and PQR-Search shows that average recall hides per-query failures; the full
novelty assessment, last checked on 26 November 2026, is in `docs/arrival-bias-research.md` §7.

---

## 17. Limitations

- **One collection, append-only.** LongEval-Sci only grows: the equivalence result does not cover
  deletions or rewritten documents. The insertion-position effect was reproduced elsewhere (GloVe, SIFT,
  a third implementation, the Big-ANN stream); the effectiveness results were not.
- **Two encoders, one disagreement.** Epzero's small change in D under Qwen did not reproduce under BGE;
  with three seeds, whether its arm agrees across encoders within the margin is open. Ten Qwen seeds put
  its D at −0.0034, within the margin; ten BGE seeds are prepared (amendment 7).
- **Lenient labels.** The effectiveness results rest on lenient LLM labels, which the local judge was
  validated against and shares; human grades cover 29 queries at one snapshot, and agree weakly.
- **Reused test queries.** The test queries were used once for the frozen analysis, and every
  exploratory analysis reuses them. The pre-registered replications (confirmation queries, second
  encoder, the public stream) are the defence, with predictions fixed for the next release's new queries.
- **A post-hoc measure.** The within-query ratio was chosen after the pooled result was seen, and it
  averages over query groups whose ratios differ (§9).
- **Few seeds.** Three construction seeds cannot separate seed variation from Epzero's change; ten seeds
  (§6, §7) put every arm within the margin, but were exploratory.
- **A coarse calibration.** 50 calibration queries and a coarse ef grid, a design choice that separates
  the two HNSW arms.
- **Development-only screens.** The query-driven repair results come from development screens.
- **The repair at the top 10.** The arrival repair was confirmed at full scale and on the stream by
  neighbour recall; on judged relevance it halves the newest arrivals' dropped relevant documents, but on
  374 queries its effect on mean nDCG@10 is within ±0.004 and not detectable (§13), as the order effect's
  is (§12).

---

## 18. What remains open

- **The human audit** (prepared, not yet graded): from the as-arrived and reversed builds of seed 11 at
  ef 200, 50 documents dropped from an exact top 10 that belong to snapshot 3 and 50 that belong to
  snapshot 1, and the 100 documents put in their place, presented blind to role, arrival group and order,
  on the organisers' 0–2 scale. It is the human check on §12. A second annotator is sought for a blind
  50-pair overlap.
- **Deletions and updates.** Every result here is for append-only growth.
- **A second longitudinal collection,** such as LongEval Web, for the effectiveness results; the
  neighbour-level arrival test needs no labels and is the cheap first step.
- **The ten BGE seeds** (amendment 7), packaged and ready to submit.
- **The next release**, against the predictions fixed for it (§15).
- **Why the disadvantage fades at the rate it does.** Screen M1 measured the rate (the in-degree gap
  closes by about a third over the second half of a build) and that pruning of the oldest nodes' links
  drives it; a closed form for s(i) and r(i) is not yet derived.

---

## 19. Reproducibility

The artifact holds the frozen protocol and its hash receipt, the extension protocol and its amendments,
every job script and packager, per-job evidence checksums, and the analysis code with its tests.
LongEval-derived data (vectors, texts, judgments, run files, document identifiers) are not
redistributed; the scripts rebuild them from the official release. Aggregate results are committed
beside the code (`research/longeval/*.json`).

---

## 20. Glossary

- **ANN index:** a data structure that finds most of a query's nearest vectors without scoring all of them.
- **Arrival snapshot (cohort):** the first snapshot that contains a document.
- **Calibration:** choosing the smallest search width that reaches a target recall on a sample of queries.
- **D:** the change in the effectiveness gap from snapshot 1 to snapshot 3.
- **ef:** HNSW's search beam: the number of candidates kept while searching. Larger ef: higher recall, more time.
- **efConstruction:** the beam used while inserting a document into the graph.
- **Effectiveness gap:** nDCG@10 of the approximate ranking minus nDCG@10 of exact search, same queries and vectors.
- **Excess miss rate:** the newest cohort's miss rate minus the oldest's, within the same query and rank band.
- **In-degree:** the number of incoming links a graph node has.
- **M:** HNSW's links per node (2M on the base layer).
- **Mantel–Haenszel ratio:** a risk ratio stratified by query and rank band, so that queries of different
  difficulty are compared only with themselves.
- **Neighbour recall@k:** the share of the exact top-k documents an index returns.
- **nDCG@10:** a ranked relevance measure over the top 10 documents.
- **nprobe, nlist:** IVF's number of clusters searched, and of clusters in total.
- **Panel:** the 374 test queries with a positive judgment at every snapshot.
- **Pre-registered:** designed, with its prediction, and committed before the run.
- **Substitute:** a document in an approximate top 10 that is not in exact search's top 10.

---

## 21. References

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
26. Zhang, Wei, Engels and Shun. CleanANN: Efficient and Robust Full Dynamism in Graph-Based Approximate Nearest Neighbor Search. SPAA 2026. https://doi.org/10.1145/3816782.3819219 (arXiv:2507.19802)
