# Late Arrivals Are Found Last: Arrival Bias in Incrementally Built Graph Indexes

*Working draft for PVLDB (plan: `docs/plan-paper-a.md`; evidence: `docs/paper-a-evidence-map.md`). Started on
28 November 2026. Results marked* pending *wait for screens A1, A2, A4 (LongEval in-degrees) and A5.*

## Abstract

Graph indexes for approximate nearest-neighbour search are usually grown by inserting vectors one at a time, and
evaluated by recall averaged over the collection. We show that this average hides a systematic disadvantage for
the vectors inserted last. In three HNSW implementations, an independent implementation of FreshDiskANN's
insertion and DiskANN's own library, on a real growing collection (LongEval-Sci, 1.66 million documents), a
public streaming runbook (Big-ANN MSTuring-10M) and public data with randomly assigned arrival cohorts, a
document's exact neighbours are missed up to twice as often when it arrived last as when it arrived first, for
the same query and rank band. Inserting the same cohorts in reverse order reverses who loses, while overall
recall moves by less than half a point. The mechanism is exposure: each later insertion links to an existing
node with probability k_j/(j − 1), so a node's incoming links grow with ln(n/i), and pruning removes them at a
rate set by the mean size of a full neighbour list; a model with no parameter fitted to the build it predicts
reproduces in-degree by insertion position to within 0.3 links. DiskANN's 1.3× slack on neighbour lists, not
its frozen start point, is what protects its late arrivals. A repair that asks each insertion's closest unchosen
candidates for a link and refines each batch removes the bias for about 9% more insertion work. Heavy deletion
with tombstones doubles the bias; *pending: compaction with the repair*.

## 1. Introduction

Vector search systems add data continuously: documents, products and messages arrive in order, and rebuilding a
graph index from scratch for every batch is expensive. Incremental insertion, as in HNSW [Malkov and Yashunin] and
FreshDiskANN [Singh et al.], is therefore the common path, and streaming benchmarks [Big-ANN; CANDOR-Bench]
evaluate it by recall averaged over the queries and the collection.

That average treats every vector alike. But in a growing collection the newest vectors are often what users look
for, and they are exactly the ones an incrementally built graph has had the least time to connect. We ask
whether a graph index finds its newest vectors less often than its oldest, why, and what it costs to fix.

**Contributions.**

1. **A measurement** that separates arrival from query difficulty: for each query and exact-rank band, the miss
   rate of a later cohort against the first (a Mantel–Haenszel ratio), and a controlled reversal that holds the
   final collection, queries and exact targets fixed and moves the same cohorts to the other end of the build.
2. **The bias across systems**: three HNSW implementations and two Vamana implementations, a real growing
   collection, a public streaming runbook, random cohorts on public data, *pending: arXiv in submission order*.
3. **The mechanism**, derived and tested: exposure, exact without a fitted parameter, minus pruning, whose rate is
   set by the mean size of a full neighbour list; and the design choice that removes most of the bias in DiskANN
   (list slack).
4. **A repair** and its cost, with deletions, and against published repairs (*pending: CleanANN, NGFix at equal
   cost*).

## 2. Setting

An incremental graph index inserts vectors x_1, ..., x_n in order. Insertion j searches the current graph, chooses
k_j ≤ M out-neighbours with a pruning rule (HNSW's heuristic, or Vamana's RobustPrune with α > 1), and adds the
reverse link to each chosen neighbour, pruning that neighbour's list when it would exceed its bound M₀ (2M in
HNSW's base layer; R, or ⌊1.3 R⌋ before pruning in DiskANN). Search is a beam search of width ef (or L) from an
entry point.

**Arrival cohorts.** We split the insertion order into cohorts: LongEval's three snapshots, Big-ANN's runbook
thirds, or random thirds on public data. A query's exact top 100 (exact scan, re-scored in float64) holds targets
from every cohort.

**The within-query ratio.** For each query and exact-rank band (1–10, 11–50, 51–100) we count targets returned and
missed by cohort, and combine the strata into a Mantel–Haenszel risk ratio of a later cohort's miss rate against
the first's. Comparing within a query matters: queries whose neighbours are mostly new are harder in every
insertion order, and a pooled ratio confounds arrival with difficulty. Intervals resample queries (and seeds
where there are several).

## 3. The bias across systems

**A real growing collection.** LongEval-Sci [CLEF LongEval 2025] grows from 1.0 to 1.66 million documents over
three snapshots. Rebuilt on all 1,661,900 documents in three orders (as arrived, shuffled, cohorts reversed; three
seeds; M 16, efConstruction 100; ef 200; 381 queries), snapshot-3 arrivals are missed more as arrived and less
reversed:

| Graph | Implementation | As arrived | Shuffled | Reversed |
| --- | --- | --- | --- | --- |
| M 16 | FAISS HNSW | 1.28 [1.12, 1.47] | 1.03 [0.93, 1.14] | 0.82 [0.72, 0.92] |
| M 16 | Epzero | 1.42 [1.25, 1.63] | 1.00 [0.90, 1.10] | 0.82 [0.72, 0.92] |
| M 32 | FAISS HNSW | 1.66 [1.40, 2.01] | 0.99 [0.87, 1.12] | 0.67 [0.54, 0.81] |
| M 32 | Epzero | 1.65 [1.38, 2.02] | 0.99 [0.87, 1.11] | 0.69 [0.54, 0.82] |

Overall recall@100 differs by at most 0.44 points between orders, while the newest cohort's miss rate moves by 2.5
points and the oldest's by 1.6 in the opposite direction: an average over the collection cannot see the effect. A
denser graph (M 32) raises recall from 0.93 to 0.96–0.97 and sharpens the bias. hnswlib gives the same ratios on a
300,000-document sample (as arrived 1.34 [1.20, 1.51], shuffled 1.00, reversed 0.77), and a second encoder
(BGE-base) 1.31 / 0.99 / 0.81.

**Position, not content.** With arrival groups assigned at random on SIFT and GloVe, shuffled ratios are 0.92–1.03
and reversed ratios' intervals lie below 1. In shuffled builds the miss rate rises steadily with insertion position
(4.9–5.0% in the first tenth to 7.4–7.8% in the last, LongEval), with position independent of content.

**A public streaming runbook.** Big-ANN's clustered MSTuring-10M runbook streams 10 million vectors as 32 clusters,
each arriving with its own neighbours. Reversing the order changes the gap between the newest and oldest thirds' miss
rates by −11.22 [−11.52, −10.91] points (Epzero) and −9.08 [−9.38, −8.79] (FAISS HNSW); a vector's position within
its own cluster adds only 0.7–1.1 points. What counts is the global insertion position.

**Vamana.** An independent implementation of FreshDiskANN's insertion shows the bias as strongly as HNSW (as
arrived 1.32 on LongEval, 1.99 on SIFT; reversed 0.65 and 0.42). DiskANN's library shows it on SIFT (1.43) but
hardly on LongEval (0.99); §5 explains why.

**arXiv in submission order.** *Pending (screen A1): about 2.8 million abstracts, BGE-base vectors, inserted in the
order they were submitted, new-content and uniform held-out queries.*

## 4. The mechanism

**Exposure.** If insertion order is independent of content, insertion j chooses any given existing node with
probability k_j/(j − 1), so node i is chosen by Σ_{j>i} k_j/(j − 1) ≈ k̄ ln(n/i) later arrivals. Measured in
300,000-vector builds of LongEval and SIFT at M 16 and M 32, the number of later choosers matches this sum to within
1% in every tenth of the build: there is no preferential term, only time. The number of neighbours an insertion keeps
grows over the build (13.1 to 17.1 at M 32, 10.6 to 12.3 at M 16), and the sum must use each insertion's own count.

**Pruning.** A node's in-degree is the links kept from later choosers, S(i), plus reverse links in earlier nodes'
lists, R(i). Recording every link's creation and removal in a 300,000-insertion build shows how lists are pruned:
in bulk (a full list of 33 candidates is cut to about 21), mainly by length (the farthest fifth of a list is removed
2.6 times as often as the closest), with a smaller effect of a link's age of its own (1.4 times, oldest against
youngest fifth).

**A model with nothing fitted to the build it predicts.** Let L̄ be the mean size of a list that has been pruned at
least once, measured in a build with another seed (23.4 at M 16). A link in a full list is removed at rate 1/L̄ per
addition; the additions after position p number A(p) = Σ_{j>p} k_j/(j − 1). Then

  S(i) = Σ_{j>i} (k_j/(j − 1)) · exp(−max(0, A(j) − (L̄ − k_j)) / L̄),  R(i) = k_i · exp(−A(i)/L̄).

At M 16 this predicts S within 0.3 links and R within 1.4 links in every tenth of a build it was not fitted on; the
oldest tenth keeps 26.0 links from later nodes against 26.3 predicted. At M 32 the reverse links that newer nodes
keep are under-predicted by up to 4.3 links; a simulated list process with the measured bulk-pruning rhythm and age
profile brings that to 1.8 but over-predicts the oldest tenth's S, which loses more than its age alone explains,
consistent with pruning by length. Exposure is exact; pruning is bounded by the two models to within about 4 links
in the oldest tenth and 1.8 elsewhere.

**Consequence.** The newest nodes enter with about k̄ incoming links and gain few before the build ends; the oldest
end with about 2.4 k̄. Late nodes are not unreachable (at most 0.06% have no incoming link); at ef 200 no missed
target was ever scored by the search. The misses are coverage failures of the traversal.

## 5. Why DiskANN protects its late arrivals

The independent Vamana driver and DiskANN's library differ in two known ways: DiskANN appends reverse links until a
list would exceed ⌊1.3 R⌋ before pruning it back to R, and it starts searches from a frozen node. Adding each to the
driver (R 32, L 100, α 1.2):

| Driver variant | SIFT, as arrived | LongEval, as arrived | Newest tenth's in-degree, SIFT / LongEval |
| --- | --- | --- | --- |
| as published (FreshDiskANN) | 1.99 | 1.32 | 18.6 / 17.4 |
| + list slack 1.3 | 1.29 [1.07, 1.56] | 1.04 [0.92, 1.17] | 25.0 / 27.4 |
| + frozen start, copy of the first vector | *pending* | *pending* | *pending* |
| DiskANN's library | 1.43 | 0.99 | 26.4 / 28.1 |

Slack alone moves the driver to DiskANN's behaviour on both datasets: a newcomer's reverse links are kept until a
list holds 41 links, so its incoming links are not pruned away as the build continues. (A frozen node at the data's
mean, which DiskANN does not use, wrecked the graph on unit-normalised embeddings: α-pruning let it occlude most
other candidates.)

## 6. Repair

**The idea.** A late insertion lacks the incoming links that exposure gives earlier nodes from later arrivals. Stand
in for those later arrivals: when a vector is inserted, ask its closest construction candidates that it did not
choose to link to it (`extra_backlinks`), and reconnect each new batch once (`refine_links` from the batch's first
position, at a search width of 100). Both are a few lines in an HNSW implementation; neither is saved in the index or
changes search.

**How it was found.** Seventeen development screens on a 300,000-document LongEval sample, each with its prediction
recorded before its run, narrowed the design: extra incoming links remove most of the newest cohort's excess at
small cost (4 links: 1.62 → 0.78 points, 8 links: → 0.41); a guaranteed floor of incoming links at insertion does
nothing, because the gradient is accumulation, not rejection; refining each batch at width 100 removes the rest.

**At full scale** (all 1,661,900 LongEval documents, three seeds, ef 200, decision margins recorded first):

| Repair | Newest cohort's excess (381 queries) | Confirmation queries (100) |
| --- | --- | --- |
| none (grown) | 3.36 points | 1.12 |
| 8 links + refine each batch at 100 | 0.57 (−2.79 [−3.41, −2.21]) | −0.37 |
| rebuild | 0.43 | |

The repaired index is indistinguishable from a rebuild (excess +0.14 [−0.48, +0.79] against it), the oldest cohort is
no worse (+0.20 [−0.02, +0.42] points, less than a rebuild's +0.40), and recall@100 rises 0.56 points. The cost is
1.09× single-threaded insertion time and 54 s of refining on 24 cores per batch, against a 182 s rebuild; search work
rises 2.5%. On the Big-ANN stream the newest third's excess goes from 5.63 to −0.65 points and recall@100 at ef 200
from 74.7% to 80.9%, above a rebuild's 79.2%. With a second encoder, 3.31 → 0.36 points.

**With deletions.** Sparse deletions (10% of the live documents tombstoned before each later cohort) leave the bias as
it was (+1.88 grown against +1.64 rebuilt) and the repair removes it (−1.93 [−2.36, −1.52] points). Heavy deletion
with tombstones (Big-ANN's deletion runbook: 10 million inserted, 5.4 million deleted) doubles it, in both
implementations measured:

| Arm (Epzero, three seeds) | Newest cohort's excess | Miss rate, oldest / newest third | Recall@100 |
| --- | --- | --- | --- |
| grown, tombstones | +8.37 [+7.99, +8.74] | 13.5 / 18.6% | 0.772 |
| rebuilt from the live set | +4.22 [+3.81, +4.62] | 20.1 / 21.6% | 0.731 |
| repaired, tombstones kept | +6.20 [+5.86, +6.53] | 11.4 / 14.5% | 0.827 |
| rebuilt and repaired | +1.23 [+0.87, +1.60] | 17.0 / 16.2% | 0.797 |

Tombstoned nodes keep routing searches, which raises recall by 4 points but mostly for the old documents built around
them; new arrivals cannot link to them. The repair alone helps every cohort but closes only a quarter of the gap.
Compaction and the repair address the two causes separately: rebuilding the live set removes the tombstones' routing
(+8.37 → +4.22), and growing it with the repair removes the arrival order's (+4.22 → +1.23), 85% of the grown index's
excess in all, with recall 2.5 points above it. Keeping the tombstones gives the highest recall (0.827) but most of
the bias.

**Against published repairs.** CleanANN's GuidedBridgeBuild links nodes of the same depth in an insertion's search
tree, at depths ⌊log₂|D|⌋ + 2 to + 4. With the authors' code at that window (R 32, L 100, α 1.2), the bridges barely
act: the as-arrived ratio is 1.47 without and 1.50 with them on SIFT 300K, and 1.01 with and without on LongEval
300K, at no extra build time, because the insertions' search trees rarely reach depth 20. CleanANN's arrival
behaviour is DiskANN's (1.43 and 0.99), set by the list slack it inherits (§5). In a depth probe on 50,000 SIFT
vectors, shallower windows that do fire lowered recall@10 by about a point at 1.1–3.6× the build time. *Pending: a
shallow-window arm at 300K; NGFix, which needs new builds scored by arrival.*

## 7. Related work

**Insertion order.** Elliott and Clark showed that HNSW's recall depends on insertion order when the order is
correlated with the data (ordering by intrinsic dimensionality or category moves recall by up to 12 points).
CleanANN showed that FreshVamana's insertion is not robust to the order of insertion on clustered synthetic data
(recall per round about 10 points below a rebuild under a uniformly random order) and repaired it with bridge edges.
Neither measures which vectors bear the misses: both report recall over the collection.

**Dynamic indexes.** FreshDiskANN introduced incremental Vamana with lazy deletion and consolidation; MERIT shows that
tombstoned nodes keep being visited and use up the search budget, and replaces them in place; Ada-ef shows that a
search-width table fitted before insertions falls slightly below its target after them. GRAB-ANNS names newcomer
isolation qualitatively. None reports recall by arrival, or a model of in-degree by insertion position.

**Evaluation.** Streaming benchmarks (Big-ANN's streaming track, CANDOR-Bench) average recall over the queries and
the collection; PQR-Search and VIBE's hard-query split show that averages hide per-query failures. Our measure adds
the other axis: which documents, by when they arrived.

**Growing networks.** In preferential-attachment models a node's degree grows with its age; the HNSW and Vamana graphs
studied here grow by uniform attachment (exposure) and lose links through pruning, a mechanism without a
preferential term, which we model in §4.

## 8. Limitations

- **One labelled real collection.** LongEval-Sci is the only real growing collection with relevance labels; the
  neighbour-level bias is replicated on public data, a public runbook and (pending) arXiv, the effectiveness results
  only on LongEval.
- **The measure was chosen after the first pooled result.** Every replication since (order rerun, second encoder,
  Big-ANN stream, hnswlib, Vamana, deletions, A1-A5) used it as recorded before its run.
- **The pruning model is an approximation.** Exposure is exact without a fitted parameter; the measured-rate model
  predicts in-degree within 0.3 links at M 16 but under-predicts newer nodes' reverse links at M 32 by up to 4.3, and
  the list-process model trades that error for the oldest tenth's.
- **Single-threaded construction.** All builds insert one vector at a time on one thread, so that the order is
  exact; concurrent insertion changes the effective order and is not studied.
- **Implementations.** Three HNSW implementations (FAISS, hnswlib, Epzero) and two Vamana (DiskANN, an independent
  driver); other graph families (NSG, τ-MNG, RoarGraph) are untested.
