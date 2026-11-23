# Arrival bias in growing vector indexes: what we found, what is known, and how we plan to fix it

Working document, 23 November 2026. Revision 2 (same day) applies the corrections of an independent
review: the headline is now the controlled order reversal with absolute risks; the reciprocity,
"every miss is unreachable", IVF, wider-search, M 32 and cost statements are corrected; prior public
proposals and new related work are added; the success criteria are absolute; and the plan adds
paired target analysis, an equal-degree donor intervention, durability curves and an order-invariant
partition baseline. It covers the research direction, the evidence, the literature
review, current competing work, the plan for the engine, and the experiments that remain. Results
marked *exploratory* were obtained after the frozen protocol (`FINAL_PROTOCOL_V1`) and are not part of
it. The frozen study stays the record of what was pre-registered; nothing here changes it.

Companion material:

- Literature report: [`reports/ANN index staleness bias.md`](../reports/ANN%20index%20staleness%20bias.md),
  with its source notes in [`research_notes/ANN index staleness bias/`](../research_notes/ANN%20index%20staleness%20bias/)
  (graph indexes, IVF and quantizers, benchmarks, IR and fairness, production systems).
- Earlier evidence: [`research/longeval/HELDOUT_EXPLORATORY.md`](../research/longeval/HELDOUT_EXPLORATORY.md)
  (sections 15–17) and the `insertion-*.json` aggregates in `research/longeval/`.

---

## 1. Summary

**The problem.** When a vector index grows one insertion at a time, whether a document can be found
depends on *when it was inserted*, not only on how similar it is to the query. In HNSW, the graph
index used by most vector databases, documents inserted late get about half the incoming links of
documents inserted early, and search misses them more often. Average recall, the number every
benchmark and vendor reports, hides this completely: it can stay flat while the newest part of the
collection is systematically under-served.

**The evidence so far** (LongEval-Sci 2026, three snapshots growing from 869,902 to 1,661,900
scientific documents; two independent HNSW implementations; public datasets as controls):

- **The headline (judgement-free).** Same 1,661,900 documents, same queries, same exact top-100
  targets; only the build order changes. With M 16 and ef 200, exact neighbours from the newest
  snapshot are missed **1.42× (Epzero) and 1.28× (FAISS)** as often as those from the oldest when
  built as the collection arrived, 1.00 and 1.03 when shuffled, and **0.82** for both when the
  snapshot blocks are reversed. In absolute terms the newest group's excess miss risk moves from
  **+2.79 to −1.49 percentage points** (Epzero; FAISS +1.93 to −1.48). Moving the same cohorts to the
  other end of the build reverses which cohort loses more neighbours, within queries and rank bands.
  The pre-specified contrast (rank-standardized ratio, as arrived minus reversed) was +0.77
  [0.58, 0.98] for Epzero and +0.62 [0.46, 0.80] for FAISS; the within-query version was added after
  inspecting that result.
- **Random cohorts on public data** show the same pattern without any drift (SIFT: 1.92 as inserted,
  0.99 shuffled, 0.48 reversed; GloVe 1.17 / 0.99 / 0.80). The size varies a lot by workload.
- **Relevant documents.** Among query–document pairs graded relevant in the exact top 10, HNSW drops
  new ones 2.1–2.8× as often as old ones (raw risk ratio: 4.0% vs 2.0% for Epzero at snapshot 3).
  Within the same query the ratio is weaker and mostly not significant (Epzero 1.28 [0.86, 1.96];
  FAISS calibrated 1.70 [1.01, 3.12]). Whether this costs ranked gain, after replacements, is not yet
  measured.
- **IVF is not perfectly neutral.** Frozen IVF shows no difference among relevant top-10 pairs (0.99),
  but at the exact top-100 level (snapshots 2+3 vs 1) it has a small excess: 1.10 [1.01, 1.20],
  +0.44 pp, far below HNSW's.
- **Mechanism, partly.** Base-layer in-degree falls from 28.6 to 13.3 between the first and last tenth
  of insertions, identically in shuffled builds. Older nodes have had more chances to receive later
  forward links even without any rejection, so the gradient alone does not prove that the pruning
  rule rejects newcomers; that needs edge-event logging.
- **Wider search shrinks the absolute gap but not the ratio**: at ef 800 the excess falls to +0.80 pp
  (Epzero) and +0.58 pp (FAISS), while the ratio stays near 1.24–1.35. Existing repair passes narrow
  the in-degree gap (2.15× → about 1.4–1.5×) and reduce but do not remove the bias (1.13–1.26 at ef
  200–400).
- **Traversal, not ranking.** At ef 200 no missed target was ever scored: the search never expanded
  any node that links to it. Every scored exact target was returned. This diagnoses the traversal at
  that budget; it does not show the target is unreachable in the graph.

**What the literature says.** Among the sources checked (to 23 November 2026), we found no study
combining these controls and outcomes: fixed corpus, queries and exact targets, with only the
position of real arrival cohorts in an incremental graph build changed, across two engines and with
random-cohort controls. The surrounding ideas are not new. Insertion order changes aggregate HNSW
recall (Elliott & Clark, ICTIR 2024). Newcomer isolation has been named, with a freshness-favouring
pruning rule (GRAB-ANNS, 2026 preprint). In-degree skew has been studied as a geometric effect, with
post-hoc in-degree-aware pruning (hubness, SIGMOD 2026). A June 2026 practitioner essay (Tian Pan)
already recommends reporting recall stratified by document age, for date-sharded indexes, with
illustrative rather than measured numbers. The SIGMOD 2026 hubness paper lists rebuild-free
maintenance of degree balance and reachability under updates as an **open problem**.

**The claim, narrowed by two reviews.** "Every index goes stale" is too broad: partition (IVF) and
quantizer staleness needs real distribution drift and mostly costs speed (our IVF arm shows at most
a small excess). Exact flat search with deterministic ties is a counterexample to any universal
construction-history claim, and an IVF-Flat index with a fixed router and complete list scans gives
identical results under every insertion order. The cross-family taxonomy is therefore a set of
hypotheses. The defensible, new claim is the controlled one: **in incremental HNSW, the physical
position of a cohort in the build order changes which cohort loses exact neighbours, within the
same queries and rank bands.**

**The plan.** (1) Pin down the mechanism (why in-degree repair leaves a residual). (2) Redesign how
Epzero links new vectors so that retrievability does not depend on arrival order, at no loss in speed
or recall and without rebuilds, and add per-age recall telemetry. (3) Show this on LongEval and
public streaming workloads against every known fix. (4) Write it up: a CLEF 2027 paper on the
evaluation blind spot (the index as a hidden temporal factor), and a separate systems paper on the
engine.

### Decisions taken on 23 November 2026 (advisor unavailable)

Made by the author after two independent reviews; each can be revisited, with the change dated.

1. **CLEF 2027 paper: built on the controlled order reversal, with the Epzero fix included if it
   earns its place by a fixed date.** The paper leads with the reversal and the redistribution result
   (mean recall barely moves while cohorts swing), absolute risks beside ratios, the random-cohort
   controls, and the frozen study as the reason a mean-only view misses it. The pre-specified
   contrast is reported first; the within-query analysis is labelled as chosen afterwards. If, by
   **1 March 2027**, an Epzero design meets the margins of point 3 on the confirmation readings, the
   paper also presents it as the remedy (a bigger claim: the bias is measured, explained and
   removed); otherwise it is a measurement paper and the engine work goes to a systems paper. A
   later systems paper can extend the engine either way.
2. **Confirmation, separated from exploration.** The 100 LongEval-Sci training queries (no shared ID
   or text with the test queries; never used for any arrival-order outcome) are the confirmation set
   for the reversal, read once in the order rerun (`scripts/pace/LONGEVAL_INSERTION_ORDER_IDS.md`).
   The BGE encoder replication and a public streaming workload are the confirmation readings for any
   engine fix; the 300K sample, SIFT, GloVe and the test queries are for development only.
3. **Margins for engine work, set from measured precision.** Comparing two builds with the 381 test
   queries gives a standard error of about 0.2 points for the newest cohort's excess, 0.2–0.4 points
   for the oldest cohort's miss rate and for overall recall; recall also varies by up to 0.5 points
   between seeds of the same order (computed 23 November 2026 from the full-scale per-query
   tables, two as-arrived seeds as a null contrast). So: at least three seeds per arm, and on
   LongEval snapshot 3, as-arrived order, M 16, ef 200, exact top-100:
   - **Effect:** the newest cohort's within-query excess falls below the baseline's (+2.79 pp) with a
     paired 95% interval excluding 0, and its point estimate is at most **1.0 point**, near the
     order-independent reference (the shuffled build, −0.04).
   - **No harm:** the upper 95% bound of the increase in the oldest cohort's miss rate is at most
     **0.75 points**, and of the decrease in overall recall@100 at most **0.5 points**.
   - **Cost:** at equal measured query work (distance evaluations per query within 5%; CPU time
     reported), insertion CPU time at most 1.3× and memory at most +5%.
   With 100 confirmation queries the standard errors are about 0.5–1.0 points, so the confirmation
   reading requires the effect criterion (interval excluding 0) and reports non-harm with its
   interval rather than gating on it.
4. **Engine experiments, in order:** (a) the equal-degree donor intervention (does *who* links to a
   target matter at fixed in-degree?); (b) the simple baselines at matched work: wider search, a
   second entry point, and a fixed-router partition index with complete list scans (order-invariant
   by construction); then designs A–C and F. A batch-handoff controller is a secondary branch, given
   the lifelong gradient in section 4.6b.
5. **Extension protocol decisions** (judge temperature 0; section-3 thresholds as the gate with the
   20B bar as a second tier; the larger judging pool, including the order rerun's substitutions;
   snapshot 3 against 1 as the reversal test, recorded as decided after its outcome was seen on the
   test queries, with the confirmation queries unseen; an arrival-order human audit replacing the
   21 November sample, with a second annotator sought; BGE-base-en-v1.5 in scope with its revision
   pinned and predictions recorded) are written into `research/longeval/docs/protocol/`.
6. **The order rerun with returned IDs** (package `longeval-insertion-order-03`) comes first on
   PACE: the paired target analysis, the relevance consequence, the audit and the confirmation all
   depend on it.
7. **Preprint**: not decided. Timing risk argues for an early arXiv preprint of the measurement; the
   CLEF 2027 call (expected tentatively with a late-April or early-May 2027 deadline) must be checked
   for its policy first.

---

## 2. How we got here

### 2.1 The original study

The LongEval-Sci study asked whether approximate nearest-neighbour (ANN) search changes longitudinal
dense-retrieval evaluation: as a collection grows, does the gap between ANN and exact search grow,
and could that distort conclusions about how systems age?

- Collection: LongEval-Sci 2026, three cumulative, insert-only snapshots (869,902 / 1,322,720 /
  1,661,900 documents).
- Encoder: Qwen3-Embedding-4B, 2,560 dimensions, inner product.
- Queries and labels: 381 test queries, a panel of 374, GPT-OSS-120B relevance labels.
- Arms: Epzero HNSW, FAISS HNSW, FAISS IVF (frozen training), each maintained incrementally, three
  construction seeds, calibrated to matched recall.
- Primary quantity *D*: the change from snapshot 1 to 3 in the ANN-minus-exact nDCG@10 gap, with a
  pre-registered equivalence margin of ±0.01.

Frozen results: Epzero D = −0.0059 [−0.0129, −0.0006] (inconclusive under the frozen rule), FAISS
HNSW −0.0007 and IVF −0.0015 (both negligible). The sensitivity analyses (section 17 of
`HELDOUT_EXPLORATORY.md`) showed FAISS and IVF rule out changes larger than about 0.004, Epzero's D
comes mostly from one seed and its interval includes 0 once seeds are resampled, and label noise is
negligible (SD ≤ 0.0007).

### 2.2 Why that was not enough

The result is precise, but its message — approximation costs little and stays small — is a modest
headline for a CLEF long paper, and reviewers could reasonably call 0.004 negligible. The decision
(23 November 2026) was to look for a bigger, more general problem inside the data, improve the engine
from source to solve it, and evaluate an improved engine later as a new, pre-specified arm.

### 2.3 Where the bigger problem was hiding

The insertion-order experiments (`insertion_pilot.py`, `insertion_strata.py`) were originally a
mechanism check. They showed that HNSW misses late-inserted neighbours more often, that the effect
follows insertion position, and that in-degree falls with position. The open question was whether
this reaches *relevant* documents — the ones that matter for retrieval quality. Section 4.1 answers it.

---

## 3. The problem statement

### 3.1 Definition

Take a collection that grows over time and an index maintained by inserting each new item into the
existing structure. For a query *q* and an item *d* that exact search would return in the top *k*,
let *miss(q, d)* = 1 if the approximate search does not return *d*. Group items by arrival (the
snapshot, or decile of insertion position, in which they entered the index).

**Arrival bias** is a dependence of P(miss) on arrival group that is not explained by the query or by
the item's rank in the exact list. We measure it two ways:

- **Pooled miss ratio**: misses among new items ÷ misses among old items, over all (query, item)
  pairs. Easy to read, but it mixes in query composition: queries whose answers include new items
  may simply be harder.
- **Within-query ratio** (Mantel–Haenszel risk ratio, strata = query × exact-rank band 1–10 / 11–50 /
  51–100): compares old and new items *for the same query at the same exact rank*. This isolates the
  index's treatment of the item and is the headline measure.

Both are computed at two levels: **neighbour level** (exact top-100 as ground truth, no labels
needed, many pairs) and **relevance level** (only relevant documents in the exact top 10, judged by
the LongEval labels).

Causal controls: the same items built **as they arrived**, **shuffled**, and **reversed**. If the
ratio follows position (above 1 as arrived, about 1 shuffled, below 1 reversed), the cause is the
index's construction history, not the content of new documents.

### 3.2 Why it matters

- **Exposure.** Equally relevant new documents are shown less often. LongEval's own 2026 overview
  notes that recency does not drive relevance on LongEval-Sci, so the argument is not that new
  documents matter more, but that the index treats equally relevant documents unequally by age.
- **Freshness in RAG and search.** New content is often exactly what users need (news, new papers,
  updated documentation). An index that quietly under-retrieves it fails where freshness matters.
- **Evaluation blind spot.** LongEval studies how retrieval systems persist over time, yet the index
  itself is an unreported, time-dependent factor. Every ANN benchmark (Big-ANN streaming, VIBE,
  vendor suites) scores average recall, which cannot show how misses are distributed across cohorts
  (although a mean can still move with insertion order, and Big-ANN's runbooks contain what a cohort
  analysis needs).
- **Production exposure.** Engines that insert directly into one long-lived graph (pgvector HNSW,
  Vespa, Redis, Cosmos DB DiskANN, Weaviate) are exposed directly. Engines that brute-force a small
  fresh buffer hide it only for the newest window (minutes to hours of writes); after that, data
  moves into merged graphs where the same mechanism applies (Lucene's merge path inserts the smaller
  segments' vectors into the largest segment's graph — they become late insertions again).

### 3.3 What the problem is *not*

- Not a drift or content effect: shuffled builds remove it; random arrival groups on public data show
  it.
- Not deletion damage: LongEval-Sci is insert-only, and the "unreachable points" literature is about
  delete/reinsert churn.
- Not removed by searching wider, but shrunk: the ratio stays elevated from ef 100 to ef 800 while the
  absolute excess falls from +2.79 to +0.80 pp (Epzero) at about 3.5× the query time. Any fix must
  beat "search wider" at matched work.
- Much weaker in frozen IVF: none among relevant top-10 pairs, a small excess (+0.44 pp) among exact
  top-100 neighbours.

---

## 4. Evidence

All numbers below are exploratory (post-freeze). Bracketed ranges are 95% query-bootstrap intervals.

### 4.1 Relevant documents, by arrival (LongEval-Sci, held-out runs, three seeds averaged)

Of the query–document pairs graded relevant (GPT-OSS-120B grade > 0) in exact search's top 10, the
share that approximate search drops from its own top 10. "Pooled ratio" here is the raw ratio of the
two drop rates; "within-query" is a Mantel–Haenszel risk ratio stratified by query (at top 10 every
target is in rank band 1–10, so this is query adjustment only). Counts are pairs, not unique
documents.

| Snapshot | Arm | Relevant pairs in exact top 10 (old / new) | Dropped, old | Dropped, new | Pooled risk ratio | Within-query ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | Epzero, calibrated | 2,621 / 518 | 1.8% | 3.7% | 2.02 [1.29, 3.06] | 1.14 [0.80, 1.61] |
| 2 | FAISS HNSW, ef 200 | 2,621 / 518 | 1.1% | 3.6% | 3.15 [1.78, 5.20] | 1.69 [0.98, 2.84] |
| 2 | FAISS HNSW, calibrated | 2,621 / 518 | 0.9% | 2.5% | 2.90 [1.53, 5.16] | 1.55 [0.89, 2.85] |
| 2 | IVF, frozen training | 2,621 / 518 | 2.5% | 2.9% | 1.17 [0.65, 1.86] | 0.91 [0.57, 1.35] |
| 3 | Epzero, calibrated | 2,393 / 761 | 2.0% | 4.0% | 2.07 [1.33, 3.14] | 1.28 [0.86, 1.96] |
| 3 | FAISS HNSW, ef 200 | 2,393 / 761 | 1.4% | 3.5% | 2.59 [1.61, 4.32] | 1.60 [0.94, 2.72] |
| 3 | FAISS HNSW, calibrated | 2,393 / 761 | 1.0% | 2.8% | 2.83 [1.60, 4.98] | 1.70 [1.01, 3.12] |
| 3 | IVF, frozen training | 2,393 / 761 | 2.4% | 2.4% | 0.99 [0.59, 1.52] | 0.87 [0.57, 1.26] |

Reading: HNSW drops relevant new pairs two to three times as often (pooled); IVF shows no difference
at this level (its small neighbour-level excess is in 4.7). The
within-query ratio is weaker and mostly not significant at this sample size (761 new relevant pairs),
so part of the pooled ratio is query composition. A dropped relevant document can be replaced by
another relevant one, so the consequence for ranked gain (replacements, their grades and judgement
coverage, net change) still has to be measured. This is the result reviewers will push on hardest
(section 10.1).

### 4.2 Nearest neighbours, with causal controls (full snapshot 3, 1,661,900 documents)

Within-query ratio (Mantel–Haenszel, strata = query × exact-rank band), newest arrival group
(snapshot 3) against oldest (snapshot 1), exact top-100, mean of three build seeds with a paired query
bootstrap (conditional on those builds). "Reversed" reverses the three snapshot blocks; within each
block the order stays lexical by ID. "Shuffled" randomizes individual insertions. In shuffled builds
the original arrival groups have similar miss rates while the *physical* insertion deciles still have
a gradient, so plots must label which of the two is on the x-axis. These within-query ratios were
chosen after the pre-specified rank-standardized contrast had been seen (section 1).

| Setting | Implementation | As arrived | Shuffled | Reversed |
| --- | --- | --- | --- | --- |
| M 16, ef 100 | Epzero | 1.34 | 1.05 | 0.82 |
| M 16, ef 100 | FAISS | 1.27 | 1.06 | 0.81 |
| M 16, ef 200 | Epzero | 1.42 | 1.00 | 0.82 |
| M 16, ef 200 | FAISS | 1.28 | 1.03 | 0.82 |
| M 16, ef 400 | Epzero | 1.32 | 0.95 | 0.79 |
| M 16, ef 400 | FAISS | 1.24 | 0.98 | 0.78 |
| M 16, ef 800 | Epzero | 1.35 | 0.87 | 0.66 |
| M 16, ef 800 | FAISS | 1.24 | 0.90 | 0.71 |
| M 32, ef 200 | Epzero | 1.65 | 0.99 | 0.69 |
| M 32, ef 200 | FAISS | 1.66 | 0.99 | 0.67 |

The M 32 rows also use efConstruction 200 (against 100 for M 16), so the two settings change
together, and a lower baseline miss rate can inflate a risk ratio. M and construction width have to
be crossed before the M 32 result is interpreted.

Absolute excess miss risk of the newest group (percentage points, M 16), beside the ratios:

| Engine | ef 200 as arrived | ef 200 reversed | Paired change | ef 800 as arrived | ef 800 shuffled |
| --- | --- | --- | --- | --- | --- |
| Epzero | +2.79 | −1.49 | +4.28 [3.33, 5.27] | +0.80 | −0.38 (ratio 0.87 [0.72, 0.995]) |
| FAISS | +1.93 | −1.48 | +3.41 [2.55, 4.32] | +0.58 | −0.28 |

Heterogeneity (Epzero, M 16, ef 200, as arrived): the ratio is similar across exact-rank bands (1.35,
1.29, 1.49) but differs across queries. Grouping queries by the share of *later* arrivals among their
exact neighbours: 2.25 [1.54, 3.19] for the third with the fewest later neighbours, 1.62 for the
middle third, 1.16 [0.95, 1.39] for the third with the most. **A late arrival is most disadvantaged
when it is surrounded by older documents.** In shuffled builds the three thirds are 1.15, 1.07 and
0.92.

Three draws of snapshot-1 size that keep the arrival groups in proportion reproduce the pattern
(`insertion-order-draw{1,2,3}.json`), as do three construction seeds per order.

### 4.3 Public datasets with arrival assigned at random (300K sample, M 16, ef 200)

Arrival groups assigned at random, so the groups are statistically identical and only position
differs:

| Dataset | Implementation | As arrived | Shuffled | Reversed |
| --- | --- | --- | --- | --- |
| SIFT (128-d, L2) | Epzero | 1.92 [1.72, 2.14] | 0.99 [0.91, 1.07] | 0.48 [0.41, 0.55] |
| SIFT | FAISS | 1.72 [1.53, 1.93] | 1.00 [0.91, 1.09] | 0.50 [0.44, 0.58] |
| GloVe (cosine) | Epzero | 1.17 [1.14, 1.20] | 0.99 [0.97, 1.02] | 0.80 [0.77, 0.82] |
| GloVe | FAISS | 1.09 [1.06, 1.13] | 0.99 [0.97, 1.01] | 0.81 [0.79, 0.83] |

This is the cleanest evidence that the effect is a property of incremental construction: no drift,
no content difference, only order.

### 4.4 Mechanism: incoming links fall with insertion position

Mean base-layer in-degree by tenth of insertion order (Epzero, M 16, efConstruction 100):

| Build | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LongEval 300K sample, as arrived | 28.6 | 25.4 | 23.3 | 20.6 | 17.2 | 17.1 | 16.7 | 14.7 | 14.2 | 13.3 |
| LongEval 300K sample, shuffled | 29.4 | 25.3 | 22.6 | 20.4 | 18.7 | 17.1 | 15.7 | 14.6 | 13.4 | 12.4 |
| LongEval full 1.66M, as arrived | 28.4 | 24.6 | 22.5 | 19.5 | 16.8 | 16.3 | 16.1 | 13.9 | 13.5 | 12.7 |
| LongEval full, M 32, as arrived | 47.8 | 40.9 | 36.8 | 31.5 | 26.9 | 25.8 | 25.0 | 21.6 | 20.5 | 19.2 |
| SIFT 300K, as arrived | 28.7 | 25.1 | 23.3 | 20.4 | 19.0 | 17.5 | 16.1 | 14.8 | 13.9 | 12.5 |

The gradient is the same whatever the content and whatever the order, so it is caused by position.

**Why it happens** (Epzero `engine/src/hnsw.cpp`, `connect`, and the same logic in hnswlib/FAISS):

1. A new node searches the graph and keeps up to M **forward** links, chosen by the diversity
   heuristic. These are always written.
2. For each chosen neighbour, it tries to add the **reverse** link (neighbour → new node). If the
   neighbour's list has a free slot, the link is added. If the list is full (2M slots at the base
   layer), the neighbour **re-runs the diversity heuristic** over its existing links plus the
   newcomer. The heuristic drops a candidate if an already-kept neighbour is closer to it than the
   owner is — and in a dense, established region, the newcomer is usually "covered" by an incumbent,
   so it is rejected.
3. Old nodes keep collecting incoming links from every later arrival's **forward** links, which are
   unconditional. New nodes have had no later arrivals to link to them.

So in-degree grows with age, and the newest nodes depend on a few conditional reverse links.

**Resolved on 23 November 2026: it is mostly passive accumulation, not rejection.** Even a process
with no rejection at all, in which each arrival picks *m* incumbents uniformly, gives older nodes more
incoming links: the expected number of later choices of a node is the sum of *m/(t−1)* over the later
steps *t*. Measured directly (screen D2, `screen-d2-min-backlinks.md`): right after insertion a node
already has on average 12.5 incoming links from the 12.6 neighbours it chose (only 1.4% fewer than
4), yet at the end of the build in-degree runs from 28.3 (first tenth) to 13.7 (last tenth).
Rejection by full neighbours (step 2 above) is real but minor; the gradient comes from old nodes
receiving forward links from every later arrival nearby. A guaranteed floor of incoming links at
insertion therefore had no effect, while giving late nodes extra incoming links from nodes they had
not chosen (screen D1) removed most of their excess. Likewise, the query-tercile result (largest ratio when few of a query's answers are
later arrivals) describes queries, not each target's own neighbourhood. The stronger ratio at M 32
is confounded with efConstruction 200 and a lower base rate (4.2). GRAB-ANNS
describes the same thing informally as incumbents forming a "geometric shield" around new nodes; we
measure it and show it is driven by build position.

### 4.5 Existing repairs do not remove it (300K sample, as arrived, M 16)

| Build | In-degree, first → last tenth | Recall@100 at ef 200 | Within-query ratio at ef 100 / 200 / 400 | Extra cost |
| --- | --- | --- | --- | --- |
| As arrived | 28.6 → 13.3 | 0.946 | 1.28 / 1.31 / 1.26 | build 282 s, one thread |
| + global refine pass | 28.0 → 20.0 | 0.960 | 1.07 / 1.22 / 1.26 | +101 s wall time on 10 threads |
| + fill free link slots | 36.5 → 30.4 | 0.965 | 1.04 / 1.13 / 1.24 | +27 s wall time on 10 threads |

The costs are wall times under different thread counts. 101 s on 10 threads is up to 1,010
thread-seconds of capacity; it is not a measured single-thread time or CPU time. Cost comparisons
need equal thread counts and measured CPU time.

The repairs raise recall overall but leave new documents missed 1.13–1.26× as often at the widths
people actually use. They also **do not equalise in-degree**: the first-to-last-tenth in-degree ratio
falls from 2.15 to about 1.4 (refine) and 1.47 (fill), so the residual bias is consistent with the
remaining in-degree gap.

### 4.6 Anatomy of a miss (300K sample, ef 200, k 100)

Every exact top-100 target the search failed to return was classified by whether the search ever
computed its distance:

| Build | Arrival group | Returned | Reached but not returned | **Never reached** |
| --- | --- | --- | --- | --- |
| As arrived | 1 (oldest) | 95.5% | 0.0% | 4.5% |
| As arrived | 2 | 93.4% | 0.0% | 6.6% |
| As arrived | 3 (newest) | 93.1% | 0.0% | 6.9% |
| + fill free slots | 1 | 96.2% | 0.0% | 3.8% |
| + fill free slots | 3 | 94.9% | 0.01% | 5.1% |

At this budget, **no missed target was ever scored**: the search never expanded any node that links
to it, and every scored exact target was returned. This diagnoses the traversal at ef 200. It does
not show that the target is unreachable in the graph, and it does not rule out routing fixes (a
wider beam, another entry point) on the same graph. Separating "a directed path exists", "a
predecessor was expanded", "the target was scored", "admitted to the candidate list" and "returned"
needs reachability checks and replays on the unchanged graph (section 9).

Properties of each node's links, by tenth of insertion order (as arrived):

| Tenth | 1st | 3rd | 5th | 7th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- |
| In-degree | 28.6 | 23.3 | 17.2 | 16.7 | 14.2 | 13.3 |
| Outgoing reciprocity (share of the node's out-links whose reverse link exists) | 0.80 | 0.82 | 0.86 | 0.90 | 0.95 | 0.97 |
| Similarity of in-link sources to the node | 0.52 | 0.54 | 0.54 | 0.57 | 0.58 | 0.58 |
| Mean insertion position of in-link sources | 0.32 | 0.42 | 0.43 | 0.49 | 0.52 | 0.57 |

**Correction (earlier draft).** The reciprocity row is the share of a node's *outgoing* links that
are mutual, not the share of its incoming links. The two share a numerator (mutual links) and have
different denominators (out-degree vs in-degree), so the earlier statement that 97% of a late node's
*incoming* links are reverses of its own choices does not follow from this table. A mutual edge also
records nothing about which direction was created first. The next run reports mutual count, in-degree
and out-degree per node, both fractions, and edge-creation events.

What the table does show: late nodes' out-links are almost all reciprocated, and their in-link
sources are more similar to them (shorter links) and later in the order than those of early nodes.

**Working explanation, to be tested by intervention.** A late node may lack incoming links from the
established nodes around it, which are the nodes a search expands when it reaches that region. A
descriptive test (how many of each target's 10 exact neighbours link to it, and whether the arrival
gap shrinks within strata of that count) is prepared. Because in-coverage is itself a consequence of
construction (a mediator), a gap that disappears after conditioning on it would not be enough to
choose a design. The deciding test changes edges directly: give targets the same number of incoming
links from different donors (uniform, geometrically closest, or nodes on development query paths)
and measure rescues on queries not used to choose donors (section 9).

### 4.6b Where the misses sit in the build: redistribution, and a lifelong gradient

Full snapshot 3 (1,661,900 documents), M 16, ef 200, exact top-100, mean of three seeds. Raw miss
rates by arrival cohort and overall recall, per build order:

| Engine | Order | Recall@100 | Miss, snapshot 1 | Miss, snapshot 2 | Miss, snapshot 3 |
| --- | --- | --- | --- | --- | --- |
| Epzero | as arrived | 0.9352 | 5.17% | 8.36% | 9.41% |
| Epzero | shuffled | 0.9338 | 6.09% | 7.53% | 7.62% |
| Epzero | reversed blocks | 0.9308 | 6.72% | 7.52% | 6.92% |
| FAISS | as arrived | 0.9359 | 5.26% | 8.21% | 8.77% |
| FAISS | shuffled | 0.9353 | 5.85% | 7.55% | 7.58% |
| FAISS | reversed blocks | 0.9317 | 6.69% | 7.25% | 6.83% |

**The build order mostly redistributes misses; it barely changes the mean.** Overall recall differs
by at most 0.44 points between orders, while the newest cohort's miss rate moves by 2.5 points and
the oldest's by 1.6 in the opposite direction. Helping the newest cohort can shift losses onto older
documents, which is why the success criteria check every cohort. (Raw rates also carry content: even
shuffled, snapshot 3's raw rate exceeds snapshot 1's; the within-query ratios adjust for that.)

Miss rate by **physical insertion tenth** (Epzero; FAISS is the same within about 0.5 points):

| Order | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| As arrived | 3.6% | 5.5 | 5.5 | 6.8 | 5.9 | 7.3 | 7.5 | 8.6 | 8.8 | 10.0% |
| Shuffled | 5.0% | 5.6 | 6.2 | 6.6 | 6.6 | 6.9 | 7.4 | 6.9 | 7.4 | 7.8% |

In the shuffled build, where position is independent of content, the miss rate rises steadily
across **the whole build**: a document inserted halfway through, with half the collection arriving
after it, is still missed more often (6.6%) than one from the first tenth (5.0%). The disadvantage
fades only slowly as a document ages; it is not confined to the newest batch.

**Consequence for an exact "fresh buffer" (design D).** Protecting the newest tenth by exact scan
would rescue only about 12% of all misses (11.6–12.1% as arrived), because the targets and misses
are spread across the whole build. On this collection a tenth is 166,190 vectors of 2,560 dimensions,
which costs far more per query than the whole ef-200 graph search (2.3–2.7 ms). A realistic buffer of
about 1% would rescue on the order of 1% of misses. A buffer protects the newest window; it does not
address a disadvantage that lasts most of a document's life. This is one final-snapshot view, so a
replay with checkpoints (section 9, durability) is still needed to measure how quickly it fades.

### 4.7 The IVF contrast

IVF with centroids trained on snapshot 1 and kept frozen shows no arrival difference among relevant
top-10 pairs (pooled 0.99, within-query 0.87). Among exact top-100 neighbours (snapshots 2 and 3
against 1, held-out arms), it shows a small one:

| Arm | Within-query ratio | Excess miss risk |
| --- | --- | --- |
| Epzero HNSW, ef 200 | 1.33 [1.21, 1.47] | +2.13 pp [1.46, 2.79] |
| FAISS HNSW, ef 200 | 1.25 [1.14, 1.38] | +1.65 pp [0.99, 2.31] |
| IVF, frozen training | 1.10 [1.01, 1.20] | +0.44 pp [0.04, 0.85] |
| IVF, refreshed training | 1.08 [0.99, 1.19] | +0.36 pp [−0.04, 0.77] |

The endpoints and groupings differ from section 4.1, so the rows are not interchangeable. IVF's
excess is a fraction of HNSW's, consistent with the literature (HAKES, Ada-IVF, MicroNN: frozen IVF
degrades only slightly under organic growth). With a fixed router and complete list scans, IVF's
results do not depend on *physical* insertion order at all, so its small excess must come from the
new documents' *content* (where they land relative to the router). A direct test: permute
insertions into the identical trained IVF-Flat index with fixed probes and stable IDs; the results
should not change. A family comparison should use a direct, paired HNSW-versus-IVF contrast.

### 4.8 What has not been measured yet

- How much of HNSW in-degree is position and how much is geometry (exact k-occurrence) — directly
  relevant to the SIGMOD 2026 hubness paper.
- Whether the "timing minority" hypothesis holds in both directions (section 7.3).
- IVF and quantizers under *induced* drift; global-range 8-bit quantization clipping of late arrivals.
- Calibration staleness by cohort (ef calibrated on snapshot 1, held fixed).
- A second real-timestamp collection, a second encoder (BGE-base-en-v1.5 is planned), more seeds.
- Other graph builders: hnswlib, Lucene (different reverse-link rule), Vamana/FreshVamana, NSG.
- The Big-ANN streaming runbooks re-analysed by item age.

---

## 5. Literature review

The review (23 November 2026) covered five areas in parallel, with every claim checked against the
source where it could be opened. Peer-reviewed (PR) and preprint (PP) status is marked. Full notes and
URLs are in `research_notes/ANN index staleness bias/`.

### 5.1 Graph indexes: insertion order, in-degree, reachability

| Work | Status | What it shows | What it does not do |
| --- | --- | --- | --- |
| Malkov & Yashunin, HNSW (TPAMI 2018) | PR | The algorithm, including the conditional reverse-link update | No analysis of in-degree by insertion time |
| Elliott & Clark, "The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in HNSW" (ICTIR 2024, arXiv 2405.17813) | PR | Whole-index recall moves up to 12.8 points when insertion order is sorted by LID or category | Order is *correlated with the data*; no per-item or early-vs-late analysis, no in-degree |
| GRAB-ANNS (arXiv 2604.16402, Mar 2026) | PP | Names the mechanism ("isolation of newcomers": incumbents shield new nodes from the diversity test); fix multiplies the distance by α < 1 for fresh batch members during reverse rewiring | GPU only; measured only by average recall on SIFT filtered search; no measurement by age; no numbers in the text for the ablation |
| FlatNav, "Down with the Hierarchy" (arXiv 2412.01940) | PP | Regresses node *access counts* on insertion order: explains ≤ 0.5% of variance on real data | Never measures in-degree or recall by age; direction of effect not reported |
| Xu et al., "Through the Lens of Hubness" (SIGMOD 2026, PACMMOD 4(3) art. 243, doi 10.1145/3802120) | PR | See 5.2 | See 5.2 |
| MN-RU, "Enhancing HNSW Index for Real-Time Updates" (arXiv 2407.07871) | PP | 3–4% of points become unreachable under repeated delete/reinsert; repairs one-hop neighbours | Deletion-driven; calls unreachability during plain construction very unlikely |
| FreshDiskANN (arXiv 2105.09613), IP-DiskANN (arXiv 2502.13826) | PR / PP | Streaming graph indexes with delete consolidation or in-place deletes; stable *average* recall over long update sequences | No per-age analysis |
| PVLDB 2025 construction revisit (arXiv 2410.01231) | PR | Incremental insertion caps candidate quality; refinement improves connectivity | No in-degree-by-order analysis |
| Library evidence: FAISS #5968/#5969 (Nov 2026), VSAG #2653, hnswlib PR #674 (shuffle by default), Lucene #15427 (proposes plotting neighbour quality against insertion time), Chroma #7963 and #7998 | Practitioner | Duplicate or tied vectors end with in-degree 0 because incumbents win the reverse-link re-prune (FAISS #5968); shuffling helps corpus-ordered data; Chroma #7963 shows *early* small documents lost reachability after a large later batch | Unreviewed, partly confounded by other bugs; none measures by age |

Implementation detail that matters: hnswlib and FAISS re-run the full heuristic when a neighbour is
full (the newcomer can be rejected). Lucene appends the candidate to the neighbour's array and then
runs its diversity check to choose one entry of the enlarged, sorted array to remove, which can be
the newly appended candidate itself, so Lucene does not guarantee the back-link either. The rules
still differ, so measured reverse-link acceptance by distance and list occupancy is a useful
cross-engine comparison (cite the exact Lucene revision and check with a full-list unit example).

Further work identified by an independent review (23 November 2026):

| Work | Status | Relevance |
| --- | --- | --- |
| Tian Pan, "The vector index that was sharded by ingestion date" (blog, 2 June 2026) | Practitioner essay | Recommends building query sets "by stratifying on document age" and reporting recall "by document age". The architecture is date-sharded with a per-shard top-k and merge, where *old* documents lose; all numbers are illustrative. So "average recall can hide age disparity" is a prior public proposal; our contribution is the measurement inside one growing graph |
| Jaesol Shin, HNSW incremental insertions (blog, 28 and 30 October 2026) | Practitioner experiments | pgvector and Qdrant, incremental vs fresh builds over the same final vectors (10K+10K; 1M+1M MS MARCO), random vs clustered order, per-query recall. Incremental minus fresh within about ±0.75 pp. Does not split recall by insertion time or mention in-degree |
| Slipstream (arXiv 2606.02992, June 2026) | PP | Reuses the previous insertion's candidates when arrivals have temporal locality; up to 30.8× insertion throughput at recall@10 ≥ 0.95. Relevant to bursty streams and insert routing |
| LSM-VEC (arXiv 2505.17152) | PP | Disk-based dynamic vector search with LSM-style updates |
| Zhao, Wang & Ngo, approximate k-NN graph construction (IEEE TMM, doi 10.1109/TMM.2021.3073811) | PR (not yet read by us) | Online graph construction with neighbourhood search and reverse-neighbour information predates local repair; our repair must state precisely what differs |
| "As We May Search" (Zerhoudi et al., ICTIR 2026, arXiv 2606.29652) | PR | Local-first IR; HNSW extends dense retrieval to 1M documents at about 2% quality loss: an aggregate ANN cost, the kind of mean that can coexist with a cohort effect |
| "ANN Search: Recall What Matters" (arXiv 2606.04522) | PP (not yet read) | Argues recall@k is an incomplete measure of ANN quality for RAG |
| Two-stage fairness in recommendation (doi 10.1145/3539597.3570469) | PR (not yet read) | Models fairness at candidate generation; relevant to "index-induced loss" framing |

### 5.2 The SIGMOD 2026 hubness paper in detail

This was the main novelty risk; it has been read in full.

- **Claim**: in-degree skew in proximity graphs is caused by *hubness*, an intrinsic property of
  high-dimensional data. Points near the centre of the distribution are closer to everything
  (spatial centrality, after Radovanović et al. 2010); skew grows with local intrinsic
  dimensionality. Theorems: hubs capture greedy search paths (Thm 1); anti-hubs become unreachable as
  skew grows (Thm 2).
- **Outlier queries**: the slowest half of queries on a brute-force kNN graph have ground-truth
  neighbours with low in-degree (Table 6). The paper adds that *reachability*, not raw in-degree,
  is what matters.
- **Taxonomy**: HNSW and NSG (diversification), Vamana and τ-MG (relaxed pruning), LSG and Vista
  (data-driven adaptivity) are read as hubness-mitigation strategies.
- **Fix**: a post-hoc *hubness-aware pruning* with a factor γ that relaxes the pruning rule for
  low-in-degree nodes and tightens it for hubs. It lowers in-degree skewness at +0.2% indexing time
  for HNSW; the authors note it cannot reach severely isolated nodes that never appear in candidate
  pools.
- **Open problem they name** (Sec. 6.2.3): "naive insertions and deletions can introduce new degree
  imbalances, break navigational paths, or isolate previously reachable nodes. Developing
  bounded-cost update mechanisms that preserve degree regularity and global reachability—without
  triggering full index rebuilds—remains an important open problem."
- **What it does not do**: it never considers insertion order, position or age. All builds are static
  and use 128 indexing threads (order scrambled). Its unit is the query, not the document.

**Consequences for us.** (1) Our shuffled-build gradient shows part of HNSW's in-degree variance is
positional, not geometric — some "anti-hubs" in an incrementally built HNSW are just late arrivals.
Decomposing in-degree and miss probability into geometry (exact k-occurrence) and position is a
direct, citable test of their framework. (2) Their γ rule is peer-reviewed prior art for
in-degree-aware pruning and must be a baseline. (3) Their "reachability, not raw in-degree" fits our
residual after in-degree repair. (4) Their open problem is our engine target.

### 5.3 Partition (IVF) and quantizer staleness

| Work | Status | Finding |
| --- | --- | --- |
| DeDrift, "Robust Similarity Search under Content Drift" (ICCV 2023) | PR | Real temporal drift (YFCC 2007–2013; VideoAds 2020–2022). Stale coarse quantizer costs up to 11 points of 10-recall@10 at a tight budget, ~1 point at a generous one; PQ/OPQ codecs lose ≤ 0.011. Sliding-window protocol, never old and new in one index |
| HAKES (PVLDB 2025) | PR | "As more data are added the recall degrades slightly": IVF is robust to moderate drift from the same model |
| Ada-IVF (arXiv 2411.00970), Quake (OSDI 2025), SPFresh/LIRE (SOSP 2023), UBIS, MicroNN | PR / PP | Staleness mostly a speed problem (e.g. frozen IVF at 0.27× the QPS of a rebuild at equal recall); recall loss appears when nprobe is fixed under drift |
| PQR-Search (arXiv 2611.07579, Nov 2026, submitted to ICDE 2027) | PP | Per-query recall guarantees for dynamic vector databases; shares of queries below 0.99 recall rise under updates (IVF-Flat 13.5% → 22.3%, hnswlib 11.1% → 16.7%, FreshDiskANN 13.8% → 20.0%); "graph indexes degraded gradually even under random insertion alone"; its own method is IVF-based at 2.5× HNSW latency |
| CoDEQ (arXiv 2512.18335), IVF-TQ (arXiv 2605.17415) | PP | Claims of PQ degradation under (synthetic) drift; unreplicated |

No paper breaks recall down by item age for any partition or quantized index. Datasets with real
timestamps used in ANN work: DeDrift's YFCC and VideoAds, Quake's Wikipedia-12M trace. Search-width
auto-tuners (LAET, DARTH, CTR) assume test queries resemble calibration queries; none measures
calibration drift as a collection grows.

### 5.4 Benchmarks: none can see the effect

| Benchmark | What it reports | Per-item or per-age? |
| --- | --- | --- |
| Big-ANN NeurIPS'23 streaming track (arXiv 2409.17424; NeurIPS 2025 D&B) | Recall@10 averaged over all queries and checkpoints of clustered insert/delete runbooks | No. But the repository ships ground truth at every checkpoint, so per-age recall can be computed afterwards. A caching bug, found months later, reversed the ranking (one entry fell from 0.985 to 0.092) — an illustration of how much a single number hides |
| VIBE (arXiv 2505.17810; DMLR 2026) | Average recall vs QPS, hardest/easiest 100 queries; includes out-of-distribution query sets | No streaming at all |
| Intel SVS dynamic, Cosmos DB DiskANN streaming posts, VectorDBBench | Average recall per update step | No |
| CANDY (arXiv 2406.19651), CANDOR-Bench (SIGMOD 2026), deletion evaluation (arXiv 2512.06200) | Average recall over time | No |

What a fair age-invariance benchmark needs: live-set ground truth at each checkpoint plus each item's
insertion step; miss rate per true neighbour grouped by the neighbour's age, reported as a new/old
ratio with intervals; random, clustered and real-timestamp orders; insert-only growth and churn; a
from-scratch rebuild as control; matched-recall operating points; tail as well as average reporting.

### 5.5 Information retrieval: retrievability, exposure, temporal evaluation

- **Retrievability** (Azzopardi & Vinay, CIKM 2008): how retrievable each document is under a
  query population; measured with Gini/Lorenz. Follow-ups study dense retrievers (CIKM 2023
  synthetic-query estimation; arXiv 2303.11648; T-Retrievability, CIKM 2025; Valentini et al.,
  EMNLP 2026 Findings) — always the *model's* bias, never the index's. Samar et al. (IJDL 2018)
  measured retrievability by timestamp in web archives, unrelated to ANN.
- **Exposure fairness** (Singh & Joachims 2018; Diaz et al., CIKM 2020; TREC Fair Ranking
  2019–2021): defined on the ranking; nothing at the candidate-generation or index stage.
- **ANN and effectiveness**: Lin (ACL 2025 Industry) — HNSW slightly below exact and varies between
  builds; Leto et al. (NeurIPS 2024 workshop) — ANN recall 0.7 costs only 2–3% gold-document
  recall; arXiv 2412.15246 — HNSW below exact on end-to-end RAG. All averages.
- **LongEval**: 2026 Sci overview (CEUR Vol-4283) — effectiveness falls as the collection grows,
  system rankings correlate only weakly across snapshots, recency "does not appear to be a driving
  factor of relevance". No LongEval paper studies ANN indexing.

**Gap**: no study holds the embedding model fixed and compares exact against approximate search per
document, and none groups documents by arrival in a growing index.

### 5.6 Production systems

Most products use an LSM-style design: new writes go to a small buffer that is brute-force scanned
(Pinecone memtable, Milvus growing segments, Qdrant appendable segments, Turbopuffer's log, LanceDB's
unindexed tail, Chroma's 100-vector buffer), then into indexed segments built or merged in the
background. This hides arrival bias only for the newest window. Systems inserting straight into one
graph — pgvector HNSW, Vespa, Redis, Cosmos DB DiskANN, Weaviate after its async queue — are exposed
directly, and Lucene's merges turn recent documents into late insertions. Vendors document and
handle deletion damage and IVF staleness; none documents or measures recall for new versus old items.
Pinecone states that "very few of our use cases are primarily static corpora".

---

## 6. Current and latest competing work (timing)

| Date | Work | Distance from our claim |
| --- | --- | --- |
| Mar 2026 | GRAB-ANNS (preprint, GPU) | Names the mechanism and has a freshness-biased pruning rule; could add age analysis in a revision |
| Jun 2026 | Hubness paper (SIGMOD 2026) | In-degree skew and post-hoc in-degree-aware pruning; attributes everything to geometry; names rebuild-free maintenance as open |
| Aug 2026 | hnswlib PR #674 (shuffle by default) | Fixes corpus-order effects only |
| Nov 2025 – Nov 2026 | Lucene #15427, #15504 | Proposes measuring neighbour quality by insertion time; no data |
| 5–11 Nov 2026 | PQR-Search (preprint, ICDE 2027 submission) | Per-query recall failures under updates; one grouping step from a per-age plot |
| 21 Nov 2026 | FAISS #5968/#5969 | Exact-duplicate case of our mechanism (in-degree 0 for tied newcomers) |
| Oct–Nov 2026 | Chroma #7963, #7998 | Practitioner reports of reachability loss by insertion history |
| 2 Jun 2026 | Tian Pan blog (date-sharded indexes) | Already recommends recall stratified by document age (illustrative numbers, different architecture) |
| 2 Jun 2026 | Slipstream (preprint) | Streaming insertion with temporal locality |
| 28–30 Oct 2026 | Jaesol Shin blogs (pgvector, Qdrant) | Incremental vs fresh builds at 1M+1M, per-query recall; no age split |

Several groups are within one step of our measurement. **Timing risk is high.** A short preprint of
the measurement and its shuffle control would establish priority early; whether that fits the CLEF
2027 rules and timeline is a decision to make (the CLEF 2027 dates still have to be verified).

---

## 7. Novelty assessment and the refined claim

### 7.1 Already known

- Recall depends on insertion order when the order is correlated with the data (Elliott & Clark).
- Newcomer isolation, named qualitatively (GRAB-ANNS).
- In-degree skew from data geometry, and in-degree-aware post-hoc pruning (hubness paper).
- Unreachable points from deletion churn (MN-RU and others).
- IVF centroids and fixed search widths go stale under drift, mostly as cost (DeDrift, Quake, Ada-IVF).
- Average recall hides per-query failures (PQR-Search, VIBE's hard-query split).

### 7.2 New (among the sources checked to 23 November 2026)

Also known, and not ours to claim: that average recall can hide document-age disparities (proposed
publicly by Tian Pan in June 2026 for date-sharded indexes); that insertion paths change per-query
recall (Shin, October 2026); that newcomers can be under-connected (GRAB-ANNS).

1. **The controlled reversal (strongest).** Holding the final corpus, encoder, queries and exact
   targets fixed, moving the same real arrival cohorts to the other end of an incremental HNSW build
   reverses which cohort loses more exact neighbours, within queries and rank bands, in two engines,
   with absolute changes of 3.4–4.3 percentage points, and the pattern appears with randomly assigned
   cohorts on public data.
2. The judgement-free measurement on a real, insert-only growing collection, beside the frozen
   finding that mean effectiveness barely changes.
3. A relevance-level signal (relevant new pairs dropped 2–3× as often, pooled), still imprecise
   within queries and not yet translated into ranked gain.
4. Diagnostics: at the study's budget, misses are traversal failures (never scored); existing
   repairs narrow but do not close the in-degree gap and leave a residual.
5. The contrast with frozen IVF on the same data (much smaller excess).

Suggested novelty sentence (from the review): *We quantify how moving the same arrival cohorts
through the construction order of two incremental HNSW implementations changes the probability of
losing exact neighbours on a growing scientific retrieval collection. We report the absolute cost by
cohort, test the effect under randomly assigned arrival groups, and distinguish it from changes in
average judged effectiveness.* A mechanism or remedy joins that sentence only once its intervention
has survived an independent workload.

### 7.3 The claim, stated carefully

> **In incremental HNSW, a cohort's physical position in the build order changes how often its exact
> neighbours are lost.** Holding the documents, queries and exact targets fixed, moving the newest
> cohort from the end to the start of the build moves its excess miss risk from about +2–3 to about
> −1.5 percentage points, in two implementations, and the same happens with randomly assigned
> cohorts. A mean over queries, such as average recall or mean nDCG, cannot show how errors are
> distributed across cohorts. Whether the fix belongs in linking, in routing, or in a different index
> architecture is an open question that the interventions in section 9 decide.

The broader statements (late arrivals among established neighbourhoods are the most disadvantaged;
partition and quantizer indexes depend on history only through drift; calibrated widths through
change since calibration) remain hypotheses.

**Working hypotheses about the mechanism** (to test, not to claim):

- **Entrenchment**: incumbents' lists were chosen before the newcomer existed and the pruning rule
  protects them.
- **Reachability over in-degree**: what matters is whether a node's incoming links come from parts
  of the graph that searches actually pass through (well-connected, older nodes) or only from other
  recent, weakly reachable nodes.
- **Timing minority**: an item is disadvantaged when most of its true neighbours arrived at a
  different time than it did — in *either* direction. This would explain both our late-arrival bias
  and Chroma #7963 (early small clusters lost after a large later batch). If it holds, the result
  generalises from "new documents lose" to a structural law of incremental graphs.
- **Position vs geometry**: part of what the hubness literature calls anti-hubs in incremental graphs
  is position.

### 7.4 Is it common to all index families?

Not as an *effect*: our frozen IVF shows none on LongEval, and the literature agrees that partitions
and codebooks go stale only under real drift. As a *structure* it is: every family's retrieval
depends on history, through a different variable. What would show this properly is one protocol
crossed over three factors — insertion order (as arrived / shuffled / reversed) × drift regime
(i.i.d. split, organic LongEval growth, induced topic drift) × index family, each against a
from-scratch rebuild:

| Family | History variable | Expected without drift | Expected under drift | Evidence |
| --- | --- | --- | --- | --- |
| Incremental graph (HNSW, FreshVamana) | Insertion position | Bias (ours) | Larger? | Ours |
| Batch-built graph (two-pass Vamana, NSG, CAGRA) | Build sample only | None predicted | None if rebuilt | Untested |
| IVF, frozen centroids | Training sample | None (ours, HAKES) | Cost first, misses at low nprobe | Partly published |
| PQ / OPQ | Training sample | None | Small | Published |
| Global-range 8-bit (SQ8) | Range sample | None | Clipping of late out-of-range items | Unmeasured |
| Calibrated ef / nprobe | Calibration set | Degrades even under random insertion (PQR) | Tail grows | Aggregate only |

---

## 8. How we plan to fix it in Epzero

The engine may be redesigned at its core if that is what solving the problem takes. Every change is
kept only if measured gains justify it.

### 8.1 Goals and success criteria (to be fixed in writing before the confirmatory runs)

1. **Discoverability of every cohort, in absolute terms.** The primary gate is absolute: the excess
   miss risk of the newest cohort, and the build-to-build order contrast, each within a pre-specified
   margin (for discussion: at most one extra missed item per 100 eligible pairs), *and* no cohort,
   including the oldest, worse than the baseline. A ratio of 1 can be reached by making old
   documents worse; that must count as failure. The draft's ratio range [0.95, 1.05] is a secondary
   ambition until event counts support it; power is simulated from development data first.
2. **Beat the simple alternatives at matched work.** Searching wider already shrinks the absolute
   excess (to +0.80 pp at ef 800, at about 3.5× the query time). A new design must beat that, a
   second entry point, and the exact-buffer and fixed-router baselines at equal measured query work,
   and then at matched held-out recall.
3. **No loss on standard metrics**: recall and queries per second at matched recall no worse than the
   current engine (VIBE-style curves).
4. **A stated resource envelope**: insertion cost defined precisely (CPU time per upsert, latency, or
   sustainable throughput at a stated query load; target ≤ 1.3×), a peak-memory cap including
   overlapping versions, a p99 query budget, all on the same hardware; bytes per vector also checked
   on a compressed embedding, where a 4-byte counter is proportionally large.
5. **Serving correctness**: identical results to exact live-set search after insert, upsert, delete,
   flush, merge, concurrent queries and recovery; ties, duplicates, repeated labels, empty cohorts.
6. **Observability from real queries**: per-age miss estimates in `health()` from sampled real queries
   against independently computed exact references, with denominators and intervals. Self-queries
   only catch gross connectivity defects. Training, monitoring and evaluation queries stay disjoint.
7. **A decision rule that permits a negative result**: if a simpler configuration or a partition
   design meets the gates more cheaply, that is the answer.

### 8.2 Candidate designs

These are ordered from smallest change to largest. The screening on the 300K sample decides which go
forward, with development queries for choosing and unseen queries or a separate workload for the
final reading.

**0. Same graph, different search policy (first baseline).** Wider search, a second entry point, or a
matched-work restart on the unchanged graph. If this meets the gates at negligible ingestion cost,
ship the configuration and report the larger redesign as unnecessary for that regime.

**A. Arrival-aware reverse linking.** When a full neighbour would reject the newcomer, evict instead
the existing link that is most *redundant* — for example, one whose target has high in-degree or is
reachable through another kept neighbour — and guarantee the newcomer a minimum number of incoming
links. Needs a per-node in-degree counter (4 bytes per node, small next to a 10 KB vector). Differs
from GRAB-ANNS (fixed α for batch members) and the hubness γ rule (post-hoc, in-degree only) by
acting during ingestion and choosing the evicted link by redundancy.

**B. Hindsight re-linking (deferred local repair).** Each node keeps a small buffer of recent arrivals
that chose it as a forward neighbour but were rejected. When a node has accumulated enough of them,
it re-selects its links over its current list plus those arrivals — the list it would have chosen
had it known about them. This approximates order-independent construction at amortised cost of a
few selections per insertion, instead of a full search per node as in global refine.

**C. Local in-coverage guarantee.** The miss anatomy (4.6) points here: a newcomer must be linked
from the established nodes *around* it — the ones a search expands when it reaches that region — not
only from the neighbours it chose itself. At insertion, the construction search already visits those
nodes (it returns efConstruction candidates); C adds the newcomer to the lists of the closest few of
them that do not yet link to it, choosing which existing link to give up by redundancy as in A. If
the coverage test confirms that holding local in-coverage fixed removes the arrival gap, C (with A's
eviction rule) becomes the lead design. A variant also gives one link from a well-connected node on
the search path ("gateway"), in case long-range entry turns out to matter.

**D. Fresh buffer plus sealed segments (architectural).** New vectors go to a small exact-scanned
buffer, so the newest items are never missed; batches are merged into the main graph using B (or
built with an order-insensitive two-pass construction). Removes the bias for the newest window by
design; the cost is a second, small search per query.

**E. Fixed routing with independent partitions (order-invariant reference).** A frozen trained coarse
router, deterministic assignment, fixed probes, complete scans of probed lists and deterministic
tie-breaking. At a fixed live set and router, candidate membership is decided by the probed lists,
so physical arrival order drops out entirely; the reviewer's synthetic check found IVF-Flat returned
identical top-50 sets under three insertion orders. Semantic drift can still put recent vectors in
hard partitions, and scan caps, early stopping, retraining or graphs inside partitions reintroduce
history. If it meets the latency and memory budget with equal or better worst-cohort recall, it is a
legitimate answer. A hybrid (stable router, graphs inside large partitions, exact delta, bounded
per-partition rebuilds published atomically) is the next step if exhaustive partitions are too slow.

**F. Per-age recall telemetry and self-repair.** Estimate misses by age from sampled real queries
against independent exact references and expose them, with sample sizes and intervals, through the
existing `health()` report; trigger local repair (B) when the absolute excess crosses a threshold.
No engine reports this today. A cheap per-batch audit (from the second review): for a sampled query
with graph answer A and a batch B, rescore A ∪ B exactly and count the members of B in that union's
top k that A missed (U). U is an upper bound on the true misses from B on that query: any global
top-k member of B is in the union and can only rank higher there. It costs one scan of B, so auditing
many batches often can cost more than it saves. In the reviewer's synthetic screen the bound held
everywhere but was loose (1,262 counted against 790 true misses) and ranking batches by it did worse
than protecting the newest (FIFO); we reproduced those numbers exactly. Useful as an alarm, not yet as
a policy.

For D, a delta item is safe only while it is exhaustively scored, enough local candidates are
taken, scores and ties agree, and the final merge preserves it; once sealed it inherits the segment's
errors. Probe immediately before and after sealing and compaction, and count query fan-out, dual
versions, rebuild bytes and backlog.

### 8.2b Development screens so far (23 November 2026)

Each design and prediction was committed before the run; the full records are `docs/screen-d1-*.md`
to `docs/screen-d5-*.md`.

| Screen | What | Result |
| --- | --- | --- |
| D1 | Post-hoc: +4 incoming links to every newest-cohort document, donors closest / random / best-connected (FAISS) | Excess at ef 200 1.36 → 0.35–0.42 pp; donor choice made no difference (prediction against); oldest cohort unharmed |
| D2 | Online floor `min_backlinks` 2 / 4 / 8 (Epzero) | No effect (prediction against): new nodes already get ~12.5 incoming links at insertion; the gradient is **passive accumulation** (section 4.4). Option removed |
| D3 | Online `extra_backlinks` 4 / 8: the closest construction candidates a new node did not choose link to it | Excess 1.62 → 0.78 / 0.41 pp; as arrived minus reversed halved; oldest +0.08 / +0.16 pp; recall +0.18 / +0.25 pp; work +0.4 / +0.9%; build +2 / +4% |
| D4 | `extra_backlinks` 8 on SIFT, GloVe and two more LongEval seeds | Excess cut 67–87% everywhere; oldest +0.05 to +0.21 pp; recall +0.25 to +0.75 pp; work +0.6% to +5.2% (SIFT just above the predicted 5%) |
| D5 | Recall against work over the whole width range | Not a uniform speed-up: 1–5% more work at low recall, 6–21% less at recall@10 ≥ 0.98; neutral to slightly positive overall |
| D6 | Clean sequential timing, 17 widths, `extra_backlinks` 0 / 4 / 8 | With 4: single-thread throughput −0.1% to +24% across recall@10 0.90–0.99 on all three datasets (+24% at 0.98 on LongEval, +13% at 0.99 on SIFT); build +0–5%. With 8: smaller throughput gain, −2.3% in one cell. **4 is the candidate default** |

`extra_backlinks` (design A in its online form: a stand-in for the later arrivals a new node has not
had) is the lead candidate, with 4 as the candidate default. It is on the Epzero branch
`min-backlinks`, off by default. Before any claim: three seeds per arm, timing on a PACE node, the
compressed (SQ8) configuration, then the confirmation readings (BGE vectors, a public streaming
workload) and the margins of decision 3.

### 8.3 Baselines every design must beat

Plain HNSW (Epzero, FAISS, hnswlib) at several search widths; a second entry point; the GRAB-ANNS
freshness α rule reproduced in its own distance convention (an Epzero port is labelled an
adaptation); the hubness paper's γ rule (from its description; their released artifact can be
inspected for details); Lucene-style append-then-select back-linking; Vamana α > 1 and FreshVamana;
Epzero's global refine and free-slot filling; a full rebuild (an offline reference, since a
whole-index shuffle uses future data); an exact-scan buffer alone; the fixed-router partition index
(E). All at equal memory and edge budgets, first at equal measured query work, then at matched
held-out recall. Every comparison also reports harm to the oldest cohort and to graph-wide recall.

### 8.4 Where Epzero stands on standard benchmarks today

Honesty matters here: winning every public leaderboard is a goal, not a given. On a Gold 6226,
Epzero with AVX2 kernels is 1.08–1.41× Glass on Yandex at recall 0.90–0.98, but 0.86–0.88× on
ImageNet-CLIP at 0.90, where its per-step cost is 17–20% higher. Round 10 (kernels on more datasets,
the speed sweep on ImageNet) and the VIBE submission address that gap separately. The arrival-bias
work must not cost speed; ideally the better-linked graph also helps average recall (the free-slot
pass raised recall@100 at ef 200 from 0.946 to 0.965).

---

## 9. Experimental programme

Each new experiment's design and prediction are written down before it runs. Jobs on PACE stay on
one CPU type (Gold 6226) and never overlap across CPU types. LongEval data never leaves the cluster's
or the local `build/` folders.

### Priority 0 (decide the claim and the mechanism)

| Experiment | Settles |
| --- | --- |
| **Measurement record first**: one aggregate row per corpus, snapshot, engine, order, seed, M, efConstruction, ef, contrast, cutoff and label family, with numerators, denominators, queries with both cohorts, distinct targets, tie policy, bootstrap unit, estimator name and provenance; "new" defined explicitly | Which numbers can be cited; removes ambiguity between raw, rank-standardized and within-query ratios |
| **Paired target-level analysis**: each (query, exact target) observed in both builds; counts found in both, missed in both, rescued, newly lost; paired risk difference by query and construction replicate; individual seed contrasts | How many losses actually move; prevents a ratio near 1 hiding cancellation |
| Miss anatomy (done, 4.6): at ef 200 no missed target was scored | Misses are traversal failures at that budget |
| Edge events: reverse-link proposals, acceptance, survival and evictions by age, distance and list occupancy; both reciprocity denominators | Passive opportunity vs rejection by the diversity heuristic |
| Reachability and routing on the unchanged graph: directed reachability from entries for sampled targets; replay with a second entry and a wider beam at measured work | Whether the fix belongs in routing or in connectivity |
| **Equal-degree donor intervention**: on copies of one graph, give targets the same number of new incoming links from uniform, geometrically closest, or development-query-path donors; match donor caps and displaced edges; score rescues on unused queries, plus damage to old targets | Whether *who* links matters beyond *how many* (design C vs A) |
| Local in-coverage (descriptive, prepared): arrival gap within strata of how many of a target's 10 exact neighbours link to it | Describes the pattern; cannot by itself choose a design (mediator) |
| Position across many permutations, with graph levels from a stable ID hash in a separate arm | Within-target position effect; contribution of level assignment |
| Timing-minority schedules: early minority, late minority and balanced arrivals inside the same cluster, plus bursty and alternating streams | Whether both minorities lose (an interaction) or only late arrivals |
| **Durability**: replay the stream, query the live set after fixed numbers of writes and across flush and merge boundaries; follow the same targets as they age | How long a disadvantage lasts: minutes, one batch, or a lifetime |
| IVF order control: permute insertions into the identical trained IVF-Flat index (fixed probes, stable IDs, ties) | IVF's small excess is content, not order |
| Position vs geometry: regress HNSW in-degree and per-item miss probability on exact k-occurrence (geometry), insertion position, local density/LID, upper-layer membership, hop distance from the entry point, and the timing-minority share of the exact neighbourhood | How much of "hubness" in incremental graphs is position; tests the timing-minority hypothesis in both directions |
| ≥ 5 construction seeds, ef ladder, M ∈ {8, 16, 32, 48}, absolute differences alongside ratios | Non-determinism; dose-response; why M 32 is worse than M 16 |
| Drift-free control: i.i.d. synthetic data and a random split of LongEval inserted in fixed groups | That the graph effect needs no drift |
| Cross-implementation: hnswlib, FAISS, Lucene (milder rule predicts a weaker gradient), FreshVamana, batch-built Vamana/NSG | Common to incremental graphs, or an artefact of one pruning rule |
| Big-ANN streaming runbooks re-analysed by item age (ground truth already exists) | The blind spot on the community's own benchmark |

### Priority 1 (generality and the IR case)

| Experiment | Settles |
| --- | --- |
| Second real-timestamp collection (arXiv by submission date, Wikipedia trace if released, or YFCC) and the BGE-base-en-v1.5 replication | Beyond one collection and one encoder |
| IVF and quantizers under induced topic drift: nprobe ladder, PQ, RaBitQ, global-range SQ8 clipping by cohort | Whether any non-graph family shows an age effect, and when |
| Calibration staleness: ef/nprobe calibrated on snapshot 1 and held fixed, recall by cohort at snapshots 2–3 | The calibration leg of the general claim |
| Relevance-level power: all label types, judged@10 by cohort (pool bias against new documents), the full query pool | The weak within-query relevance result and label-bias objections |
| Three consequences on the same queries: exact-neighbour loss; loss of judged-relevant exact candidates; net ranked gain after replacements; plus recall of relevant candidates at the real candidate depth and final nDCG@10 after a fixed reranker | How far the claim reaches from neighbours to effectiveness |
| Blind, crossed human audit of losses and replacements with a second assessor on an overlap (the existing snapshot-3 overlap, 1,581 pairs on 29 queries, has GPT-OSS-120B linear-weighted kappa 0.24 under different topics and scales) | Human relevance of what is lost |
| New queries or an independent timestamped text collection, pre-specified contrast and margin, held for the final reading | Confirmation on data not used to explore |
| Retrievability by cohort with synthetic queries (Gini/Lorenz under exact vs ANN); self-retrieval at k = 1 per item | Connects to the IR retrievability literature at scale |

### Priority 2 (the fix)

| Experiment | Settles |
| --- | --- |
| Remedy screen on the 300K sample: designs A–E against the baselines of 8.3, scored on the within-query ratio, recall, QPS, build cost, memory | Which design goes forward |
| Pre-registered confirmatory run: improved Epzero as a new arm on the full LongEval snapshots (and the held-out design), plus public data and the Big-ANN runbooks | The headline engine result |
| VIBE and standard recall/QPS curves for the improved engine | No regression on standard benchmarks |

---

## 10. Publication plan

**CLEF 2027 long paper: "The index is a hidden temporal factor."** As collections grow, dense
first-stage retrieval with an incrementally built graph index under-exposes late-arriving documents
while every metric LongEval and the ANN community report stays flat. Contributions: the arrival
miss-ratio metric (neighbour and relevance level, with retrievability by cohort); evidence on the
three LongEval-Sci snapshots plus a second collection; causal controls (shuffle, reverse, rebuild);
the IVF contrast; evaluation of remedies; a reporting recommendation for LongEval (report the index,
its build history, and the exact-vs-ANN cohort ratio for dense runs). The frozen study (D values and
sensitivity analyses) becomes the paper's evidence that average effectiveness is stable — which is
exactly why the per-document view is needed.

**Separate systems paper on the engine**: arrival-invariant linking plus per-age telemetry, judged on
public streaming workloads. Kept separate so the two do not double-publish.

### 10.1 Reviewer objections and our answers

| Objection | Answer |
| --- | --- |
| The relevance-level within-query ratio includes 1 (Epzero 1.28 [0.86, 1.96]) | Lead with the neighbour-level within-query ratio and its controls (far more pairs, no labels); add relevance-level power (P1) |
| The absolute effect is small (4.0% vs 2.0%) | Exposure argument; dose-response with growth, lower ef and more snapshots; ratio stable as ef rises |
| "Insertion order effects are known" | Elliott & Clark study data-correlated orders and whole-index recall; FlatNav studies access counts; we show a position effect per document with random and shuffled controls |
| "That is just hubness" | The hubness paper attributes skew to geometry and uses static, multi-threaded builds; our shuffled gradient and the position-vs-geometry decomposition separate the two |
| Labels are biased against new documents (pools from mostly lexical runs; click labels) | Report judged@10 by cohort and every label type; neighbour-level results need no labels |
| One collection, one encoder | Second collection and BGE replication (P1); public datasets already show it |
| HNSW builds vary | Multiple seeds; Epzero's in-degree gradient is identical across seeds |
| "Just rebuild" | Rebuilds are what streaming systems avoid; a global refine took 101 s wall time on 10 threads after a 282 s single-thread build (CPU time still to be measured) and still leaves a residual |
| "Just search wider" | It shrinks the absolute excess (+0.80 pp at ef 800) at about 3.5× query time; any proposed fix is compared against it at matched work |
| Why is M 32 worse? | Not yet explained: M and efConstruction changed together and the base rate is lower. Cross M 16/32 with efConstruction 100/200, compare at matched work and recall, report absolute differences |
| The within-query analysis was chosen after seeing the data | Say so; report the pre-specified rank-standardized contrast (+0.77, +0.62) beside it, and confirm on unseen queries or a new workload |
| Bootstrap intervals ignore build variability | They are conditional on the three builds; show per-seed contrasts and add build-level uncertainty before claiming anything about future builds |

---

## 11. Risks and open questions

- **Being scooped**: several groups are one step away (section 6).
- **Mechanism not fully pinned down**: misses are traversal failures at the study's budget, and the
  in-degree gradient could be passive opportunity rather than rejection; if the equal-degree donor
  intervention shows no effect of who links, the fix may be harder than designs A–C suggest; D (fresh buffer plus sealed segments) is the fallback
  that removes the newest-window bias by construction.
- **Cost of the fix**: a fix that slows insertion or search much would not be adopted; criteria in
  8.1.
- **Generality**: the cross-family claim may end up as "graphs, plus drift-conditional effects
  elsewhere" — still a clear, honest result.
- **Venue timing**: CLEF 2027 deadlines must be verified; a preprint decision is pending.

---

## 12. Other work in progress

- **Further construction seeds** for the held-out primary arms (seven seeds; package built, design and
  prediction recorded in `scripts/pace/LONGEVAL_HELDOUT_SEEDS.md`; not yet submitted).
- **Second encoder**: BGE-base-en-v1.5 replication of the held-out design (predictions to be recorded
  before any vectors exist).
- **Engine performance**: round 10 kernel and speed results, and the VIBE submission (three datasets).
- **Label follow-up and human audit**: waiting on the advisor's decisions.

---

## 13. Glossary

- **ANN**: approximate nearest-neighbour search; trades a little accuracy for large speed-ups.
- **HNSW**: hierarchical navigable small-world graph, the most widely used ANN index; each vector is a
  node linked to up to M (2M at the base layer) neighbours.
- **ef / efConstruction**: the search-list width at query time / during construction; wider is more
  accurate and slower.
- **IVF**: inverted-file index; vectors are assigned to clusters (centroids), and a search scans the
  nearest few clusters (nprobe).
- **In-degree**: how many other nodes link to a node; a node with few incoming links is hard for a
  graph search to reach.
- **Hubness**: in high dimensions, a few points are near neighbours of very many others (hubs) and
  some of almost none (anti-hubs).
- **Within-query (Mantel–Haenszel) ratio**: compares miss rates of new and old items within the same
  query and rank band, then pools across queries.
- **Arrival group / cohort**: the snapshot (1, 2 or 3) in which a document first appeared.
- **Retrievability**: how easily a document can be retrieved across a population of queries.

## 14. Key references

Peer-reviewed unless marked (PP).

- Malkov & Yashunin. Efficient and robust approximate nearest neighbor search using HNSW graphs. TPAMI 2018.
- Xu, Dai, Li, Wang, Yue. Through the Lens of Hubness: A Revisit on Graph-Based ANN Search. SIGMOD 2026 (PACMMOD 4(3), art. 243). https://doi.org/10.1145/3802120
- Elliott & Clark. The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in HNSW. ICTIR 2024. https://arxiv.org/abs/2405.17813
- GRAB-ANNS: High-Throughput Indexing and Hybrid Search via GPU-Native Bucketing (PP). https://arxiv.org/abs/2604.16402
- Down with the Hierarchy: the "H" in HNSW stands for "Hubs" (PP). https://arxiv.org/abs/2412.01940
- Xiao et al. Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation (PP). https://arxiv.org/abs/2407.07871
- Singh et al. FreshDiskANN. https://arxiv.org/abs/2105.09613 ; IP-DiskANN (PP). https://arxiv.org/abs/2502.13826
- Revisiting the Index Construction of Proximity Graph-Based ANN Search. PVLDB 18(6) 2025. https://arxiv.org/abs/2410.01231
- Loh & Lee. Per-Query Recall Guarantees for ANN Search in Dynamic Vector Databases (PP, ICDE 2027 submission). https://arxiv.org/abs/2611.07579
- Baranchuk et al. DeDrift: Robust Similarity Search under Content Drift. ICCV 2023.
- Simhadri et al. Results of the Big ANN: NeurIPS'23 competition. https://arxiv.org/abs/2409.17424
- VIBE: Vector Index Benchmark for Embeddings. https://arxiv.org/abs/2505.17810
- Azzopardi & Vinay. Retrievability: an evaluation measure for higher order information access tasks. CIKM 2008. https://doi.org/10.1145/1458082.1458157
- Singh & Joachims. Fairness of Exposure in Rankings. KDD 2018. https://arxiv.org/abs/1802.07281
- Diaz et al. Evaluating Stochastic Rankings with Expected Exposure. CIKM 2020.
- Radovanović, Nanopoulos, Ivanović. Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data. JMLR 2010.
- LongEval 2026 Sci extended overview. CEUR-WS Vol-4283.

The full list, with verification status and the practitioner sources (GitHub issues, vendor
documentation), is in the research notes.
