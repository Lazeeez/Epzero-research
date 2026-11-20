# Separating Index History, Search Effort, and Relevance in a Growing Dense Retrieval Collection

**Working manuscript. Development evidence only. Not submission-ready.**
**2026-11-18:** retained as the empirical foundation for the author's new method-paper
objective. The method contribution and competitive evaluation are under development
in [METHOD_RESEARCH_PLAN.md](../repair/METHOD_RESEARCH_PLAN.md); this draft does not contain
a validated novel algorithm or a measured method speedup. The empirical objective
described below records the earlier study and no longer defines the final paper's aim.

Authors, venue, final protocol, held-out results, matched native-cost comparison and
external replication decision remain open. No claim of established novelty is made here.

## Abstract — provisional

When a dense retrieval collection grows, a reduction in approximate-search quality can
reflect construction history, search effort, or changes in the exact retrieval problem.
We study these alternatives using three cumulative LongEval scientific-retrieval snapshots
containing 0.87–1.66 million fixed 2,560-dimensional document vectors. Development
experiments cover two HNSW implementations, three construction modes and three seeds.
Incremental construction and rebuilding with the same insertion history reproduce the
same states and rankings in all tested paired conditions. Two severe query failures are
recoverable without changing the graph. A calibrated offline search-work budget removes
zero-neighbor-recall cases in the development cohort, while leaving substantial per-query
shortfalls. Relevance scoring further shows that equal mean neighbor recall need not
produce equal nDCG. These observations motivate evaluating maintenance decisions using
explicit construction controls, native search costs and relevance judgments. IVF controls
also reproduce same-training rebuilds and show retraining effects that depend on probe
settings. Held-out temporal relevance evaluation remains pending; current results do not establish
general index aging, a new search algorithm or a broadly optimal maintenance policy.

## 1. Research question

An approximate index is part of the retrieval system. Its construction and query settings
can alter which documents are returned even when document and query representations are
fixed. In a growing collection, a quality decline therefore does not by itself establish
that an index must be rebuilt. A fresh construction with a different insertion order or
training sample changes more than whether the object was retained.

We ask: **what evidence distinguishes a maintenance problem from insufficient search,
and when does that distinction matter for relevance effectiveness?** The intended
contribution is a controlled empirical answer with operational consequences. It does not
depend on introducing a new ANN algorithm or on obtaining a positive aging result.

Our present scope is cumulative insert-only growth with unchanged shared documents and
vectors. It excludes deletions, embedding refreshes and document rewrites. We do not claim
that existing exact-search LongEval baselines suffer hidden ANN degradation.

## 2. Related work and proposed distinction

Elliott and Clark show that HNSW insertion order and vector-space properties influence
recall and can affect embedding-model rankings [1]. Thus insertion-order sensitivity alone
is not our proposed novelty. Lin compares HNSW and flat search using retrieval quality,
indexing time and query performance [2]; exact-versus-approximate relevance comparison is
also not new by itself.

PLAID SHIRTTT addresses streaming retrieval where earlier centroid training can poorly
represent later tokens [3]. Quake adapts partitions and query execution to changing vector
workloads [4]. These precedents make a broad claim that dynamic index maintenance is an
unexplored problem untenable.

The candidate distinction is the joint evaluation of verified temporal inputs, identical-
history rebuild controls, measured search work, and snapshot-specific relevance outcomes.
Whether this combination yields a sufficiently new finding must be established by the
remaining experiments and a fuller related-work comparison. This section is not an
exhaustive novelty review.

## 3. Data and validity controls

The three archived snapshots contain 869,902, 1,322,720 and 1,661,900 documents. Shared
document IDs retain bitwise-identical vectors and unchanged audited source text. No IDs
are removed. The stored row ordering changes, so all transitions are joined by external
document ID rather than by row position.

We use the archived float32 document vectors and a pinned CUDA float32 reconstruction of
the organizer-specified query encoder, with TF32 disabled. Vector-level and exact-ranking
precision diagnostics were performed separately. The reconstruction does not establish
the original server's complete historical runtime configuration.

The 100 training queries are deterministically split into 50 calibration and 50 development
queries. Exact top-100 references scan the entire current collection with float64
accumulation. The separate 381 test queries have not been evaluated for retrieval in this
development program. Their IDs and judgment coverage were previously audited.

All results bind source, query, document, configuration and output identities. Returned
PACE artifacts are checked locally. Synthetic tests establish execution correctness;
they are not counted as evidence of retrieval effectiveness on the real collection.

## 4. Experimental design

### 4.1 Construction controls

For Epzero and FAISS HNSW, use seeds 11, 29 and 47 with M=16 and efConstruction=100,
single-thread construction, full float32 vectors and no compression. Compare incremental
insertion, rebuilding the identical chronological insertion sequence, and a declared
shuffled rebuild. Evaluate each trajectory at all three snapshots.

Original search regimes use ef=200 or a calibration-selected effort from 100, 200, 400,
800 targeting mean neighbor Recall@100 of 0.95. Construction-state hashes and exported
rankings are compared directly for the identical-history control.

### 4.2 Search diagnosis

For selected severe failures, trace the unchanged graph and upper route while varying
search effort. A deferred-candidate experiment expands a selected candidate after it was
admitted and then evicted without expansion. This is a diagnostic intervention selected
with knowledge of the failing cases, not an online selection policy.

The full-cohort work diagnostic uses complete native ef=3200 score streams, evaluating
offline prefixes at declared distance budgets. It includes upper-layer and entry scoring.
These prefixes support analysis of actual distance work, but do not measure a native
budget-capped search API or its latency. They must not be compared with ef=200 as if both
beam capacity and average work were held constant.

### 4.3 IVF controls

Three arms separate frozen-training incremental insertion, rebuilding with the same
training and insertion history, and retraining on the current snapshot before rebuilding.
IndexIVFFlat uses 1,024 lists and a deterministic bounded sample of 65,536 live vectors.
Twenty spherical clustering iterations, one restart and explicit seeds are declared.
Probe calibration and frozen/current settings are evaluated separately. Native scanned-
vector counters are checked against probed-list lengths; coarse comparisons are reported
separately. The experiment records native-call latency and maintenance costs.

### 4.4 Relevance and final inference

Current development relevance analysis uses DCTR and raw training labels belonging to
snapshot 1. Later snapshots scored against those labels are historical-label diagnostics.
They cannot establish contemporaneous temporal effectiveness, since newly added documents
may be unjudged. Final evaluation will use snapshot-specific test judgments with explicit
query populations and paired query-level analysis. The final protocol remains unfrozen;
see `../FINAL_EVALUATION_PROTOCOL_DRAFT.md`.

## 5. Verified development observations

### 5.1 Keeping an index alive versus identical-history rebuilding

All 18 paired snapshot comparisons reproduce the same index fingerprint and both ranking
files under incremental construction and identical-history replay. This rules out a benefit
from that particular rebuild operation in these conditions. It does not rule out different
training, construction settings, insertion histories, or other data-update regimes.

Shuffled rebuilding is not uniformly beneficial. The largest observed final-snapshot
improvement is concentrated in Epzero seed 11; other tested seeds/backends do not exhibit
the same pattern. We therefore do not select a favorable seed as the primary result.

### 5.2 Search failures and work

The two investigated seed-11 failures recover on unchanged graphs at larger effort.
The controlled deferred-candidate study produces two targeted 99% recall rescues; paired
wider searches return identical full rankings at equal recorded distance counts. The
intervention has not demonstrated an advantage over increasing search work.

All nine snapshot/seed calibration conditions select 4,000 evaluations from the declared
work grid. Final development mean neighbor recall is 0.9618, 0.9668 and 0.9668 for seeds
11, 29 and 47. No query has zero recall there, but minima are 0.64, 0.66 and 0.69.
The unchanged discrete budget choice is not proof that the optimal budget is constant.

### 5.3 Neighbor recall and relevance differ

Against historical DCTR labels at snapshot 3, exact nDCG@10 is 0.266470. The offline
4,000-evaluation condition yields:

| Seed | Mean neighbor Recall@100 | nDCG@10 | ANN minus exact nDCG@10 |
|---:|---:|---:|---:|
| 11 | 0.9618 | 0.257788 | -0.008682 |
| 29 | 0.9668 | 0.257788 | -0.008682 |
| 47 | 0.9668 | 0.242584 | -0.023886 |

Equal mean neighbor recall can therefore conceal different relevance outcomes. At native
ef=3200, all three seeds reach nDCG@10 0.266523, slightly above the exact similarity
ranking. This is possible because exact nearest neighbors are not a relevance optimum.
These are descriptive development observations, not significance or held-out claims.

### 5.4 IVF maintenance depends on the search regime

Nine full-data IVF trajectories cover the same three seeds and snapshots. Incremental
and same-training rebuild modes have identical serialized states and development rankings
at all nine paired snapshot conditions. At final current-calibrated nprobe=128, mean
neighbor recall changes from 95.88% to 95.64%, 95.46% to 95.34%, and 96.54% to 95.78%
after retraining for seeds 11, 29 and 47. Training plus rebuilding takes approximately
154–157 seconds versus 19 seconds for delta insertion, including adapter overhead.

The direction is not universal. Seed 29 selects 256 probes at snapshot 1 and 128 later.
At its original 256-probe setting, final retraining improves recall from 97.96% to 98.30%.
All-list searches recover the exact top-100 neighbor sets. These results motivate retaining
both search regimes when assessing maintenance and do not establish general harm from
retraining. Same nprobe also does not mean equal work because inverted-list sizes differ.

Against historical DCTR labels, current-calibrated final nDCG@10 decreases after retraining
in all three seeds, while relevant-document Recall@100 increases for two seeds and decreases
for one. These descriptive development measures cannot replace snapshot-specific held-out
relevance judgments. Full evidence and limits are in `../IVF_RESULTS.md`.

## 6. Limitations and unfinished evidence

The current evidence covers one collection, one document representation, three snapshots,
three construction seeds and a small reused development query cohort. There is no
established inference over unseen query populations or other collections. Historical
training labels cannot validate temporal relevance change. Query reconstruction differs
in unspecified ways from the historical embedding runtime. Native cross-backend latency
and maintenance comparisons are incomplete, and offline search prefixes do not establish
a deployable algorithm's speed. IVF experiments are independently verified; matched-protocol
native HNSW cost jobs are prepared but have not returned. Native indexes and centroids were
not reconstructed/retrained locally; returned state hashes are corroborated by independent
score, geometry, training-sample, source and control-ranking checks.

The paper must retain these limits even if final results are favorable. A null maintenance
effect or negligible relevance gap is an allowed outcome. Publication strength depends on
the scope and consequence of the completed evidence, not the number of executed jobs.

## 7. Reproducibility

The artifact ledger is maintained in `../CURRENT_STATUS.md` and the experiment-specific
reports. All large PACE downloads and temporary native state remain on compute-node
temporary disk. Only bounded evidence is returned. Original validated sources and sealed
experiment supplements are preserved. The final release will include executable frozen
protocols, checksums, figures and reproduction commands after held-out validation.

## References

1. Owen Pendrigh Elliott and Jesse Clark. *The Impacts of Data, Ordering, and Intrinsic
   Dimensionality on Recall in Hierarchical Navigable Small Worlds*. 2024.
   https://arxiv.org/abs/2405.17813
2. Jimmy Lin. *Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted
   Indexes?* 2024. https://arxiv.org/abs/2409.06464
3. Dawn Lawrie, Efsun Kayi, Eugene Yang, James Mayfield, and Douglas W. Oard.
   *PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval*. 2024.
   https://arxiv.org/abs/2405.00975
4. Jason Mohoney et al. *Quake: Adaptive Indexing for Vector Search*. 2025.
   https://arxiv.org/abs/2506.03437
