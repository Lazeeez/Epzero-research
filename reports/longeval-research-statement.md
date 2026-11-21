# Does the index matter over time? Approximate search, index history and collection growth in longitudinal dense retrieval

**Statement of research** · Lajat · November 2026

## Summary

Evaluation campaigns such as the LongEval lab at CLEF measure how well retrieval systems hold up as a document collection grows and changes. Their dense-retrieval results are obtained with exact search. Deployed systems do not use exact search: they use an approximate index that is built once and then updated in place for months. This research asks what that difference does to a longitudinal evaluation. It measures, on a real relevance-judged collection that nearly doubles in size, how much effectiveness an approximate index loses, whether that loss changes over time, whether an index that has been kept alive behaves differently from a fresh one, and which documents bear the loss.

The main experiments are complete. In brief: the loss is small and, for most configurations, stable; keeping an index alive is equivalent to rebuilding it; but an index that grows by appending retrieves its newest documents less reliably than its oldest, and this is caused by when a document enters the index. A follow-up study now examines how much the available relevance labels can say about the size of the loss. The work is aimed at the CLEF 2027 conference.

## Background

**Longitudinal evaluation.** Most retrieval benchmarks are snapshots: a fixed collection, a fixed set of queries, one score. Real collections are not fixed. LongEval addresses this by releasing the same collection at several points in time and asking how a system's effectiveness changes from one to the next. Its 2026 scientific-search track provides three cumulative snapshots of a growing collection, with queries and relevance judgments at each.

**Dense retrieval and its index.** A dense retriever represents every document and query as a vector and returns the documents whose vectors are closest to the query's. Finding the closest vectors exactly means comparing the query with every document, which is too slow for a large collection. In practice a system builds an approximate nearest-neighbour index. The two families in common use are graph indexes, which link each document to its near neighbours and answer a query by walking the graph, and partition indexes, which group documents into cells and search only the most promising cells. Both return most, but not all, of the true nearest documents, and both have a setting that trades speed against the share of true neighbours found.

**Why time matters for an index.** An approximate index is a long-lived data structure. New documents are inserted into it as they arrive; it is rarely rebuilt, because rebuilding a large index is expensive. Its structure therefore reflects the order in which documents arrived, and its search setting was usually tuned at some earlier size. Neither fact matters for exact search, which is why a longitudinal evaluation that uses exact search cannot see their effect.

## The gap this research addresses

Two bodies of work each cover half of the question.

- **Work on vector indexes** studies how index quality changes under insertions and deletions, and how to repair it (FreshDiskANN, SPFresh, Quake, DeDrift, and recent work on when to repair a graph index). It measures quality as the share of true nearest neighbours found, on workloads that have no relevance judgments. It cannot say what a lost neighbour costs a user.
- **Work on longitudinal evaluation** measures effectiveness against relevance judgments as collections change (the LongEval overviews; Keller et al.; Kuissi et al.). It uses exact search or an index that is rebuilt and never ages.

Static studies connect the two at a single point in time: on a fixed collection, a well-tuned graph index costs little effectiveness (Lin; Tu et al.). Elliott and Clark show that the order of insertion affects how many neighbours a graph index finds. What has not been done is to follow the effectiveness cost of approximation across the snapshots of a real, relevance-judged, growing collection, with the index maintained the way a deployed system maintains it.

## Aim and research questions

The aim is to establish what an evaluation campaign, and an operator, should expect from an approximate index as a collection grows: what is safe to assume, what is not, and why.

1. **Approximation over time.** How much effectiveness does approximate search lose at each snapshot, and does that loss change as the collection grows? If it is stable, conclusions drawn with exact search carry over to deployed systems. If it grows, they do not.
2. **Index history.** Does an index that has been updated in place across all snapshots give different results from one rebuilt from the same documents? Operators rebuild on a schedule because they suspect that an index degrades with age. A clear answer either justifies that cost or removes it.
3. **Who bears the loss.** Are the documents that arrived most recently retrieved as reliably as those that have been in the index longest? New documents are what a growing collection, and a longitudinal evaluation, are about. A loss that is small on average could still fall mainly on them.
4. **Remedies.** When approximate search misses relevant documents, is it better to repair the index or simply to search more widely? Several repair methods have been proposed; the practical question is whether they beat spending the same effort on a wider search.
5. **What the evidence can support.** The relevance judgments released by such campaigns were collected for the systems that took part. How far can they support a comparison between systems that differ only in their index?

## Approach

**Hold everything constant except the index.** The study uses the LongEval 2026 scientific-search collection: three cumulative snapshots growing from about 0.87 to 1.66 million documents, with the organisers' own document and query embeddings. The same embeddings are searched exactly and approximately at every snapshot. Because the model, the documents, the queries and the judgments are identical in both cases, any difference in effectiveness is the contribution of the index and of nothing else.

**Indexes as they are deployed.** The study covers both index families: a graph index in two independent implementations, so that a result is not an artefact of one piece of software, and a partition index. Each is maintained incrementally, with documents inserted as the snapshots arrive. For each, the study also builds a fresh index from the same documents, which separates the effect of an index's history from the effect of its contents. Search settings are both tuned to a common accuracy target at each snapshot and held fixed, which separates the effect of growth from the effect of re-tuning.

**A confirmatory core.** The central analysis, on approximation over time and on index history, was specified in full and fixed before any held-out test query was used, and it was then run once. This guards the main claim against choices made after seeing results. Development and tuning used a separate set of training queries.

**Controlled follow-up experiments.** Questions that arose from the first results are answered by experiments that change one thing at a time. The most important concerns the newest documents: the same collection is indexed several times, changing only the order in which documents enter the index, so that the effect of a document's position can be separated from the document itself. This was run at full collection scale, repeated on smaller subsets, and repeated on two public vector benchmarks that have no time dimension at all. These analyses were designed after the first results were seen and are reported as exploratory.

**A study of the labels.** Finally, the research examines the relevance judgments themselves: how well they agree with human assessment, how completely they cover the documents that an approximate index returns, and how those two properties limit any statement about effectiveness.

## Progress and findings

| Question | Status | Finding in brief |
| --- | --- | --- |
| 1. Approximation over time | Complete (confirmatory) | Small loss; stable for two of three configurations, inconclusive for the third |
| 2. Index history | Complete (confirmatory) | No effect: updating in place is equivalent to rebuilding |
| 3. Who bears the loss | Complete (exploratory) | The newest documents, because of when they are inserted |
| 4. Remedies | Complete (development) | Repairs do not beat a wider search at equal effort |
| 5. What the evidence supports | In progress | The labels are lenient and incomplete; a follow-up is under way |

**The loss from approximation is small, and mostly stable.** At the accuracy level commonly used in practice, where the index finds about 95% of the true nearest documents, approximate search loses between 0.005 and 0.012 nDCG@10 at every snapshot. For two of the three configurations, the change in that loss as the collection nearly doubled stayed within a margin of 0.01 that was fixed in advance. The third was inconclusive. Its cause is instructive: its search setting had been tuned on a small sample of queries and was never widened, while the share of neighbours it found slipped by about a point as the collection grew. A follow-up analysis shows that a tuning sample of that size misses such a slip about one time in five.

**An index does not degrade with age when a collection only grows.** An index updated in place across all three snapshots gave exactly the same results as one rebuilt from the same documents in the same order, in every one of 27 comparisons across index types and random seeds. Under append-only growth there is no accumulated damage for a rebuild to repair. What a rebuild can change is the order of insertion, which leads to the next finding.

**The newest documents are retrieved less reliably, and the cause is their position.** In a graph index that grew as the documents arrived, a document from the latest snapshot is missed about 1.3 to 1.4 times as often as a document from the first snapshot that is equally close to the same query. When the same collection is indexed in random order, the two are missed equally often. When it is indexed with the latest documents first, they are missed about 0.8 times as often. The documents are the same in all three cases; only their position in the index changed. The pattern holds at three collection sizes, and the effect of position appears again on two public benchmarks. A partition index, which has no such order, shows only a small difference. So an evaluation that asks how a system serves new content is asking about exactly the documents that a graph index serves worst.

**Repairing the index does not beat searching more widely.** Four repair strategies, including a recently published one, were tested. Each recovers some of the missed documents at a fixed search setting. None does better than an ordinary wider search that costs the same amount of work.

**The labels limit what can be said about the size of the loss.** The judgments in the release were produced by a language model, and on the queries that also have human grades the two agree only weakly. They also cover only the documents returned by the systems that took part: most of the documents that an approximate index returns in place of a missed one have no judgment and are counted as not relevant. The measured loss is therefore partly a property of the labels. The findings on index history and on the newest documents are unaffected, because they do not use relevance labels.

## What remains

- **The label follow-up.** Every document that any compared system returned will be judged by one consistent assessor, following the organisers' own judging procedure, with a small blind human audit alongside. This will show how much of the measured loss remains when every system is judged equally. The criteria for accepting those judgments were written down before any were produced.
- **Open questions.** Queries whose best answers are recent documents turned out to be harder in every insertion order; why is not yet understood. The collection only grows, so deletions and rewritten documents are untested.
- **Writing.** A full paper and a reproducible record of every experiment.

## Intended contributions

1. The first measurement of what approximate search costs in effectiveness over time, on a relevance-judged, evolving collection, under an analysis fixed in advance.
2. A clear answer for operators: under append-only growth, an index does not need rebuilding because of its age, but its search setting needs re-checking as the collection grows.
3. Evidence that a graph index built by appending under-serves its newest documents, with a controlled demonstration of the cause.
4. Guidance for evaluation campaigns on what pooled, model-generated relevance labels can and cannot support when systems differ only in their index.
5. Negative results for index repair at equal effort, and a fully documented, reproducible experimental record.

## Plan

| When | Step |
| --- | --- |
| Done (November 2026) | Main experiments and analysis; follow-up experiments on document arrival |
| December 2026 – January 2027 | Label follow-up study |
| January – March 2027 | Manuscript and reproducible artifact |
| Spring 2027 | Submission to CLEF 2027 |

**Target venue:** the CLEF 2027 conference (Bucharest, September 2027).

## Scope and limitations

- One collection and one embedding model. The conclusions are scoped to them; only the effect of insertion position has been reproduced elsewhere.
- The collection only grows, so the equivalence between updating and rebuilding does not extend to deletions or rewritten documents.
- The effectiveness measurements depend on the released labels. The follow-up addresses their coverage; human grades exist for only a small set of queries.
- The held-out queries were used once for the confirmatory analysis. The follow-up analyses reuse them and are exploratory.

## Selected references

- Cancellieri et al. [LongEval at CLEF 2025: Longitudinal evaluation of IR systems on web and scientific data.](https://doi.org/10.1007/978-3-032-04354-2_20) CLEF 2025.
- Bouaraba et al. [Extended overview of the CLEF 2026 LongEval lab.](https://ceur-ws.org/Vol-4283/paper374.pdf) CEUR vol. 4283, 2026.
- Keller, Breuer and Schaer. [Evaluation of temporal change in IR test collections.](https://doi.org/10.1145/3664190.3672530) ICTIR 2024.
- Kuissi et al. [Still fresh? Evaluating temporal drift in retrieval benchmarks.](https://arxiv.org/abs/2603.04532) arXiv:2603.04532, 2026.
- Lin. [Operational advice for dense and sparse retrievers: HNSW, flat, or inverted indexes?](https://doi.org/10.18653/v1/2025.acl-industry.61) ACL 2025 (Industry Track).
- Tu et al. [Approximate nearest neighbor search and lightweight dense vector reranking in multi-stage retrieval architectures.](https://doi.org/10.1145/3409256.3409818) ICTIR 2020.
- Elliott and Clark. [The impacts of data, ordering, and intrinsic dimensionality on recall in hierarchical navigable small worlds.](https://doi.org/10.1145/3664190.3672512) ICTIR 2024.
- Malkov and Yashunin. [Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs.](https://doi.org/10.1109/TPAMI.2018.2889473) IEEE TPAMI, 2020.
- Singh et al. [FreshDiskANN: A fast and accurate graph-based ANN index for streaming similarity search.](https://arxiv.org/abs/2105.09613) arXiv:2105.09613, 2021.
- Xu et al. [SPFresh: Incremental in-place update for billion-scale vector search.](https://doi.org/10.1145/3600006.3613166) SOSP 2023.
- Mohoney et al. [Quake: Adaptive indexing for vector search.](https://arxiv.org/abs/2506.03437) arXiv:2506.03437, 2025.
- Baranchuk et al. [DeDrift: Robust similarity search under content drift.](https://doi.org/10.1109/ICCV51070.2023.01012) ICCV 2023.
- Mandarapu and Kunkunuru. [When to repair a graph ANN index.](https://arxiv.org/abs/2607.00728) arXiv:2607.00728, 2026.
