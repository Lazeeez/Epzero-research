# Does the index matter over time? Approximate search, index history and collection growth in longitudinal dense retrieval

**Statement of research** · Lajat · November 2026

## Motivation

Search collections are not static. Documents arrive every day, and a retrieval system that worked well last year may not work as well today. The LongEval lab at CLEF studies this directly: it measures how retrieval effectiveness changes as a collection and its queries evolve over time.

Its dense retrieval results, however, describe a system that is rarely deployed. The lab's dense baseline uses exact search, comparing each query with every document. Real systems cannot afford that at scale. They use an approximate nearest-neighbour index, which is built once and then updated in place as new documents arrive. An approximate index trades a little accuracy for a large gain in speed, and how that trade behaves over months of growth has not been measured on a relevance-judged collection.

Two research communities each hold half of the picture. Work on vector indexes tracks index quality over time, but in terms of nearest-neighbour recall, on workloads without relevance judgments. Work on longitudinal evaluation tracks effectiveness over time, but with exact search or with an index that never changes. This research joins the two.

## Research questions

1. **Approximation over time.** Does the effectiveness lost to approximate search stay constant as a collection grows, or does it change, so that conclusions drawn with exact search do not transfer to deployed systems?
2. **Index history.** Does an index that has been updated in place for a long time behave differently from one rebuilt from scratch? Operators rebuild on a schedule largely because they cannot answer this.
3. **Who bears the loss.** When a collection grows, are its newest documents retrieved as reliably as its oldest? These are the documents a longitudinal evaluation is about.
4. **Remedies.** When approximate search fails, what is the cheapest way to recover: repairing the index, or simply searching harder?
5. **Evidence.** How far can the relevance labels that such campaigns release support a comparison between indexes?

## Approach

I study these questions on LongEval-Sci 2026, a public scientific-search collection released as three cumulative snapshots that nearly double in size, with relevance judgments at each snapshot.

The central idea is to hold everything constant except the index. The same document and query embeddings are searched exactly and approximately at every snapshot, so that any difference in effectiveness is the index's contribution and nothing else. I compare the two families of index that dominate practice, a graph index in two independent implementations and a partition index, each maintained the way a deployed system would maintain it.

The study follows a confirmatory design. The main analysis was specified and fixed before any held-out query was used, and it was run once. Further questions that arose from those results are answered with controlled follow-up experiments and are reported as exploratory.

## Progress and findings so far

The main experiments are complete. In summary:

- **Approximation over time.** At the accuracy levels commonly used in practice, the effectiveness lost to approximate search is small. For two of the three configurations, its change as the collection nearly doubled stayed within a margin fixed in advance; the third was inconclusive. The cause was how its search setting had been tuned, not the index itself, and this points to a practical risk: a setting chosen once on a small sample can fall behind as a collection grows.
- **Index history.** An index kept alive and updated in place gave the same results as one rebuilt from the same history. Under append-only growth, accumulated "drift" is not what operators need to worry about.
- **Newest documents.** A graph index that grows by appending retrieves its most recent documents less reliably than its oldest ones. Controlled experiments at full collection scale show that this follows from when a document enters the index, not from the documents themselves, and the effect replicates on public benchmarks.
- **Remedies.** Several index-repair strategies were tested. None did better than spending the same effort on a wider search.
- **Evidence.** The findings on index history and on newest documents do not depend on relevance labels. The effectiveness measurements do, and the released labels limit how precisely the size of the loss can be stated. A follow-up study, now under way, addresses this.

## Expected contributions

1. The first measurement of what approximate search costs in effectiveness over time on a relevance-judged, evolving collection, under a design fixed in advance.
2. Practical guidance for operators on when rebuilding an index is necessary and when it is not.
3. Evidence about which documents an approximate index under-serves as a collection grows, and why.
4. Guidance for evaluation campaigns on what their relevance labels can and cannot support when systems differ only in their index.
5. A reproducible, fully documented experimental record.

## Plan

| When | Step |
| --- | --- |
| Done (November 2026) | Main experiments and analysis; follow-up experiments on document arrival |
| December 2026 – January 2027 | Follow-up study on relevance labels |
| January – March 2027 | Manuscript and reproducible artifact |
| Spring 2027 | Submission to CLEF 2027 |

**Target venue:** the CLEF 2027 conference (Bucharest, September 2027).

## Scope and limitations

- The study uses one collection and one embedding model. Its conclusions are scoped to them, with one finding replicated on public benchmarks.
- The collection only grows. Deletions and rewritten documents are outside the scope.
- The effectiveness measurements depend on the relevance labels that the campaign released; the follow-up study examines that dependence.

## Selected references

- Cancellieri et al. [LongEval at CLEF 2025: Longitudinal evaluation of IR systems on web and scientific data.](https://doi.org/10.1007/978-3-032-04354-2_20) CLEF 2025.
- Bouaraba et al. [Extended overview of the CLEF 2026 LongEval lab.](https://ceur-ws.org/Vol-4283/paper374.pdf) CEUR vol. 4283, 2026.
- Keller, Breuer and Schaer. [Evaluation of temporal change in IR test collections.](https://doi.org/10.1145/3664190.3672530) ICTIR 2024.
- Lin. [Operational advice for dense and sparse retrievers: HNSW, flat, or inverted indexes?](https://doi.org/10.18653/v1/2025.acl-industry.61) ACL 2025 (Industry Track).
- Malkov and Yashunin. [Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs.](https://doi.org/10.1109/TPAMI.2018.2889473) IEEE TPAMI, 2020.
- Elliott and Clark. [The impacts of data, ordering, and intrinsic dimensionality on recall in hierarchical navigable small worlds.](https://doi.org/10.1145/3664190.3672512) ICTIR 2024.
- Xu et al. [SPFresh: Incremental in-place update for billion-scale vector search.](https://doi.org/10.1145/3600006.3613166) SOSP 2023.
- Kuissi et al. [Still fresh? Evaluating temporal drift in retrieval benchmarks.](https://arxiv.org/abs/2603.04532) arXiv:2603.04532, 2026.
