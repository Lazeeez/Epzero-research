# Does the index matter over time? Approximate search, index history and collection growth in longitudinal dense retrieval

**Statement of research** · Lajat · November 2026 (updated 21 November)

## The problem

LongEval measures how retrieval effectiveness changes as a collection and its queries evolve. Its dense runs use exact search: the organisers' Qwen3-Embedding-4B baseline compares each query with every document. Deployed dense retrieval instead runs on an approximate nearest-neighbour (ANN) index, usually an HNSW graph or an IVF partition, which is built once and then updated in place as documents arrive.

So its conclusions describe a system that is rarely deployed. Three questions are open:

1. **Approximation over time.** Does the effectiveness lost to approximate search stay constant as the collection grows, or does it change, so that conclusions drawn with exact search do not transfer?
2. **Index history.** Does an index updated in place for months behave differently from one rebuilt from scratch? Operators rebuild on a schedule largely because they cannot answer this.
3. **Who bears the loss.** When a collection grows by appending, are its newest documents, the ones a longitudinal evaluation is about, retrieved as reliably as its oldest?

Both literatures stop short of the join. Systems work tracks index quality over time in neighbour-recall units on unjudged workloads (FreshDiskANN, SPFresh, Quake, DeDrift, When to Repair). IR work tracks effectiveness over time with exact search or a fixed index (LongEval, Keller et al., Still Fresh?). Static studies put the cost of an HNSW index under about 0.01–0.02 nDCG@10 (Lin; Tu et al.), and Elliott and Clark show that HNSW recall depends on insertion order. Nobody has followed these quantities across snapshots of a real, relevance-judged collection.

## Approach

I use LongEval-Sci 2026 (CORE scientific search): three cumulative snapshots of 0.87M, 1.32M and 1.66M documents with released Qwen3-Embedding-4B vectors, 100 training queries and 381 held-out test queries.

The method follows Stahlberg and Byrne's split of search error from model error: run the **same embeddings with exact search and with approximate search** at every snapshot. The signed ANN-minus-exact difference is the index's contribution.

| Arm | Purpose |
| --- | --- |
| Exact float64 search | Reference with no index error |
| HNSW (two independent implementations), updated in place | Deployed graph index |
| Same HNSW, rebuilt from the identical insertion history | Isolates index history from content |
| IVF-Flat, training frozen at snapshot 1 vs refreshed | Partition index and its maintenance choice |
| HNSW built on the final snapshot in three orders: as arrived, random, reversed | Separates a document's position in the build from the document |

Every index uses three construction seeds. Search settings are chosen on calibration queries to reach 95% neighbour recall, and also held fixed, to separate drift from recalibration.

## Research questions

- **RQ1 (history).** Does keeping an index alive as the collection grows change its results compared with rebuilding from the same insertion history?
- **RQ2 (approximation over time).** How large is the ANN-minus-exact nDCG@10 gap at each snapshot, and does it change from snapshot 1 to snapshot 3?
- **RQ3 (failures).** Where do approximate searches fail, and what does recovering those failures cost?
- **RQ4 (repair).** Do query-driven graph repairs beat simply searching wider at equal search work?
- **RQ5 (arrival, added after the held-out results).** Are later arrivals retrieved less reliably, and is that a property of the documents or of when they were inserted?

## Findings

The protocol for RQ1 and RQ2 was frozen and hashed before any test-query retrieval ran, and its analysis was run once. Everything marked *exploratory* was designed after those results were seen. Every number is re-verified from complete logs and checksums.

**Approximation over time (RQ2, frozen).** The analysis covers the 374 test queries with positive GPT-OSS-120B judgments at all three snapshots. The contrast is the change in the ANN-minus-exact nDCG@10 gap from snapshot 1 to snapshot 3, with a pre-specified margin of ±0.01.

| Arm (calibrated to 95% neighbour recall) | Gap change | 95% interval | Holm p | Verdict |
| --- | --- | --- | --- | --- |
| HNSW implementation 1 | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| HNSW implementation 2 | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| IVF-Flat, frozen training | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

The gap itself is 0.005–0.012 nDCG@10 at every snapshot, and the exact-search reference reproduces the lab's official dense baseline to within 0.0014. For scale, exact search's own change over the same snapshots is +0.0070 [−0.0000, +0.0139], so the margin is about as large as the trend it is read against.

The two HNSW arms differ because of calibration, not implementation. Implementation 1's 50 calibration queries stayed just above the 95% target, so its search width never changed, while its neighbour recall on the test queries fell from 94.5% to 93.5% and its gap widened from −0.006 to −0.012. Implementation 2's search was widened and its gap stayed flat. At the next wider setting, implementation 1's change is −0.0026 [−0.0059, +0.0002] (exploratory). In a resampling study, a 50-query calibration set keeps the too-narrow width in 21–22% of draws; requiring one point of recall above the target cuts that to 5.5% (exploratory).

**Index history has no effect (RQ1, frozen).** Rebuilding from the same insertion history reproduces the incrementally updated index exactly, for both HNSW implementations and for IVF: fingerprints, rankings and scores are identical in all 27 comparisons. In append-only growth, "rebuild or keep updating" is a question of insertion order and construction randomness, not of accumulated drift.

**The latest arrivals are under-retrieved, and insertion position is the cause (RQ5, exploratory).** For each exact top-100 neighbour I record whether the index returned it, and compare later arrivals with first arrivals *of the same query at the same depth* (a Mantel–Haenszel risk ratio stratified by query and rank band). In the held-out runs, HNSW misses later arrivals 1.25–1.33 times as often as first arrivals; IVF 1.08–1.10 times.

To separate the documents from their position, both HNSW implementations were rebuilt on all 1,661,900 final-snapshot documents in three insertion orders, with three seeds each.

Miss ratio of the latest arrivals against the first arrivals, by insertion order:

| Implementation | As arrived (inserted last) | Random order | Reversed (inserted first) |
| --- | --- | --- | --- |
| HNSW implementation 1 | 1.42 [1.25, 1.63] | 1.00 [0.90, 1.10] | 0.82 [0.72, 0.92] |
| HNSW implementation 2 | 1.28 [1.12, 1.47] | 1.03 [0.93, 1.14] | 0.82 [0.72, 0.92] |

The same documents are missed more when they are inserted last, as often as any others at a random position, and less when they are inserted first. This holds at 300,000, 870,000 and 1.66 million documents and at four search widths. In a random-order build the miss rate rises steadily with insertion position, from 5% in the first tenth of insertions to 7.4–7.8% in the last. The effect of position replicates on two public benchmarks (GloVe and SIFT) with randomly assigned arrival groups.

Two cautions apply. The within-query ratio was chosen after a simpler pooled ratio was seen, so it is post hoc; the pooled contrast recorded before the run is confirmed as well (+0.62 [+0.46, +0.80] and +0.77 [+0.57, +0.98]). The pooled ratio is larger, about 1.6, because queries whose neighbours are recent documents are harder in every insertion order.

**Failures and repair (RQ3 and RQ4).** On the held-out queries, 82–89% lose nothing and the worst 5% carry 71–85% of the loss (exploratory). In development, four pre-registered repair methods, including the published NGFix/RFix repair, each raise recall at a fixed setting, but none beats an ordinary wider search at the same number of distance computations.

**What the labels can support (exploratory).** The effectiveness results are measurements under the organisers' LLM labels, and two properties of those labels limit them:

- **Leniency.** On the 29 test queries that also have human grades, the LLM labels agree with them at a kappa of 0.24 [0.14, 0.36], and call 87% of the pairs relevant against 60%.
- **Coverage.** The labels cover a pool of the submitted systems. Of the documents an HNSW index substitutes into a top 10, 76–82% are ungraded and count as non-relevant. With ungraded documents removed from every ranking, the measured gap roughly halves.

Neither figure bounds the true loss, because the relevance of the ungraded substitutes is unknown. The results on neighbour recall, index history and arrival do not depend on any label set.

## Claim and contribution

> On a relevance-judged longitudinal collection that nearly doubles in size, approximate dense retrieval at about 95% neighbour recall loses 0.005–0.012 nDCG@10 against exact search under the organisers' labels. The change in that loss stayed within ±0.01 for two of three calibrated indexes and was inconclusive for the third, whose 50-query calibration missed a one-point drift in recall. Incremental maintenance is exactly equivalent to rebuilding from the same history. An HNSW index built by appending misses its latest arrivals 1.25–1.33 times as often as first arrivals of the same query, and insertion position accounts for it: the ratio is 1.0 at a random position and 0.82 when the same documents are inserted first. Four graph repairs do not beat a wider search at equal work. How much of the measured loss is real depends on documents the release never graded, which the follow-up measures.

The contributions are:

1. The first measurement of the ANN-minus-exact effectiveness gap over time on a relevance-judged test collection, under a pre-registered, hash-frozen held-out protocol.
2. An exact equivalence result for incremental maintenance versus rebuild in append-only growth.
3. Evidence at full collection scale, with a controlled reversal and replication on public benchmarks, that an HNSW index built by appending under-retrieves its latest arrivals because of where they were inserted.
4. A measurement of how often a small calibration set misses recall drift, and a margin that reduces it.
5. An account of what pooled LLM labels can and cannot support when comparing indexes.
6. Pre-registered negative results for four repair strategies, and a checksummed, reproducible artifact.

## What was fixed in advance

The primary analysis was frozen with hashes of its configuration and code on 19 November 2026: the label set and 374-query panel, the contrast, paired sign-randomization tests with Holm adjustment, paired bootstrap intervals and the ±0.01 margin. The insertion-order design and the thresholds for accepting a local judge were each recorded before their output existed.

## Plan

| When | Step |
| --- | --- |
| Done (Nov 2026) | Frozen held-out analysis; insertion-order experiment at full scale; calibration and label analyses |
| In progress | Pilot of a local judge that reproduces the organisers' prompt and parser, on a GPU node |
| Dec 2026 | Freeze the follow-up protocol; grade the whole pool (6,350 query–document pairs) with one assessor; a blind human audit of 200 pairs; coverage analysis |
| Dec 2026–Jan 2027 | Manuscript and reproducible artifact |

**Target venue:** SIGIR 2027 full papers (abstracts 14 January, papers 21 January 2027) if the label follow-up completes in time; otherwise the CLEF 2027 conference track, whose dates are not yet announced.

## Limitations

- One collection and one embedding model. Only the insertion-position effect was replicated elsewhere.
- The collection is append-only, so the equivalence result does not cover deletions or rewritten documents.
- The effectiveness results rest on lenient LLM labels with incomplete coverage; the follow-up addresses coverage.
- The test queries were used once for the frozen analysis. Every exploratory analysis reuses them.
- The full-collection insertion-order experiment used one graph setting. Why queries about recent documents are harder was not examined.

## References

- Lin. [Operational advice for dense and sparse retrievers: HNSW, flat, or inverted indexes?](https://doi.org/10.18653/v1/2025.acl-industry.61) ACL 2025 (Industry Track).
- Tu et al. [Approximate nearest neighbor search and lightweight dense vector reranking in multi-stage retrieval architectures.](https://doi.org/10.1145/3409256.3409818) ICTIR 2020.
- Stahlberg and Byrne. [On NMT search errors and model errors: Cat got your tongue?](https://doi.org/10.18653/v1/D19-1331) EMNLP 2019.
- Keller, Breuer and Schaer. [Evaluation of temporal change in IR test collections.](https://doi.org/10.1145/3664190.3672530) ICTIR 2024.
- Cancellieri et al. [LongEval at CLEF 2025.](https://doi.org/10.1007/978-3-032-04354-2_20)
- Bouaraba et al. [Extended overview of the CLEF 2026 LongEval lab.](https://ceur-ws.org/Vol-4283/paper374.pdf) CEUR vol. 4283.
- Elliott and Clark. [The impacts of data, ordering, and intrinsic dimensionality on recall in hierarchical navigable small worlds.](https://doi.org/10.1145/3664190.3672512) ICTIR 2024.
- Singh et al. [FreshDiskANN.](https://arxiv.org/abs/2105.09613) arXiv:2105.09613, 2021.
- Xu et al. [SPFresh.](https://doi.org/10.1145/3600006.3613166) SOSP 2023.
- Mohoney et al. [Quake: Adaptive indexing for vector search.](https://arxiv.org/abs/2506.03437) arXiv:2506.03437, 2025.
- Baranchuk et al. [DeDrift: Robust similarity search under content drift.](https://doi.org/10.1109/ICCV51070.2023.01012) ICCV 2023.
- Mandarapu and Kunkunuru. [When to repair a graph ANN index.](https://arxiv.org/abs/2607.00728) arXiv:2607.00728, 2026.
- Kuissi et al. [Still fresh? Evaluating temporal drift in retrieval benchmarks.](https://arxiv.org/abs/2603.04532) arXiv:2603.04532, 2026.
- Hua et al. [Dynamically detect and fix hardness for efficient approximate nearest neighbor search (NGFix/RFix).](https://doi.org/10.1145/3769783) Proc. ACM Manag. Data 3, 2025.
- Greenland and Robins. [Estimation of a common effect parameter from sparse follow-up data.](https://doi.org/10.2307/2530643) Biometrics 41(1), 1985.
