# Does the index matter over time? Approximate search, index history and collection growth in longitudinal dense retrieval

**Statement of research** · Lajat · November 2026

## The problem

LongEval measures how retrieval effectiveness changes as a collection and its queries evolve. Its dense runs use exact search: the organisers' Qwen3-Embedding-4B baseline compares each query with every document, rebuilt at each snapshot. Deployed dense retrieval does not work that way. At scale it runs on an approximate nearest-neighbour (ANN) index, usually an HNSW graph or an IVF partition, which is built once and then updated in place as documents arrive.

So LongEval's conclusions about temporal robustness describe a system that is rarely deployed. Two questions are open:

1. **Approximation over time.** Does the effectiveness lost to approximate search stay constant as the collection grows, or does it change, so that conclusions drawn with exact search do not transfer to deployed systems?
2. **Index history.** Does an index that has been updated in place for months behave differently from one rebuilt from scratch? Operators rebuild on a schedule largely because they cannot answer this.

Both literatures stop short of the join. Systems work tracks index quality over time, but in neighbour-recall units on synthetic or unjudged workloads (FreshDiskANN, SPFresh, Quake, DeDrift, When to Repair). IR work tracks effectiveness over time, but with exact search or a fixed index (LongEval, Keller et al. 2024, Still Fresh?). Static studies show that an HNSW index costs under about 0.01–0.02 nDCG@10 on a fixed collection (Lin 2024; Tu et al. 2020). Nobody has followed that quantity across snapshots of a real, relevance-judged test collection.

## Approach

I use LongEval-Sci 2026 (CORE scientific search): three cumulative snapshots of 0.87M, 1.32M and 1.66M documents with released Qwen3-Embedding-4B vectors, 100 training queries and 381 held-out test queries.

The method follows Stahlberg and Byrne's search-error versus model-error split: run the **same embeddings with exact search and with approximate search** at every snapshot. Exact search carries the effect of the model, documents, queries and labels. The signed ANN-minus-exact difference is the index's contribution. Following Keller et al.'s component view of temporal change, this adds a system-internal term to their environment-side decomposition. It is an accounting identity, not a causal share: the index term interacts with corpus size.

| Arm | Purpose |
| --- | --- |
| Exact float64 search | Reference with no index error |
| HNSW (two independent implementations), updated in place | Deployed graph index |
| Same HNSW, rebuilt from the identical insertion history | Isolates index history from content |
| IVF-Flat, training frozen at snapshot 1 vs refreshed | Partition index and its maintenance choice |

Every index uses three construction seeds. Search settings are chosen on calibration queries to reach 95% neighbour recall, and also held fixed, to separate drift from recalibration.

## Research questions

- **RQ1 (history).** Does keeping an index alive as the collection grows change its results compared with rebuilding from the same insertion history?
- **RQ2 (approximation over time).** How large is the signed ANN-minus-exact nDCG@10 gap at each snapshot, and does it change from snapshot 1 to snapshot 3?
- **RQ3 (failures).** Where do approximate searches fail, and what does recovering those failures cost?
- **RQ4 (repair).** Do query-driven graph repairs beat simply searching wider at equal search work?

## Findings

All findings come from 100 development training queries and from the 381 held-out test queries. Every finding is re-verified from complete logs and checksums. The held-out protocol was frozen and hashed before any test-query retrieval ran, and its analysis was run once, unchanged.

**Approximation over time (RQ2, held-out).** The analysis covers the 374 test queries that have positive GPT-OSS-120B judgments at all three snapshots. The contrast is the change in the ANN-minus-exact nDCG@10 gap from snapshot 1 to snapshot 3.

| Arm (calibrated to 95% neighbour recall) | Gap change | 95% interval | Holm p | Verdict |
| --- | --- | --- | --- | --- |
| HNSW implementation 1 | −0.0059 | [−0.0129, −0.0006] | 0.117 | inconclusive |
| HNSW implementation 2 | −0.0007 | [−0.0038, +0.0027] | 0.687 | negligible |
| IVF-Flat, frozen training | −0.0015 | [−0.0037, +0.0005] | 0.291 | negligible |

The gap itself is below 0.012 nDCG@10 at every snapshot for every arm.

The two HNSW arms differ because of calibration, not implementation. Implementation 1's calibration queries stayed just above the 95% target at a fixed search width, so its setting never changed. Meanwhile, its neighbour recall on the test queries fell from 94.5% to 93.5%, and its gap widened from −0.006 to −0.012. Implementation 2's calibration dipped below the target for two seeds, so its search widened and its gap stayed flat. At the same fixed search width, the two implementations behave alike.

The gap therefore tracks neighbour recall on the evaluated queries. GPT-OSS-20B judgments reproduce the pattern. The 29 human-judged queries show gaps within ±0.0005.

**Index history has no effect (RQ1).** Rebuilding from the same insertion history reproduces the incrementally updated index exactly, for both HNSW implementations and for IVF, with three seeds and three snapshots each. The state fingerprints, and every ranking and score on the held-out queries, are identical in all 27 comparisons. In append-only growth, "rebuild or keep updating" is a question of insertion order and construction randomness, not of accumulated drift.

**Failures depend on randomness and search coverage, not age (RQ3).** Mean neighbour recall on the held-out queries falls slowly as the collection grows: from 94.5% to 93.5% at a fixed setting. The worst decile of queries falls from 0.69 to 0.65. In development, the zero-recall queries recover on the unchanged graph with a wider search, so good candidates were dropped mid-search, not made unreachable. Refreshing IVF training does not dominate keeping it frozen.

**Graph repair does not beat wider search at equal work (RQ4, development).** Four pre-registered repair methods were screened against equal-work controls:

- resuming an exhausted search from skipped candidates;
- adding edges learned from calibration queries;
- the published NGFix/RFix repair;
- bridging answer-set fragments using encoder-generated pseudo-queries.

Each raises recall at a fixed search setting, but none beats an ordinary wider search at the same number of distance computations.

**Exact search over time.** With LLM labels, exact-search effectiveness is stable to slightly rising (0.804 → 0.811). Click-based labels fall, but they judge only 6–38% of the top 10 results, and that coverage shrinks as the collection grows. They cannot support an index comparison.

## Claim and contribution

> On a real, relevance-judged longitudinal collection, the effectiveness lost to approximate dense retrieval stays below about 0.01 nDCG@10 at every snapshot, and its change over time is negligible, provided the index is held at about 95% neighbour recall on the evaluated queries. A search setting chosen once lets the gap drift with recall, and a small calibration set can miss that drift. Under append-only growth, incremental index maintenance is exactly equivalent to rebuilding from the same history, and published-style graph repairs do not beat wider search at equal work.

The contributions are:

1. The first measurement of the ANN-minus-exact effectiveness gap over time on a relevance-judged test collection, under a pre-registered, hash-frozen held-out protocol.
2. A practical rule for longitudinal evaluations: exact-search conclusions transfer to an ANN index to within about 0.01 nDCG@10 if neighbour recall is re-checked at each snapshot.
3. An exact equivalence result for incremental maintenance versus rebuild in append-only growth.
4. A diagnosis of where approximate search fails on real asymmetric query–document embeddings, and pre-registered negative results for four repair strategies at matched search work.
5. A fully checksummed, reproducible artifact.

## Held-out evaluation protocol

The protocol was approved, then frozen with hashes of the configuration and analysis code on 19 November 2026, before any test-query retrieval ran:

- **Judgments.** The primary labels are the released GPT-OSS-120B judgments, on the fixed panel of 374 test queries with a positive judgment at all three snapshots. Click, DCTR, human and GPT-OSS-20B labels are reported separately and never pooled.
- **Primary contrast.** For each ANN arm A, D_A = [nDCG(A) − nDCG(exact)] at snapshot 3 minus the same at snapshot 1. Seeds are averaged within each query.
- **Inference.** Paired sign-randomization tests with Holm adjustment across the three arms, and paired query-bootstrap 95% intervals.
- **Practical margin.** 0.01 nDCG@10. A change is negligible only if its whole interval lies within ±0.01.

## Plan

| When | Step |
| --- | --- |
| Done (Nov 2026) | Test-query encoding, frozen protocol, exact references, held-out runs and analysis |
| Dec 2026 | Latency and search-work measurements at calibrated settings, plus figures |
| Jan–Mar 2027 | Manuscript and reproducible artifact |
| Apr–May 2027 | Submission |

**Target venue:** the CLEF 2027 conference track (Bucharest, September 2027; deadline expected around April–May). If LongEval 2027 runs, a working-notes paper as well.

## Limitations

- One collection and one embedding model. The claim is scoped to them. A second dataset, for example LongEval Web, which has deletions, would be needed for generality.
- The collection is append-only, so the equivalence result does not cover deletions or rewritten documents.
- The primary labels are LLM-generated. Human labels exist for only 29 queries, at the final snapshot only.
- Calibration uses 50 queries and a coarse grid that doubles the search width at each step. This design choice is why the two HNSW arms diverge. A finer or larger calibration is future work.
- The repair results come from development screens, two of them on a sample, and are claims about equal-work search on this data.

## References

- Lin. Operational advice for dense and sparse retrievers: HNSW, flat, or inverted indexes? 2024. <https://arxiv.org/abs/2409.06464>
- Tu, Jiang, Lin et al. Approximate nearest neighbor search and lightweight dense vector reranking in multi-stage retrieval. ICTIR 2020. <https://cs.uwaterloo.ca/~jimmylin/publications/Tu_etal_ICTIR2020.pdf>
- Stahlberg and Byrne. On NMT search errors and model errors: Cat got your tongue? EMNLP 2019. <https://aclanthology.org/D19-1331/>
- Keller, Breuer and Schaer. Evaluation of temporal change in IR test collections. ICTIR 2024. <https://arxiv.org/abs/2407.01373>
- LongEval at CLEF 2025: Longitudinal evaluation of IR systems on web and scientific data. <https://arxiv.org/abs/2509.17469>
- Extended overview of the CLEF 2026 LongEval lab. CEUR Vol-4283, paper 374. <https://ceur-ws.org/Vol-4283/paper374.pdf>
- DS@GT at LongEval: Evaluating temporal performance in web search systems and topics with two-stage retrieval. <https://arxiv.org/abs/2507.08360>
- Lawrie et al. PLAID SHIRTTT for large-scale streaming dense retrieval. SIGIR 2024. <https://arxiv.org/abs/2405.00975>
- Yang et al. MURR: Model updating with regularized replay for searching a document stream. 2025. <https://arxiv.org/abs/2504.10250>
- Saravanan, Salla and Amancherla. AURORA. Findings of ACL 2026. <https://aclanthology.org/2026.findings-acl.495/>
- Singh et al. FreshDiskANN. 2021. <https://arxiv.org/abs/2105.09613>
- Xu et al. SPFresh. SOSP 2023. <https://arxiv.org/abs/2410.14452>
- Mohoney et al. Quake: Adaptive indexing for vector search. OSDI 2025. <https://arxiv.org/abs/2506.03437>
- Baranchuk et al. DeDrift: Robust similarity search under content drift. ICCV 2023. <https://arxiv.org/abs/2308.02752>
- Mandarapu and Kunkunuru. When to repair. 2026. <https://arxiv.org/abs/2607.00728>
- Kuissi et al. Still fresh? 2026. <https://arxiv.org/abs/2603.04532>
- NGFix/RFix: Dynamically detect and fix hardness for efficient approximate nearest neighbor search. 2025. <https://arxiv.org/abs/2510.22316>
- Chen et al. RoarGraph. VLDB 2024. <https://arxiv.org/abs/2408.08933>
- Malkov and Yashunin. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI, 2020.
