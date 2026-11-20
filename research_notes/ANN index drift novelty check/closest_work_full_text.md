# Closest prior works to "ANN index drift in LongEval": full-text verification

How the texts were read (2026-11-12):
- AURORA (ACL Anthology PDF) and the JISEM ETL paper (journal PDF): downloaded, converted to text with pdftotext, and read directly. The quotes are verbatim from the extracted text.
- PLAID SHIRTTT, Streaming RAG, ReproRAG, Quake, Still Fresh, When-to-Repair: the arXiv HTML full text was read through WebFetch, which passes each page through a summarizing model. The quotes are what that tool returned when asked for verbatim text. They match the papers' wording as far as I could check, but I did not compare them line by line against the PDFs. For Quake, only the first 100k of 114k characters were processed.
- BANCO: **full text not accessible.** dl.acm.org returned HTTP 403 for both the landing page and the PDF, and I found no arXiv version. The notes rely on the Crossref and Semantic Scholar metadata and the abstract only.

The proposal has three parts: (1) a per-snapshot decomposition of relevance-labelled nDCG into an exact-search part and an index part (exact minus ANN), run on real time-ordered LongEval snapshots; (2) rebuild vs incremental vs in-place maintenance (recalibrating effort, refitting the quantizer, repairing the graph), judged by relevance effectiveness; (3) a label-free query-drift signal (query-to-NN distance divided by row-to-NN distance).

## Q1. PLAID SHIRTTT (Lawrie, Kayi, Yang, Mayfield, Oard, SIGIR 2024, arXiv 2405.00975)

### Takeaway
This is the closest work on the "index fit on early data degrades relevance effectiveness over a date-ordered stream" side. It **partially overlaps** with parts (1) and (2). It shows that centroids fit early collapse nDCG, and it compares its streaming sharded index against an oracle that is fit after the fact. It does not compare against exact search, it does not decompose the change per snapshot, it does not study in-place maintenance strategies, and it has no label-free signal.

### Cited Findings
- Index: ColBERT-PLAID, which "represent[s] terms as cluster centroids plus compressed residual vectors", with "one bit for each residual vector dimension". ColBERTv2 is used for ClueWeb09 and ColBERT-X (XLM-R Large) for NeuCLIR. The paper does not state the total number of centroids. — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)
- Time structure: the stream is simulated by date. Documents are "simulated as a stream by ordering...by date", using ClueWeb09 (504M docs) and NeuCLIR (10M docs). These are real document dates, but the paper uses one collection with one fixed set of qrels. It does not use evolving snapshots, each with its own judgements. — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)
- Key motivating result (Table 1): "If centroids are generated from the first 5% of the chronologically ordered Chinese documents in the NeuCLIR test collection... nDCG@20 drops precipitously from 0.440 to 0.047." — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)
- Maintenance: hierarchical sharding in which documents are "indexed a fixed, small number of times". Each shard gets its own centroids: "Each shard model is the set of dense vectors representing the cluster centroids of the document terms in that shard." The shard parameters are A=5M/B=500K for ClueWeb09 and A=500K/B=100K for NeuCLIR. — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)
- Reference: an "Oracle Shard Model, which creates a single shard model for all of ClueWeb09". This oracle is still a PLAID (approximate, compressed) index, not exact search. Result: "PLAID SHIRTTT achieves 96% of its nDCG'@20 performance" (0.431 on Web09-12, vs BM25 0.323). On NeuCLIR22 it reaches nDCG@20 0.381, vs PSQ+HMM 0.315. — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)
- Metrics are relevance-labelled: nDCG'@20 and MAP' on the TREC Web Track 2009–2012 ("all 200 topics"), and nDCG@20, MAP and R@100/1000 on NeuCLIR 2022/2023 (41 and 65 topics). — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)
- Costs: indexing took "approximately 71 days on an NVidia V100 GPU". Query latency is 1.3 s per query on 108 CPUs, and the index is 8.4 TB vs 0.4 TB for BM25. — [arXiv 2405.00975](https://arxiv.org/html/2405.00975)

### Inferences
- This paper is the strongest precedent showing that stale quantization or centroids can cost relevance effectiveness on a time-ordered stream. The proposal should cite it as the motivation for part (2), and should present its own contribution as: exact-search attribution per snapshot + multiple LongEval snapshots with their own qrels + cheap in-place maintenance options + a label-free trigger.
- PLAID's oracle mixes centroid staleness with ordinary PLAID approximation error. Separating those two is exactly the gap that the proposal's exact-minus-ANN term fills.

### Gaps
- The number of centroids and the PLAID search parameters (ncells, centroid thresholds) were not found in the parts that were processed.

## Q2. Streaming RAG (Yuzhou Zhu, arXiv 2508.05662, July 2025)

### Takeaway
**Different.** The main index is exact (Faiss IndexFlatIP), so it has no ANN error. "Full Rebuild" is a baseline policy for selecting what goes into the knowledge base, not a study of ANN maintenance. How relevance labels are obtained on the streams is not explained.

### Cited Findings
- Title: "From Static to Dynamic: A Streaming RAG Approach to Real-time Knowledge Base". Author: Yuzhou Zhu (Dalian University of Technology). arXiv cs.IR, July 31, 2025. — [arXiv 2508.05662](https://arxiv.org/html/2508.05662)
- Index: "Batched index updates operate on a Faiss IndexFlatIP, executed every 1,000 arrivals to maintain query availability." A "Faiss IVFPQ Incremental" comparator is also listed. — [arXiv 2508.05662](https://arxiv.org/html/2508.05662)
- Data: eight streams (NYT, synthetic Poisson, Twitter, IoT, Reddit, Wikimedia edits, NASDAQ, Bitcoin mempool). Baselines: Static RAG, Full Rebuild ("reconstruct[s] the entire index after each batch"), Reservoir Sampling, Heap Filtering, SAKR. — [arXiv 2508.05662](https://arxiv.org/html/2508.05662)
- Metrics: Recall@K, MRR and nDCG@10, plus EM/F1 on SQuAD and ROUGE-L. On NYT, Recall@10 is 0.540±0.006 for Full Rebuild vs 0.580±0.004 for Streaming RAG. QA gains are "+3.2 EM, +2.8 F1" over Static RAG. Latency is under 15 ms. — [arXiv 2508.05662](https://arxiv.org/html/2508.05662)
- When asked, the fetch tool reported that the paper "does not explicitly define how relevance labels are obtained" for the streams and does not separate index approximation error from drift. — [arXiv 2508.05662](https://arxiv.org/html/2508.05662)

### Inferences
- In this paper "rebuild" means recomputing which items the memory-bounded knowledge base keeps (sketching or clustering), not rebuilding an ANN graph or quantizer. It does not cover part (2) as the proposal defines it.

### Gaps
- Exactly how the IVFPQ comparator was configured, and the ground-truth definition for the streams, were not determined.

## Q3. AURORA (Saravanan, Salla, Amancherla; Findings of ACL 2026, pp. 10179–10195)

### Takeaway
**Partially overlaps with part (2) only, in spirit.** It adapts the quantizer and codebook of an HNSW index while keeping the graph fixed, instead of rebuilding. Its drift is entirely synthetic (topic partitions, token substitution, SIFT rotation). It has no time-stamped snapshots, no decomposition against exact search, and no label-free distance-ratio signal. The authors themselves name longitudinal time-stamped evaluation as future work.

### Cited Findings
- Dense design (verbatim): "AURORA builds the HNSW graph once on base document embeddings and thereafter keeps the graph topology fixed. When drift is observed, adaptation updates only the neural codebook and residual quantizer parameters on a small support set, leaving graph edges unchanged." The quantizer is a residual neural quantizer (d=384, all-MiniLM-L6-v2, M=8 subspaces), built on FAISS 1.7.4. — [ACL Anthology 2026.findings-acl.495](https://aclanthology.org/2026.findings-acl.495.pdf)
- The data is synthetic (verbatim): "we design controlled distribution shift protocols ... Rather than modeling natural language evolution exhaustively, these protocols serve as stress tests". Task 1 splits AG News and SQuAD "into disjoint sets Cold ... and Cnew ... based on latent semantic clusters", scored by Recall@10. Task 2 is NFCorpus with token replacement ("Cancer" → "Glipglop"), scored by NDCG@10. Task 3 is "SIFT1M ... subject to continuous manifold rotation". — [ACL Anthology](https://aclanthology.org/2026.findings-acl.495.pdf)
- The queries for the generative task are also synthetic: "We generate synthetic queries targeting Dnew by sampling documents d ∈ Dnew and applying Gaussian noise". — [ACL Anthology](https://aclanthology.org/2026.findings-acl.495.pdf)
- Admitted limitations (verbatim): "Evaluating continual retrieval adaptation on naturally evolving, time-stamped corpora is an important direction for future work." Also: "longitudinal evaluation on time-stamped corpora is needed to validate performance on organic semantic shift." — [ACL Anthology](https://aclanthology.org/2026.findings-acl.495.pdf)
- Key numbers:
  - SIFT1M: OOD recall goes "from 99.5% to 73.0%" (Fig. 1).
  - "+26.9% Recall@10 on novel topics".
  - Adaptation takes 28.2 ms vs 5149.5 ms for full retraining (182×).
  - NFCorpus NDCG@10: AURORA-Sparse 0.5638 = Pure Sparse 0.5638, Pure Dense 0.5093.
  - Ablation Recall@10: static 0.289 vs full AURORA 0.413.
  - Baselines are Static HNSW, Pure DSI, Naive Fine-Tuning and Static Hybrid. No incremental-insert or graph-repair baseline is used.

  — [ACL Anthology](https://aclanthology.org/2026.findings-acl.495.pdf)
- Its drift-handling signals are uncertainty features used by a Bayesian router (entropy, norm, dim, confidence), trained with reward. The paper notes that proxy rewards are "noisy and sparse". — [ACL Anthology](https://aclanthology.org/2026.findings-acl.495.pdf)

### Inferences
- AURORA is the clearest precedent for the "refit the quantizer in place instead of rebuilding" idea. The proposal's points of difference are: real LongEval snapshots with qrels, attribution against exact search, and comparing several cheap maintenance options under one budget.

### Gaps
- None material. The full text was read.

## Q4. ReproRAG (Wang, Zhao, Tallent, Guo, arXiv 2509.18869, Sept 2025)

### Takeaway
**Different.** It studies reproducibility (determinism) of retrieval across precision formats, hardware and index types on a static MS MARCO subset. The "temporal" test inserts data and checks rank stability, not effectiveness against relevance labels.

### Cited Findings
- It tests the FAISS indexes Flat_L2/Flat_IP (exact), IVF, HNSW and LSH, with the embedding models BGE-base, E5-base and Qwen3-0.6B, on MS MARCO (100k docs, 1k queries). — [arXiv 2509.18869](https://arxiv.org/html/2509.18869)
- The metrics are agreement metrics, not relevance metrics: EMR, Jaccard, Kendall's τ, RBO, L2/cosine distance and score std. — [arXiv 2509.18869](https://arxiv.org/html/2509.18869)
- "Temporal reproducibility under incremental data insertion" gave a "perfect Kendall's Tau score of 1.000" for results that persisted. Cross-model Jaccard was 0.43–0.54. The L2 distance between FP32 and FP16 embeddings was 5.74e-04. Single-node CPU/GPU runs reproduced perfectly (1.00). — [arXiv 2509.18869](https://arxiv.org/html/2509.18869)

### Inferences
- It is useful only as a citation that ANN results are deterministic given the index, which supports treating exact-minus-ANN as a stable quantity.

### Gaps
- None material.

## Q5. "Measuring Retrieval Freshness and Accuracy Degradation in Continuous ETL-Driven RAG Systems"

### Takeaway
**Different.** It is a single-author narrative review with no experiments of its own. Its numbers are restated from CRAG, HOH, RAGBench and Streaming RAG. The "+3 points Recall@10" it reports is the 2508.05662 result.

### Cited Findings
- Citation: Deepika Annam, *Journal of Information Systems Engineering and Management* 11(2s), published Feb 13, 2026. DOI [10.52783/jisem.v11i2s.14454](https://doi.org/10.52783/jisem.v11i2s.14454). — [JISEM article page](https://jisem-journal.com/index.php/journal/article/view/14454)
- Its claims are sourced from other works, e.g. "Controlled experiments show that freshness-aware indexing yields Recall@10 improvement of up to 3 absolute points over static indexing baselines, with statistical significance at p less than 0.01 [6]". This matches Streaming RAG's own wording. — [JISEM PDF](https://jisem-journal.com/index.php/journal/article/view/14454/6951)
- The abstract claims that outdated information "can render retrieval augmentation counterproductive" and that hybrid retrieval is more resilient to staleness. Both rest on cited benchmarks (HOH, CRAG). — [JISEM article page](https://jisem-journal.com/index.php/journal/article/view/14454)

### Inferences
- It is safe to describe this as a low-weight review from an open-access journal. It does not contradict any novelty claim.

### Gaps
- I did not establish the author's affiliation or whether the journal is peer-reviewed.

## Q6. BANCO (Cheng, Dai, Ding, Lui, Huang; WWW 2026, pp. 2183–2194)

### Takeaway
**Probably different, judged from the abstract only (full text not accessible).** It is an adaptive pruning policy for proximity graphs that aims to keep ANN efficiency and accuracy as the data and queries evolve. By its description, accuracy means ANN accuracy. There is no sign of relevance labels, a decomposition against exact search, or time-ordered IR test collections.

### Cited Findings
- Citation: "BANCO: Drift-Aware Batched Bandits for Adaptive Proximity Graph Pruning", Proceedings of the ACM Web Conference 2026, pp. 2183–2194, DOI [10.1145/3774904.3792396](https://dl.acm.org/doi/10.1145/3774904.3792396). Semantic Scholar lists it as open access under CC BY-NC-ND. — [Crossref](https://api.crossref.org/works/10.1145/3774904.3792396); [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3774904.3792396)
- Abstract (verbatim): "Sustaining long-term performance requires adaptive pruning as data and query workloads evolve. ... Adaptive pruning faces three key challenges: temporal drift in data and query distributions, spatial heterogeneity across graph regions, and costly feedback due to graph-level evaluations. ... BANCO unifies diverse pruning strategies within a common decision space and optimizes them via a drift-aware batched bandit algorithm. ... Extensive experiments on four real-world datasets demonstrate that BANCO helps maintain long-term ANN search efficiency and accuracy under evolving data and workloads." — [Semantic Scholar](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1145/3774904.3792396)

### Inferences
- "Four real-world datasets" most likely means standard ANN vector sets with simulated evolution. That is unverified. It is the nearest precedent for "graph repair or pruning under drift" in part (2), but it is framed as an optimisation policy, not as attribution.

### Gaps
- Full text was blocked (HTTP 403 from dl.acm.org, no arXiv copy found). Unknown: the datasets, whether the drift is real or synthetic, the reward and metric definitions, and whether any label-free drift statistic is used. Recommend downloading the PDF manually from the ACM DL in a browser.

## Q7. Quake (Mohoney et al., OSDI 2025, arXiv 2506.03437): Wikipedia-12M workload

### Takeaway
**Partially overlaps with parts (2) and (3), in the systems sense.** The workload is genuinely time-ordered: real monthly Wikipedia additions and page views from April 2013 to December 2021. However, the queries are the page embeddings themselves, sampled by page views. The metric is neighbour recall against exact ground truth, not relevance-labelled effectiveness. The label-free APS mechanism estimates recall geometrically to set nprobe. That is close in spirit to the proposal's recalibration and drift signal, but it is a different statistic and has a different purpose.

### Cited Findings
- Workload (verbatim): "This dataset and workload trace are derived from monthly Wikipedia page additions and page-view frequencies between April 2013 and December 2021." It grows from 1.6M to 12M vectors over 103 monthly updates (about 100k each). The paper adds: "The workload simulates monthly inserts of new pages, followed by 100,000 search queries sampling page embeddings with probability proportional to their page views". — [arXiv 2506.03437](https://arxiv.org/html/2506.03437)
- Embeddings: "generated by training DistMult graph embeddings ... on the Wikipedia link structure", using inner product. They are trained once, not per snapshot (per the fetch tool's reading). — [arXiv 2506.03437](https://arxiv.org/html/2506.03437)
- Index: a single-level partitioned IVF with sqrt(|X0|) partitions. Metric: "Recall@k ... |G∩R|/k where R is the vectors returned by the approximate search, and G is the ground truth set", with a target of 90% recall@100. — [arXiv 2506.03437](https://arxiv.org/html/2506.03437)
- APS (label-free): "compute probabilities p_i for each remaining candidate partitions based on radius ρ. Iteratively scan partitions in descending probability order until cumulative recall exceeds target τ_R." — [arXiv 2506.03437](https://arxiv.org/html/2506.03437)
- Wikipedia-12M total time in hours, as search / update / maintenance (total):

  | Method | Search | Update | Maintenance | Total |
  |---|---|---|---|---|
  | Quake-MT | 1.53 | 0.01 | 0.44 | 1.98 |
  | Faiss-IVF (no maintenance) | 165.8+ | | | |
  | DeDrift | 132.6 | | | |
  | LIRE | 44.2 | | | |
  | Faiss-HNSW | 14.65 | | | 14.83 |

  Recall was "stable" under Quake. — [arXiv 2506.03437](https://arxiv.org/html/2506.03437)

### Inferences
- Quake is the strongest systems precedent showing that a no-maintenance IVF degrades over a real monthly timeline, and it compares maintenance strategies (Quake, LIRE, DeDrift, none, plus graph indexes). It measures cost at a fixed neighbour-recall target, not relevance effectiveness. It has no qrels and no split of effectiveness change into model or data change versus index change.

### Gaps
- The last ~14k characters of the HTML were not processed. Per-month recall curves, and exactly how ground truth is recomputed per snapshot, were not confirmed verbatim.

## Q8. "Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks" (Kuissi, Subrahmanyan, Thakur, Lin; arXiv 2603.04532, Mar 2026)

### Takeaway
**Partially overlaps only with the "temporal drift in relevance-labelled evaluation" framing.** It uses real snapshots (Oct 2024 vs Oct 2025, FreshStack/LangChain) with LLM-judged nugget relevance. It does not mention any ANN index, does not separate index error, and does not study maintenance.

### Cited Findings
- Setup: BM25 plus about 15 dense embedders (Qwen3, Stella, E5-Mistral, BGE-Gemma2, etc.) and score fusion. The documents are "chunk[ed] ... with a maximum of 2048 tokens". The paper reports that it collects "the top 50 documents using fusion". The fetch tool found no explicit statement of exact vs approximate search. — [arXiv 2603.04532](https://arxiv.org/html/2603.04532)
- Data: "two independent corpus snapshots of ten GitHub repositories" (Oct 2024 and Oct 2025), with 203 Stack Overflow queries and 640 nuggets. Relevance is judged by Cohere Command A. — [arXiv 2603.04532](https://arxiv.org/html/2603.04532)
- Numbers: Qwen3-4B scores α-nDCG@10 0.501 → 0.482 and Recall@50 0.436 → 0.437. "we generally observe a decrease in scores between the 2024 and 2025 snapshots for the majority of retrieval models". Kendall τ is 0.978 at Recall@50 and 0.692 at Coverage@20. — [arXiv 2603.04532](https://arxiv.org/html/2603.04532)

### Inferences
- The paper attributes score changes to the corpus and to relevance shift, which implicitly assumes that retrieval itself is exact. The proposal's exact-minus-ANN term tests exactly that assumption. This is a good related-work contrast.

### Gaps
- Whether the dense retrieval used flat or ANN search was not stated in the text that was processed.

## Q9. "When to Repair a Graph ANN Index" (Mandarapu & Kunkunuru, VaidhyaMegha; arXiv 2607.00728v2, July 2026)

### Takeaway
**Partially overlaps with part (2) (graph repair) and loosely with part (3) (label-free trigger), but in a different setting.** It runs Vamana on SIFT and Fashion-MNIST with **synthetic** bursty insert/delete churn, measures neighbour recall against an exact oracle on the live set, and uses a probe-recall trigger, which needs exact ground truth on probe queries. Its result is negative: at a matched budget, triggered repair does not beat repair on a fixed cadence.

### Cited Findings
- Churn (verbatim): "A bursty stream that alternates delete-heavy bursts with insert-heavy calm, so drift is bursty rather than smooth". This is generated churn on SIFT-128 and Fashion-MNIST-784, with about 10× turnover, not real timestamps. An earlier fetch summary called it "real time-ordered". That label is wrong. — [arXiv 2607.00728](https://arxiv.org/html/2607.00728)
- Index: Vamana (DiskANN in-memory, v0.7.0), R ∈ {16, 24, 32}, with 20k and 100k live points. — [arXiv 2607.00728](https://arxiv.org/html/2607.00728)
- Oracle (verbatim): "recall@10 against an exact brute-force oracle recomputed on the current live set after each window—never against frozen initial ground truth". No nDCG, relevance labels, quantization or IVF appear in the paper. — [arXiv 2607.00728](https://arxiv.org/html/2607.00728)
- Policies compared: P0 lazy tombstoning, P1 consolidation on a fixed cadence, and P2 consolidation triggered by a signal. The signal (verbatim) is: "The signal s(t) is the recall@10 on a small, held-out probe query set (disjoint from the evaluation set), recomputed cheaply each window". It is free of relevance labels but requires exact-NN ground truth for the probes. — [arXiv 2607.00728](https://arxiv.org/html/2607.00728)
- Finding (verbatim): "At matched consolidation count, signal-triggered local repair does not beat fixed-cadence repair on tail recall—on either dataset, at any of three graph degrees, or at a 5× larger index." The authors withdraw v1's claimed +0.014 to +0.050 min-recall gains as an "interpolated baseline trap". — [arXiv 2607.00728](https://arxiv.org/html/2607.00728)

### Inferences
- This is a direct warning for part (3): a drift trigger must beat a fixed schedule at a matched maintenance budget, and the baselines must be measured, not interpolated. The proposal should add a fixed-cadence baseline at a matched budget.

### Gaps
- None material for the verdict.

## Overall verdict summary

| Work | Real time-ordered snapshots with qrels? | Exact-search reference? | Maintenance compared | Label-free drift signal | Verdict |
|---|---|---|---|---|---|
| PLAID SHIRTTT | Date-ordered stream, one fixed qrel set | No (oracle is still PLAID) | Hierarchical shards vs early-centroid vs oracle | No | Partial: (1) motivation, (2) |
| Streaming RAG 2508.05662 | Real streams, labels unclear | Index is exact Flat | Full rebuild of memory vs sketch | No | Different |
| AURORA | No (synthetic) | No | Refit quantizer in place vs retrain | Router uncertainty only | Partial: (2) idea only |
| ReproRAG | No | Flat used as baseline | No (determinism) | No | Different |
| JISEM ETL (Annam) | No own experiments | No | No | No | Different (review) |
| BANCO | Unknown (abstract only) | Unknown | Adaptive pruning policy | Unknown | Likely different / partial on (2) |
| Quake | Real monthly Wikipedia, no qrels | Yes (NN recall) | Quake / LIRE / DeDrift / none / HNSW | APS geometric recall estimate | Partial: (2), (3) systems-side |
| Still Fresh? | Two real snapshots, LLM qrels | Not stated | No | No | Partial: framing only |
| When to Repair | No (synthetic churn) | Yes (NN recall) | Tombstone / fixed / triggered | Probe recall (needs exact GT) | Partial: (2) graph, (3) caveat |

None of these works subsumes the proposal. None of them combines the three elements: real snapshots with relevance labels, a per-snapshot decomposition of nDCG into exact and index parts, and maintenance compared by relevance effectiveness. — Sources as cited above.
