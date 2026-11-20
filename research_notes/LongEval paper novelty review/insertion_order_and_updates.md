# Insertion order, incremental updates, and late-inserted point reachability in graph/IVF ANN indexes (prior-work novelty check)

## Q1. What exactly do Elliott and Clark (2024) find about insertion order? Do they measure per-point recall by insertion position?

### Takeaway
Elliott & Clark show that HNSW *aggregate* recall@10 depends on insertion order (by local intrinsic dimensionality, LID, and by product category), by up to 12.8 points at efSearch=10. They do NOT measure recall per point by insertion position, do NOT study temporal/snapshot growth, and do NOT test whether later-inserted points are retrieved less. This is the paper a reviewer will cite most readily for "order matters", but it does not cover claim (3)'s age-conditioned miss rate.

### Cited Findings
- Bibliographic: Owen P. Elliott and Jesse Clark (Marqo), "The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds", arXiv:2405.17813, cs.IR, submitted 28 May 2024, marked "A Preprint"; no peer-reviewed venue given on the arXiv page — [arXiv](https://arxiv.org/abs/2405.17813)
- Abstract: recall "is linked to the vector space's intrinsic dimensionality and significantly influenced by the data insertion sequence"; insertion order "can shift recall by up to 12 percentage points" — [arXiv HTML](https://arxiv.org/html/2405.17813v1)
- Orders tested: descending LID, ascending LID, random, and category-sequential ("Data was indexed sequentially organized by product categories as listed on the retailers' websites"; "data within each category is randomly ordered") — [arXiv HTML](https://arxiv.org/html/2405.17813v1)
- Mechanism they propose: ascending-LID graphs "suffer from early establishment of tight localities"; descending order is "similar to simulated annealing"; "A random insertion order is consistently in between the ascending and descending LID insertion orders." — [arXiv HTML](https://arxiv.org/html/2405.17813v1)
- Numbers: largest gap "12.8 percentage points on FAISS (all-MiniLM-L6-v2)" at efSearch=10; descending LID gives "2.6 and 6.2 percentage points better recall" (hnswlib and FAISS); category order on fashion data: "for ViT-B-32, the recall difference attributable to varied insertion sequences reached up to 7.7 percentage points"; homewares effect smaller; model rankings shift "by up to three places" — [arXiv HTML](https://arxiv.org/html/2405.17813v1)
- Setup: hnswlib and FAISS, M=16, efConstruction=128, efSearch=10 (and 40); MTEB-derived retrieval datasets and CLIP product images — [arXiv HTML](https://arxiv.org/html/2405.17813v1)
- Conclusion: "The effect of insertion order for data into the graphs has real world impacts", speculating about new product categories / domain shift arriving over time, but not experimentally tested — [arXiv HTML](https://arxiv.org/html/2405.17813v1)
- Note: abstract says "up to 12" points while body says 12.8; secondary summaries disagree on direction framing — [arXiv HTML](https://arxiv.org/html/2405.17813v1); [EmergentMind summary](https://emergentmind.com/papers/2405.17813)
- Follow-up citing it: Dual-Branch HNSW with LID-driven insertion (arXiv:2501.13992) uses LID-based ordering as a design input — [arXiv](https://arxiv.org/html/2501.13992v2)

### Inferences
- Reviewers will say "insertion order affects HNSW recall is known (Elliott & Clark 2024)". The defensible novelty is: (a) the order is the *natural temporal order of a growing test collection* (LongEval snapshots), not an artificial LID/category sort; (b) the effect is measured *per document, conditioned on insertion snapshot at the same exact rank*; (c) a shuffled rebuild of the identical set isolates order from content; (d) impact is traced to IR effectiveness (nDCG) for queries whose relevant documents are new.
- Elliott & Clark's category-sequential order is the closest analog to snapshot-sequential arrival (clustered arrival). Their finding that random sits between extremes is consistent with, but different from, the claim that a shuffle removes most of the late-snapshot excess.
- Their finding of large aggregate effects at efSearch=10 shrinking at 40 is consistent with the claim that wider ef, not reordering, is the effective remedy. Cite it as support, not as a contradiction.

### Gaps
- Could not verify whether a later version or peer-reviewed venue exists (only v1 seen).
- No per-point or per-insertion-position analysis in the paper (checked via full HTML read).

## Q2. Unreachable / poorly connected points after updates; do late-inserted nodes end up under-connected?

### Takeaway
The "unreachable points" literature (MN-RU 2024 and follow-ups) is about *deletions/replacements*, and explicitly says unreachable points from construction alone are "exceedingly small" in number. Streaming-index papers (FreshDiskANN, Big-ANN'23 streaming track, Ada-IVF, DeDrift) measure aggregate recall over time, mainly from deletes or distribution drift. I found no peer-reviewed paper that measures recall of append-only inserts by insertion time. Only unreviewed GitHub issues show that clustered/sorted insertion disconnects or hurts later inserts.

### Cited Findings
- MN-RU: Xiao, Zhan, Xi, Hou, Liao (UESTC), "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation", arXiv:2407.07871 (v2), July 2024; no peer-reviewed venue listed on arXiv — [arXiv](https://arxiv.org/html/2407.07871v2)
- Definition: "An unreachable point is defined as a point that possesses outgoing edges but lacks incoming edges in all layers of the navigable small-world graph." — [arXiv](https://arxiv.org/html/2407.07871v2)
- Cause: hnswlib's replaced_update / deletion repair, which prunes edges and leaves some nodes with in-degree 0. On pure construction: "the likelihood of forming unreachable points is exceedingly small." — [arXiv](https://arxiv.org/html/2407.07871v2)
- Numbers: on SIFT with 5% delete+reinsert per iteration over about 3000 iterations, unreachable points reach 3–4% and recall drops about 3%; HNSW-RU after 200 iterations is about 3–4% on GIST and 2–3% on ImageNet; MN-RU is 2–4x faster to update — [arXiv](https://arxiv.org/html/2407.07871v2)
- Follow-up: Sengupta, LR-RN neighborhood repair for dynamic product-catalog HNSW, CEUR-WS Vol-4250 (not read in full; unverified details) — [CEUR-WS](https://ceur-ws.org/Vol-4250/paper776.pdf)
- FreshDiskANN (Singh, Subramanya, Krishnaswamy, Simhadri; arXiv:2105.09613, 2021): streaming graph index whose recall is tracked over days of concurrent inserts, deletes and searches. It reports aggregate recall stability, not recall by insertion age (abstract-level check only) — [arXiv](https://arxiv.org/abs/2105.09613)
- NeurIPS'23 Big-ANN streaming track: runbooks interleave inserts, deletes and queries and score aggregate recall — [Results paper, arXiv:2409.17424](https://arxiv.org/html/2409.17424v1); also in [NeurIPS 2025 D&B](https://papers.nips.cc/paper_files/paper/2025/hash/63092d79154adebd7305dfd498cbff70-Abstract-Datasets_and_Benchmarks_Track.html)
- GrAND (arXiv:2608.21163, unreviewed preprint): recall falls as tombstones accumulate; FreshDiskANN-GPU loses 21% recall on a MSTuring-100M sliding window (with consolidation disabled, so not like-for-like) — [arXiv](https://arxiv.org/pdf/2608.21163)
- Unreviewed issue (laurus #1241): clustered vectors inserted cluster-by-cluster give recall 0.9248 vs 0.9804 shuffled (nearest-only neighbor selection); the diversity heuristic (HNSW Algorithm 4) restores 0.9960 — [GitHub](https://github.com/mosuka/laurus/issues/1241)
- Unreviewed issue (grafeo #391): "nearest-only backlink pruning disconnects inserts behind a >m cluster" — [GitHub](https://github.com/GrafeoDB/grafeo/issues/391)
- Slipstream (arXiv:2606.02992, 2026) exploits stream locality to speed up insertion ("the newly arrived point starts from promising candidates found during the previous insertion"). Its abstract does not report later inserts being less reachable — [arXiv](https://arxiv.org/abs/2606.02992)
- Not checked (named in the brief): IP-DiskANN, "Deletions in graph indexes", LSH-APG. Treat any claims about them as unverified.

### Inferences
- The MN-RU paper is a direct counterpoint a reviewer may raise ("unreachable points are known"), but it supports the paper's framing: under append-only growth, unreachability is not the mechanism. Under append-only growth, the late-snapshot miss excess should be framed as *soft* under-connection (lower in-degree or greater distance from hubs, ef-sensitive), not hard unreachability. Checking in-degree-0 counts in the actual indexes would preempt the objection.
- The GitHub issues are the closest practitioner evidence that sorted/clustered arrival hurts later inserts and that shuffling fixes it. They are unreviewed and use non-standard (nearest-only) pruning, so cite them, if at all, as anecdotal.

### Gaps
- No study found that slices recall by insertion time or snapshot for append-only HNSW/IVF.
- IP-DiskANN, LSH-APG and "Deletions in graph indexes" were not read; their relevance is unverified.

## Q3. Hubness and in-degree: are early-inserted nodes hubs?

### Takeaway
The hub-highway paper (Munyampirwa, Lakshman, Coleman; ICML 2025 workshop / arXiv 2412.01940) directly tested the "early-inserted nodes become hubs" (preferential attachment) explanation. It found only a "modest" insertion-order effect, explaining about 0.1–0.5% of variance in node access counts on real data (about 25% on synthetic angular data). Hubness is attributed mainly to the metric space. This is the most adversarial citation against an "early nodes are hubs" story, so claim (3) should not rest on a hub-by-age mechanism unless it is measured.

### Cited Findings
- Paper: "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'", arXiv:2412.01940 (v1 Dec 2024, v3 Jul 2025); listed among ICML 2025 orals (Workshop on Vector Databases, per search snippet) — [arXiv](https://arxiv.org/abs/2412.01940); [ICML](https://icml.cc/virtual/2025/48952)
- Appendix F.3: preferential attachment is considered "since NSW graphs are built incrementally by sequentially adding points to an existing graph"; "Nodes that are added early in graph construction may become hubs"; result: "We observe a modest effect from the node insertion order in our synthetic data." On real datasets, insertion order explains about 0.1–0.5% of the variance in node access counts — [arXiv HTML v3](https://arxiv.org/html/2412.01940v3)
- Hubs are defined by search visits, not in-degree: "We identify hub nodes as those that fall into the top percentile of the empirical node access distribution" — [arXiv HTML v3](https://arxiv.org/html/2412.01940v3)
- Original NSW construction (Malkov et al.): incremental insertion "in random order by adding connections to the closest neighbors" produces a navigable small world with an exponential degree distribution (homophily-network paper, PLoS ONE / arXiv:1507.06529) — [arXiv](https://arxiv.org/abs/1507.06529)
- Ponomarenko 2026, "Greedy Routing in a Sequentially Grown One-Dimensional Random Graph" (arXiv:2604.19733, math.CO): random-order insertion, each new vertex links to its nearest inserted neighbors; route length is tied to the insertion permutation (left-to-right minima). It does not explicitly state that late vertices are harder to reach — [arXiv](https://arxiv.org/abs/2604.19733)
- UNVERIFIED (recollection, could not extract the PDF text): Malkov & Yashunin (TPAMI 2018, arXiv:1603.09320) motivate the hierarchy partly by noting that in NSW the long-range links come from elements inserted early in construction, and that the degree/structure depends on insertion. Verify the exact wording before quoting — [arXiv](https://arxiv.org/abs/1603.09320)

### Inferences
- Insertion-order effects on *which nodes are hubs* are weak on real data (Munyampirwa et al.). That makes an observed 1.5–1.6x late-snapshot miss excess, removed by shuffling, somewhat surprising. Snapshot order is not random: it is temporally clustered, which is the Elliott & Clark category-order regime, not the random-order regime Munyampirwa et al. examined. Framing it as "temporally clustered arrival", not "age per se", aligns with both papers.
- Expect a reviewer to ask for in-degree / access-count distributions by snapshot. Reporting them would turn the hub paper from a threat into a supporting citation.

### Gaps
- No paper found that reports in-degree distribution by insertion time for HNSW on real data.
- The Malkov & Yashunin passage on early long-range links is unverified.

## Q4. Per-item fairness / retrievability / exposure of ANN indexes by item age (recsys freshness, streaming vector DBs)

### Takeaway
I found no study measuring whether ANN indexes (HNSW or IVF) retrieve newer items less often than older ones at equal exact rank. Recsys cold-start work attributes new-item under-exposure to embeddings and interaction data, not to the index. Drift work (DeDrift) measures aggregate degradation from distribution shift. The age-conditioned, index-induced miss rate appears to be an open gap. This is a plausible novelty claim, but it rests on absence of evidence after a limited search.

### Cited Findings
- DeDrift (Baranchuk, Douze, Upadhyay, Yalniz; ICCV 2023, arXiv:2308.02752): "Unless a costly index reconstruction is performed frequently, content drift degrades the search accuracy and efficiency"; "The degradation is especially severe since, in general, both the query and database distributions change"; DeDrift "almost eliminates the accuracy degradation" while "up to 100x faster than a full index reconstruction". This is IVF/quantizer drift on timestamped image/video data, and the abstract does not report recall split by item age — [arXiv](https://arxiv.org/abs/2308.02752)
- Ada-IVF (Mohoney et al., arXiv:2411.00970, Nov 2024): incremental IVF maintenance; static-built indexes lose quality as data changes; reconstruction error and partition imbalance are the quality indicators; about 2x average (up to 5x) update throughput — [arXiv](https://arxiv.org/html/2411.00970v1)
- Recsys cold-start sources (mostly practitioner/blog) attribute missed new items to the embedding and interaction pipeline and mitigate it with fresh-item candidate pools. None measures index-induced age bias — [Robust training objectives for EBR, arXiv:2409.14682](https://arxiv.org/html/2409.14682v1)
- Dense-retrieval/ANN IR papers (ANCE, Anserini HNSW integration) evaluate aggregate effectiveness, not per-document retrievability under ANN — [ANCE](https://arxiv.org/pdf/2007.00808); [Anserini HNSW](https://arxiv.org/pdf/2304.12139)

### Inferences
- Classic IR "retrievability" (Azzopardi & Vinay, CIKM 2008) is the natural framing for an IR audience. A reviewer from that community may say "retrievability bias of the ANN layer is just retrievability". Cite it and position the contribution as index-induced, time-conditioned retrievability loss, at a fixed exact rank, that is separable from the ranker. (Azzopardi & Vinay was not re-fetched in this session; verify the citation details.)
- For IVF, DeDrift/Ada-IVF predict drift effects when centroids are trained on old data and new content lands in imbalanced partitions. An IVF late-snapshot excess of about 1.15x fits that story, so cite them as the expected mechanism for IVF.

### Gaps
- A search for "retrievability bias dense retrieval ANN" returned nothing directly relevant. A targeted search of SIGIR/ECIR/ICTIR retrievability-plus-dense-retrieval work (e.g. 2023–2025 retrievability of neural retrievers) was not done.
- I did not check streaming-vector-DB papers that may report recall by insertion batch (e.g. SPFresh SOSP'23, Quake OSDI'25).

## Q5. Determinism and equivalence of incremental vs batch/rebuilt HNSW/IVF; vendor practice

### Takeaway
I found no paper that states or tests that an incrementally maintained HNSW equals a rebuild from the same insertion history. For HNSW this is a property of the standard algorithm: construction is itself sequential insertion. It holds only with a fixed level-RNG seed and single-threaded insertion. Multi-threaded hnswlib builds are documented as non-deterministic. Vendors (Lucene/Elasticsearch) do not keep one incremental graph: segment merges rebuild or partially reinsert graphs, so "rebuild vs update" there differs by construction. Claim (2) is therefore likely to be read as "true by construction" rather than as a finding. Present it as a verified precondition (bit-identity check) that rules out drift, not as a contribution.

### Cited Findings
- hnswlib issue #28: the threaded add_items "gives me different result & accuracy every time I run the script"; the suggested remedy is a single thread — [GitHub](https://github.com/nmslib/hnswlib/issues/28)
- RcppHNSW test suite: an "index which was built using more than 1 thread is not deterministic"; tests check only closeness between 1-thread and multi-thread results — [RcppHNSW tests](https://rdrr.io/cran/RcppHNSW/src/tests/testthat/test_parallel.R)
- ParlayANN (Manohar et al., PPoPP 2024, arXiv:2305.04359): notes that using locks gives non-deterministic outputs and provides deterministic parallel graph-ANN builds — [arXiv](https://arxiv.org/pdf/2305.04359v2.pdf)
- HNSW construction = repeated insertion: "HNSW sequentially adds new vectors to the index … finds neighbors of the new data by performing ANNS on the index under construction" (survey statement in VLDB 2025 index-construction paper) — [PVLDB 18 p1825](https://www.vldb.org/pvldb/vol18/p1825-liu.pdf)
- Elasticsearch/Lucene: segments each have their own HNSW graph; merges historically kept the largest segment's graph and re-inserted vectors from the others. Newer merging inserts a "join set" and places the rest using small-graph neighbors, with the caveat that "the risk in this strategy is that we damage the graph quality" — [Elastic Search Labs](https://www.elastic.co/search-labs/blog/hnsw-graphs-speed-up-merging)
- SIGMOD 2026 HNSW merging paper classifies Elasticsearch as "rebuild-based" merging and studies merge quality versus rebuild (not read in full) — [Purdue PDF](https://cs.purdue.edu/homes/csjgwang/pubs/SIGMOD26_HNSWMerger.pdf)
- Elastic multi-graph search: recall across multiple segments can be higher than for a single merged segment, so segment layout changes results — [Elastic Search Labs](https://www.elastic.co/search-labs/cn/blog/multi-graph-vector-search)
- IVF: with fixed (pre-trained) centroids, incremental insertion is assignment-only. The quality issue is centroid staleness under drift, addressed by DeDrift and Ada-IVF rather than by order — [DeDrift](https://arxiv.org/abs/2308.02752); [Ada-IVF](https://arxiv.org/html/2411.00970v1)

### Inferences
- Claim (2) is correct but expected. Its value is methodological: it lets the paper say the incremental-vs-rebuild gap is entirely an insertion-order/RNG effect. State the conditions explicitly (single thread, fixed seed, no deletes, fixed IVF centroids, same float ops) and say the result does not transfer to multi-threaded builds or to Lucene-style merged segments.
- An adversarial reviewer might note that deployed systems (Lucene, Milvus segments, Weaviate async indexing) rarely maintain one incremental graph, so the paper's "keep updating" condition is an idealization. Pre-empt this in the threats-to-validity section.

### Gaps
- Milvus, Weaviate, Vespa and Pinecone documentation on incremental vs rebuilt index quality was not fetched in this session.
- I did not find a published statement that single-threaded HNSW with the same seed and order is bit-identical to batch build. This rests on the algorithm definition and the project's own check.
- Overall search budget was limited (about 20 calls); other researchers or a follow-up search should cover SPFresh, Quake, IP-DiskANN, Azzopardi retrievability, and Malkov & Yashunin's exact wording.
