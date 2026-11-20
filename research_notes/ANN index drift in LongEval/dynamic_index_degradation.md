# How ANN indexes degrade under updates, and how they are repaired or rebuilt

Research notes, compiled 2026-11-12. Scope: recall decay of graph-based and partition-based ANN indexes under inserts, deletes and replacements; repair and maintenance methods; production guidance on rebuild vs update; papers on deciding when to rebuild. Purpose: test the novelty of a proposal that compares rebuild, incremental updates and in-place maintenance (recalibrating search effort, refitting quantization, local graph repair) over time-ordered collection snapshots (LongEval).

Note on the numbers: several figures below were pulled from paper figures and HTML text by an automated page summariser, not read off the PDFs by hand. Numbers marked "approx." came from figures and should be checked against the PDF before anyone quotes them.

## HNSW under updates: deletions, unreachable points, hnswlib mark_deleted / replace_deleted, recall drift

### Takeaway
There is solid, quantified evidence that HNSW degrades under repeated delete and reinsert cycles. Recall falls by several points to about 13 points, depending on the delete policy and how many cycles are run. The mechanism is known: some points lose all their incoming edges and become unreachable, and that loss cannot be recovered by raising ef. The fixes in the literature are local neighbour repair (MN-RU), a backup index for unreachable points, or the stronger graph rules used in FreshVamana and IP-DiskANN.

### Cited Findings
- Xiao, Zhan, Xi, Hou, Liao, "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation", arXiv:2407.07871 (July 2024). The paper defines an unreachable point as one "that possesses outgoing edges but lacks incoming edges in all layers". The cause is hnswlib's replaced_update neighbour repair, which deletes edges from the neighbours of a deleted point without making sure those neighbours get replacement in-edges. — [arXiv 2407.07871](https://arxiv.org/html/2407.07871)
- Same paper: after 3000 iterations of deleting and reinserting 5% of SIFT1M, 3-4% of the original dataset was unreachable and recall fell by about 3%. The loss "persists regardless of search parameter tuning", so raising ef does not bring it back. After 200 iterations, GIST1M had 3-4% unreachable points and ImageNet (2.34M, 150-d) had 2-3%. Datasets were SIFT, GIST, ImageNet and SIFT2M. — [arXiv 2407.07871](https://arxiv.org/html/2407.07871)
- Same paper, remedies: MN-RU (and variants MN-RU γ and MN-THN-RU) updates 2-4x faster, with O(M²) rather than O(M³) cost per layer, and leaves significantly fewer unreachable points. A backup index holding unreachable points (threshold τ = 40,000) "significantly reduced the number of unreachable points". Code: https://github.com/xwt1/MN-RU — [arXiv 2407.07871](https://arxiv.org/abs/2407.07871)
- FreshDiskANN (Singh, Subramanya, Krishnaswamy, Simhadri, arXiv:2105.09613, 2021), Figure 1, on SIFT1M with 20 cycles of deleting and reinserting 5% of points:
  - HNSW with delete policy A: recall approx. 98% → 85%.
  - Vamana with delete policy A: approx. 97% → 88%.
  - NSG with delete policy B: approx. 96% → 87%.
  - The paper's conclusion: "simple graph modification rules for insertions and deletions do not retain the same graph quality over a stream of insertions and deletions." — [FreshDiskANN (ar5iv)](https://ar5iv.labs.arxiv.org/html/2105.09613)
- IP-DiskANN (2025) explains HNSW's weaker streaming recall by its delete repair: "it adds O(R³) edges per deletion, compared with O(cR) for IP-DiskANN and O(R²) for FreshDiskANN". The extra short-range edges push out long-range links. — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- hnswlib README: `mark_deleted(label)` "marks the element as deleted, so it will be omitted from search results", and `unmark_deleted` reverses it. Passing `allow_replace_deleted` at init and `replace_deleted=True` in `add_items` reuses the memory of deleted elements. `add_items` is thread-safe with other `add_items` calls but not with `knn_query`. The README says nothing about deleted nodes staying in graph traversal or about quality loss. — [hnswlib GitHub](https://github.com/nmslib/hnswlib)

### Inferences
- Mark-deleted nodes stay in the graph as routing nodes, so a tombstone-only approach keeps paths to live points intact but wastes search effort and memory. The damage to reachability comes from replacement (replace_deleted / replaced_update) repairs, which matches the 2407.07871 analysis.
- In 2407.07871, recall that cannot be recovered by tuning ef points to structural damage. A proposal that "recalibrates search effort" can therefore only recover the part of the loss caused by tombstones or distribution shift, not the part caused by unreachable points. That split is a testable claim the proposal could measure.

### Gaps
- I found no source that cleanly isolates insertion-order effects on HNSW recall, or recall drift caused by pure growth (append-only, no deletes) under a fixed ef. Quake's Wikipedia-12M workload (below) covers growth for partition indexes, not HNSW.
- The FreshDiskANN Figure 1 numbers are approximate readings from a plot.

## Streaming and dynamic graph and partition indexes: FreshDiskANN, IP-DiskANN, SPFresh/LIRE, Ada-IVF, Quake, CleANN, Big-ANN 2023 streaming

### Takeaway
Purpose-built streaming indexes can hold recall about flat over long update sequences:
- FreshVamana: about 95% 5-recall@5 over 50 cycles.
- IP-DiskANN: within about 0.5 points of FreshDiskANN's batch consolidation.
- SPFresh: recall stable or rising over 100 days of 1% daily updates.

Naive in-place updates and global-rebuild schedules both lose quality or tail latency. Partition (IVF/SPANN) indexes degrade through partition imbalance; local split, merge and reassign (LIRE, Ada-IVF, Quake) fixes this.

### Cited Findings
- FreshDiskANN: the α-RNG prune with α > 1 keeps FreshVamana's 5-recall@5 at about 95% over 50 cycles of deleting and reinserting 5%, 10% and 50% of SIFT1M, with similar results on GIST1M, Deep1M and SIFT100M. StreamingMerge applies a 10% change to a billion-scale index in about 10% of the time a full rebuild takes. The abstract reports ">95% 5-recall@5" and a "5-10x reduction in the cost of maintaining freshness" against periodic rebuilds. — [FreshDiskANN (ar5iv)](https://ar5iv.labs.arxiv.org/html/2105.09613); [arXiv abs](https://arxiv.org/abs/2105.09613)
- IP-DiskANN (Xu, Manohar, Bernstein, Chandramouli, Wen, Simhadri; arXiv:2502.13826, Feb 2025) is "the first algorithm to avoid batch consolidation by efficiently processing each insertion and deletion in-place". It uses a reverse adjacency index, local reconnection, adaptive pruning and incremental neighbour reselection. A cheap scan for dangling edges, with no distance computations, runs when deletes pass a threshold of about 20% of the index. — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- IP-DiskANN recall@10 against FreshDiskANN and HNSW:
  - MSTuring-10M SlidingWindow: 94.8% vs 94.4%, HNSW 91.8%.
  - MSTuring-30M Clustered: 92.5% vs 91.8%, HNSW 89.5%.
  - Wiki-35M ExpirationTime: 97.5% vs 97.3%.
  - HNSW scored 95.9% on Wiki-1M.
  - Delete time: Wiki-10M 2,170 s vs 3,222 s; Wiki-35M 10,027 s vs 14,535 s.
  - Runbooks: SlidingWindow, ExpirationTime (lifetimes of 10/50/100 steps) and Clustered (k-means-correlated inserts and deletes). — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- SPFresh / LIRE (Xu et al., SOSP 2023; arXiv:2410.14452).
  - Motivation: naive in-place updates to SPANN+ on a 2M index with 0.5M updates (25%) lose "more than one point" of recall, and tail latency rises 4x over the static index.
  - Workload of 100M vectors with 1% daily updates for 100 days. SPANN+ append-only P99.9 latency grows from about 4 ms to more than 10 ms. DiskANN with global rebuild has P99.9 that "fluctuates significantly with dramatic increase to more than 20ms" during rebuilds, and its recall "decreases over time". SPFresh holds P99.9 at about 4 ms (2.41x lower than DiskANN on average), and its recall "increases gradually".
  - LIRE overhead: only 0.4% of inserts trigger a split, merges happen on 0.1% of updates, and each reassignment evaluates about 5,094 vectors but moves about 79.
  - At billion scale with a 1% daily update rate, SPFresh needs 1% of the DRAM and under 10% of the cores of the state of the art. — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1); [MSR SOSP PDF](https://microsoft.com/en-us/research/uploads/prod/2023/08/SPFresh_SOSP.pdf)
- Ada-IVF (arXiv:2411.00970, 2024): "Existing vector search indexes… degrade in search quality and performance as the underlying data is updated unless costly index reconstruction is performed." It uses an adaptive maintenance policy with a per-partition "temperature" (read frequency) and a local indicator function, followed by local re-clustering. Against prior dynamic IVF maintenance it gives 2x higher update throughput on average and up to 5x. On an internal recommendation workload it cuts update time to 62% of LIRE's, with 9% higher QPS. — [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)
- Quake (Mohoney et al., OSDI 2025; arXiv:2506.03437):
  - Static Faiss-IVF and SCANN with fixed parameters "experience substantial performance decay over time" on Wikipedia-12M, which grows from 1.6M to 12M vectors over 103 monthly snapshots with pageview-skewed queries.
  - Adaptive Partition Scanning (APS) recalibrates per-query nprobe toward a recall target. On SIFT1M at 90% recall it matches an oracle nprobe at 17-29% more latency than the oracle.
  - Maintenance (split, merge, add or remove levels) is driven by the cost model C = Σ A·λ(s), where A is access frequency and s is partition size. Actions that do not lower the cost after verification are rolled back.
  - Against SVS, DiskANN, HNSW and SCANN: query latency 1.5-38x lower, update latency 4.5-126x lower.
  - Other workloads: OpenImages-13M (sliding 2M window) and MSTuring RO and insertion-heavy variants. — [arXiv 2506.03437](https://arxiv.org/html/2506.03437v2); [USENIX OSDI'25](https://www.usenix.org/conference/osdi25/presentation/mohoney)
- CleANN (arXiv:2507.19802, July 2025): "Existing dynamic graph-based indexes suffer from either query quality degradation as updates happen or expensive global graph structure updates." It combines workload-aware linking, query-adaptive neighbourhood consolidation and semi-lazy memory cleaning, for 7-1200x throughput on million-scale data with 56 threads. — [arXiv 2507.19802](https://arxiv.org/abs/2507.19802v1)
- Big-ANN NeurIPS'23 streaming track (Simhadri et al., arXiv:2409.17424):
  - Data: a 10M slice of MSTuring.
  - Hardware: Azure D8lds_v5 (8 vCPU, 16 GB), with 8 GB for the index; the runbook had to finish within 1 hour.
  - Runbook: 1280 batches over 5 rounds, with inserts, deletes and searches in a 4:4:1 ratio, over 64 clusters.
  - Metric: recall@10 against the live set at query time, averaged over checkpoints.
  - A caching bug meant the first-published scores reflected only the first snapshot. Corrected scores: DiskANN baseline 0.7218 (first reported as 0.883), PyANNS 0.8865 (DiskANN streaming plus 8-bit SQ, the winner), hwtl_sdu 0.7693. — [arXiv 2409.17424](https://arxiv.org/html/2409.17424v1)

### Inferences
- The gap between first-snapshot recall and recall averaged over the runbook (0.883 vs 0.7218 for DiskANN) is direct evidence that recall decays across a streaming run on a clustered, distribution-shifting workload.
- In the work found here, "in-place maintenance" for graph indexes means local edge repair, and for partition indexes it means split, merge and reassign plus adaptive nprobe (Quake's APS). Two of the proposal's three levers are therefore already in the literature separately: local graph repair, and search-effort recalibration in Quake APS. The novel angle is more likely the combination, plus quantization refitting, compared head to head against rebuild on real time-ordered IR snapshots.

### Gaps
- DEG (dynamic edge or exploration graph) and "graph reconstruction" papers were not checked in this pass, nor were other 2025-26 dynamic indexes such as Greator, LSM-VEC or SVFusion.
- No per-X%-update recall-loss figure was found for IP-DiskANN or CleANN beyond the endpoint recalls above.

## Benchmarks with updates (Big-ANN streaming, dynamic workloads, VIBE / ann-benchmarks)

### Takeaway
The standard update benchmark is the Big-ANN streaming runbook framework: synthetic clustered, sliding-window and expiration runbooks on MSTuring and Wiki embeddings. Quake's Wikipedia-12M (103 monthly snapshots) is the closest existing benchmark to real time-ordered snapshots. I found no evidence that ann-benchmarks or VIBE include update workloads.

### Cited Findings
- Big-ANN'23 streaming runbook design and metrics are described in the section above. — [arXiv 2409.17424](https://arxiv.org/html/2409.17424v1)
- IP-DiskANN contributes the SlidingWindow, ExpirationTime and Clustered runbooks on MSTuring and Wiki-35M. — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- Quake's Wikipedia-12M grows over 103 monthly updates with real pageview query skew, and OpenImages-13M uses a sliding window. — [arXiv 2506.03437](https://arxiv.org/html/2506.03437v2)
- SPFresh's simulated "1% daily update for 100 days" workload is in the section above. — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)
- The big-ann-benchmarks repo hosts an ongoing NeurIPS'23 leaderboard. — [GitHub leaderboard](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/ongoing_leaderboard/leaderboard.md)

### Inferences
- None of the benchmarks found uses IR collections with relevance judgements over time, like LongEval. All measure recall against exact k-NN, not downstream retrieval effectiveness such as nDCG. This looks like a real gap for the proposal to occupy.

### Gaps
- I did not verify whether VIBE or ann-benchmarks added update tracks by 2026. I found no evidence that they did, which is absence of evidence, not proof.

## Production systems: when to rebuild (Milvus, Vespa, Weaviate, pgvector, Qdrant, FAISS, Elasticsearch/OpenSearch)

### Takeaway
Production systems mostly use tombstones plus threshold-triggered background compaction or vacuum, which rewrites or reindexes segments. The common default threshold is about 20% deleted. Vespa mutates a single HNSW graph in real time. Trained components (IVF centroids, PQ codebooks) are fitted once on initial data. I found no doc that states a quantitative recall-drift or retraining trigger.

### Cited Findings
- Milvus deletes go to DEL files (bitmaps). Compaction "merg[es] small segments into large ones and clean[s] logically deleted data". It triggers when "rows in delta binlogs are more than 20% of the total rows" or delta binlogs pass 10 MB, and also when there are more than 10 segments smaller than 0.5×MaxSegmentSize. The blog says compaction helps "improve query efficiency" but does not quantify any recall effect. — [Milvus blog: How to Compact Data](https://milvus.io/blog/2022-2-21-compact.md)
- Milvus: "if a segment has already been indexed, compact also deletes the previous index file and creates new indexes", so the index is rebuilt per segment on compaction. — [DZone: How Milvus Realizes the Delete Function](https://dzone.com/articles/how-milvus-realizes-the-delete-function)
- Qdrant's vacuum optimizer has defaults `deleted_threshold = 0.2` (20% of a segment's records deleted) and `vacuum_min_vector_number = 1000`. When both are met it removes deleted records and rebuilds or reorganizes the segment. — [Qdrant optimizer docs](https://qdrant.tech/documentation/ops-optimization/optimizer/)
- Weaviate: "Tombstones are records that mark deleted objects. In an HNSW index, tombstones are regularly cleaned up, triggered periodically by the `cleanupIntervalSeconds` parameter." The environment variables TOMBSTONE_DELETION_MAX_PER_CYCLE, MIN_PER_CYCLE and CONCURRENCY tune cleanup. For PQ: "You should have 10,000 to 100,000 vectors per shard loaded before you enable PQ." The docs do not say whether the codebook is ever retrained after drift. The dynamic index switches from flat to HNSW above 10,000 objects by default. — [Weaviate vector index config](https://docs.weaviate.io/weaviate/config-refs/schema/vector-index); [Weaviate academy compression](https://docs.weaviate.io/academy/py/compression/strategy)
- Vespa offers "Real Time Indexing" with "no query or indexing overhead from searching multiple HNSW graphs", using one graph per field per node. "The costly part when performing real time changes to the HNSW graph is distance calculations while searching the graph layers to find which links to change." Changing the distance metric or max-links-per-node requires a restart but no re-feed. — [Vespa HNSW docs](https://docs.vespa.ai/en/querying/approximate-nn-hnsw.html)
- pgvector README:
  - IVFFlat: "Create the index after the table has some data", with lists = rows/1000 (or sqrt(rows) above 1M rows) as the main recall knob.
  - HNSW: "Vacuuming can take a while for HNSW indexes. Speed it up by reindexing first."
  - Filtering is applied "after the index is scanned". — [pgvector README](https://github.com/pgvector/pgvector)
- FAISS FAQ: retraining is needed "when the type of media you index becomes statistically different" (e.g. a retrained CNN). "To adapt eg. to a drifting data distrubution, see how to warm-start the training" (links a notebook). Also: "it is simpler to just construct a new one" rather than un-train. — [FAISS wiki FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)

### Inferences
- The production default is a count-based trigger (about 20% deleted for Milvus and Qdrant, about 20% for IP-DiskANN's cleanup), not a quality-based one. No system reviewed triggers maintenance from measured recall drift or quantizer distortion. A drift-detection trigger would be a meaningful difference.
- Static quantizers (IVF centroids, PQ codebooks) are trained once and then kept. FAISS points to warm-start retraining, but no vendor doc found prescribes when to refit.

### Gaps
- Not fetched: the pgvector README text about IVFFlat recall falling after many data changes and needing REINDEX. Secondary sources often state this, but I did not verify the exact wording in the README, so treat it as unconfirmed.
- Not covered: Elasticsearch and OpenSearch, which use Lucene HNSW per segment with graphs rebuilt on segment merge, and force-merge guidance.
- Not found: Weaviate guidance on PQ codebook refitting.
- Not reliably confirmed: hnswlib behaviour beyond the README.

## Papers on deciding when to rebuild (cost models, drift detection)

### Takeaway
The closest work to a rebuild-decision policy is cost-model-driven local maintenance: Quake's latency cost model with verify-and-rollback, and Ada-IVF's partition temperature and indicator function. Both target partition indexes, and both decide local actions rather than a global rebuild versus update. I found no paper with an explicit cost model for when to do a global rebuild instead of incremental updates, triggered by measured recall or quantization drift.

### Cited Findings
- Quake: the cost model C = Σ A_lj·λ(s_lj) decides split, merge and level changes, and actions are rejected if verified cost does not improve. — [arXiv 2506.03437](https://arxiv.org/html/2506.03437v2)
- Ada-IVF: an adaptive policy decides "which index partitions are problematic for performance and should be repartitioned" using per-partition read temperature. — [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)
- IP-DiskANN: a threshold of about 20% deleted triggers a lightweight dangling-edge cleanup. — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- FreshDiskANN quantifies rebuild against merge: merging a 10% change costs about 10% of a rebuild at billion scale. — [FreshDiskANN (ar5iv)](https://ar5iv.labs.arxiv.org/html/2105.09613)
- SPFresh: periodic global rebuild (DiskANN) produces P99.9 spikes above 20 ms and declining recall, compared with stable local rebalancing. — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)

### Inferences
- What exists:
  - Benchmarked comparisons of rebuild against incremental updates against in-place repair on synthetic runbooks: FreshDiskANN, SPFresh, IP-DiskANN.
  - Search-effort recalibration for IVF: Quake APS.
  - Local graph repair: MN-RU, IP-DiskANN, CleANN.
- What I did not find:
  - A study of quantization refitting over time.
  - A drift-triggered rebuild-versus-maintain decision rule.
  - An evaluation on real time-ordered IR collections (LongEval) that measures retrieval effectiveness rather than only k-NN recall.

  These three absences are where the proposal's novelty claim is strongest.

### Gaps
- My search was limited, about 20 queries and fetches. Work on learned or online rebuild triggers may exist, for example in VLDB/SIGMOD 2025-26 or the LSM-style vector index literature (LSM-VEC, Greator), and was not searched specifically. Treat "no such paper" as absence of evidence from a limited search.
