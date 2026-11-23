# How production vector databases handle growing collections, and whether recall for new inserts degrades

Research date: 2026-11-23. Quotes are marked as quotes. Information from vendor blogs is marked "vendor". GitHub/forum reports are marked "user report". Some pages were read through a summarizing fetch tool; where I could see only a search snippet and not the page itself, the finding says so.

## Q1. Per system: how inserts become searchable, whether the graph or partitions are rebuilt, and what docs say about recall over time

### Takeaway
Almost every production system is built LSM-style. New writes go into a small mutable buffer or segment. That buffer is searched by brute force, by a cheap temporary index, or by a small separate graph. Later, background jobs build or merge indexed segments, and those jobs rebuild graphs or partitions from scratch. Only Vespa, Redis, pgvector HNSW and Cosmos DB DiskANN keep one long-lived graph that they change in place. Those single-graph systems are the ones where newly inserted nodes having fewer incoming links would persist and show up. No vendor doc reports measured recall over time for *newly inserted* items. Docs discuss deletes/tombstones (HNSW) and centroid drift (IVF) instead.

### Cited Findings

**Pinecone (serverless, slab architecture)**
- Serverless architecture blog (2024-01-16, vendor): a freshness layer runs alongside the index builders and tails the same log. It "builds a compact fresh index over the most recent data" and "only cares about data that hasn't been indexed yet." The index builder "maintains a hierarchy of partitions starting from coarse-grained to more refined as it sees more data." — [Pinecone blog](https://www.pinecone.io/blog/serverless-architecture/)
- Slab architecture page (2025-11-04, vendor): "Writes always go to a new L0 slab, so they can be written very quickly without waiting for an index merge or rebuild." The read path includes a "Memtable: exact brute-force scan in memory for the most recent writes (up to ~10k vectors)." Slabs are immutable. — [Pinecone: slab architecture](https://www.pinecone.io/learn/slab-architecture/)
- Index type depends on slab size (vendor): small slabs (up to ~10k vectors) use "Ananas" (SimHash), medium slabs (~10k–100k) use PQ fast-scan ("PQFS"), and large slabs (over ~100k) use IVF. "When enough L0 slabs accumulate, compaction kicks in", merging L0→L1→L2→L3. "If a slab accumulates too many tombstones, Pinecone proactively rebuilds it." The only recall-over-time claim is marketing: "Adaptive indexing maintains retrieval quality as datasets scale." No numbers are given. — [Pinecone: slab architecture](https://www.pinecone.io/learn/slab-architecture/)
- A search summary of Pinecone material mentioned a threshold of about 10 L0 slabs triggering an L1 compaction. The slab page I fetched gives no number, so treat the threshold as unconfirmed. — [Pinecone: slab architecture](https://www.pinecone.io/learn/slab-architecture/)
- The official docs say the system is eventually consistent: "there can be a slight delay before upserted records are available to query." Freshness is checked by comparing the write's LSN with the `x-pinecone-max-indexed-lsn` query response header. — [Pinecone docs: check data freshness](https://docs.pinecone.io/guides/index-data/check-data-freshness); [understanding data freshness](https://docs.pinecone.io/guides/data/data-freshness/understanding-data-freshness)
- User reports of multi-minute visibility delays exist, for example a thread titled "Pinecone takes about 2-3 minutes for upsert vector to be available for query". These are anecdotal. — [Pinecone community](https://community.pinecone.io/t/pinecone-takes-about-2-3-minutes-for-upsert-vector-to-be-available-for-query/5701/2)

**Weaviate (HNSW, flat, dynamic, HFresh; async indexing)**
- "When asynchronous indexing is enabled, all vector indexing operations go through a queue." The docs say "there will be a short delay between object creation and the object being available for vector search using the HNSW index." The concepts page does not say how queued objects are searched. — [Weaviate docs: vector index concepts](https://docs.weaviate.io/weaviate/concepts/vector-index)
- Deletes: "Prior to cleanup, objects are marked as deleted, but they are still connected to the HNSW graph." Tombstone cleanup is "an async process" that "rebuilds the HNSW graph after deletes and updates". This is a local repair, not a full rebuild. — [Weaviate docs](https://docs.weaviate.io/weaviate/concepts/vector-index)
- The dynamic index switches from flat (brute force) to HNSW "when the object count exceeds a specified threshold (by default 10,000)". This is "only a one-way upgrade" and is marked experimental. — [Weaviate docs](https://docs.weaviate.io/weaviate/concepts/vector-index)
- HFresh is an SPFresh-inspired disk index: postings on disk, with an in-memory HNSW over the centroids. It was a technical preview in v1.36 and became GA in v1.38. Weaviate (vendor) says it "rebalances its posting lists in the background as data changes, so it does not require periodic full index rebuilds", using splits, merges and reassignment. — [Weaviate HFresh blog](https://weaviate.io/blog/hfresh); [Weaviate 1.38 release](https://weaviate.io/blog/weaviate-1-38-release); [Weaviate 1.36 release](https://weaviate.io/blog/weaviate-1-36-release)
- Forum reports describe tombstone cleanup being skipped under memory pressure (a "skipping hnsw cleanup due to memory pressure" log line) and heavy CPU use during `tombstone_cleanup_begin`. These are user reports. — [Weaviate forum: memory pressure under write/delete load](https://forum.weaviate.io/t/memory-pressure-in-single-instance-weaviate-under-continuous-write-deletion-load/22104); [forum: CPU during tombstone cleanup](https://forum.weaviate.io/t/weaviate-using-tons-of-cpu-during-tombstone-cleanup-begin/7290)

**Milvus / Zilliz (growing vs sealed segments)**
- The queryNode config docs say an interim index is enabled by default. It builds a temporary index on growing segments, and on sealed segments not yet indexed, so queries run right after insertion are faster. Parameters include `queryNode.segcore.chunkRows` (default 128) and `interimIndex.nlist` (about sqrt(chunkRows)). Milvus "eventually seals and indexes all segments." — [Milvus docs: queryNode configuration](https://milvus.io/docs/configure_querynode.md)
- The v2.6.x release notes add `growingBuildThreadRate` for multi-threaded interim index builds on growing segments. — [Milvus release notes v2.6.x](https://milvus.io/docs/v2.6.x/release_notes.md)
- A 2026 arXiv survey describes Milvus growing segments as in memory and searchable by brute force once the WAL acknowledges them. When they reach a size threshold, index nodes build HNSW/IVF indexes and flush immutable sealed segments. I saw only a search snippet of this survey. — [arXiv 2605.01260](https://arxiv.org/pdf/2605.01260)
- Historical (2021): a maintainer explained that search fell back to brute force while index build tasks were still running. — [Milvus discussion #5023](https://github.com/milvus-io/milvus/discussions/5023)
- Compaction merges segments and rebuilds the index on the merged segment, because sealed segments are immutable. This comes from the general architecture descriptions above. I did not fetch an official compaction page.

**Qdrant (segments, optimizer, indexing threshold)**
- "New points go to an appendable segment, which has no vector index." "The indexing optimizer builds a vector index for a segment once its vector data exceeds the indexing threshold." Below that threshold, "Qdrant has to scan large amounts of unindexed data for every query." — [Qdrant docs: optimizer](https://qdrant.tech/documentation/ops-optimization/optimizer/)
- Merge optimizer: "When there are more segments than this target, the optimizer merges the smallest segments into larger ones." Vacuum optimizer: "permanently removing marked points and reorganizing storage". "deleted_threshold sets the minimum fraction of deleted points in a segment required to initiate optimization." — [Qdrant docs: optimizer](https://qdrant.tech/documentation/ops-optimization/optimizer/)
- Search snippets of Qdrant's FAQ say that while background indexing runs, unindexed segments are searched by full scan, so new points stay searchable. The FAQ page did not load for verbatim reading. — [Qdrant FAQ](https://qdrant.tech/documentation/faq/)
- `indexed_only` search skips unindexed segments. A `prevent_unoptimized` optimizer option is described as experimental, added in v1.17.1. Qdrant pages disagree about the threshold's units: `indexing_threshold_kb` in kilobytes versus a vector count. — [Qdrant: prevent unoptimized usage](https://qdrant.tech/documentation/tutorials-operations/prevent-unoptimized-usage/); [Qdrant: predictable search latency](https://qdrant.tech/documentation/operate-and-maintain/predictable-search-latency/)

**pgvector (HNSW vs IVFFlat)**
- README: IVFFlat should be created "after the table has some data", and a "Choose an appropriate number of lists" step follows. If the index is built with too little data: "Drop the index until the table has more data." HNSW: "an index can be created without any data in the table since there isn't a training step like IVFFlat." — [pgvector README](https://github.com/pgvector/pgvector)
- The README does not say that IVFFlat recall drops as new rows arrive. It does say: "Monitor recall by comparing results from approximate search with exact search," and "Vacuuming can take a while for HNSW indexes. Speed it up by reindexing first." — [pgvector README](https://github.com/pgvector/pgvector)
- AWS's Aurora pgvector production guide (vendor-adjacent) states the IVFFlat staleness mechanism directly: centroids are fixed at build time, new vectors may not match the existing clusters, recall drops, and the fix is to rebuild, usually with `REINDEX INDEX CONCURRENTLY`. — [AWS Database Blog: running pgvector in production on Aurora](https://aws.amazon.com/blogs/database/running-pgvector-in-production-on-amazon-aurora-postgresql/)
- pgvector HNSW inserts update the single graph in place. There are no segments and no flat buffer, so every insert is linked into the one graph. This is inferred from the README's description and the lack of any segment concept in it.

**Elasticsearch / OpenSearch / Lucene (HNSW per segment)**
- Elastic (2025-04-07, vendor engineering): Lucene segments are immutable, so each segment has its own HNSW graph. A merge inserts the vectors from the smaller graphs into the largest graph. Write amplification is about 1 + (9/10)·log10(n/n0), so total build time is "typically only double" that of building one graph. The merge-reuse optimization was "available in the upcoming Lucene 10.2". Multi-segment recall with the new method is "slightly worse, but nearly comparable" on most datasets. After merging to one segment, recall curves are "almost identical." — [Elastic Search Labs: speeding up merging of HNSW graphs](https://www.elastic.co/search-labs/blog/hnsw-graphs-speed-up-merging)
- Elastic's design rationale post: on merge, Lucene "creates a copy of the HNSW graph from the largest input segment that doesn't have deletes and then adds vectors from other segments". It names the index-time overhead of segment merges compared with mutating one graph in place. — [Elastic: vector search rationale](https://www.elastic.co/search-labs/blog/vector-search-elasticsearch-rationale)
- So each refresh creates a new small segment with its own small graph, and queries fan out to every segment. A merge reinserts the smaller segments' vectors into a graph that keeps growing, which is a partial rebuild. The Elastic posts do not discuss recall for recently indexed documents specifically.

**Vespa (single mutable HNSW)**
- "Vectors can be added, updated, and removed from the index in true real time." The graph is mutable, so there is no need to search multiple graphs. The cost is distance calculations during insertion. Searching the HNSW index is single-threaded per query. — [Vespa docs: ANN using HNSW](https://docs.vespa.ai/en/querying/approximate-nn-hnsw)
- I found nothing in the Vespa docs about recall drift, rebuilding, or in-degree of late inserts.

**Turbopuffer (SPFresh-style centroid index on object storage)**
- Every write is a new file in a WAL on object storage, with at most one WAL entry per second per namespace and concurrent writes batched. "After data is committed to the log, it is asynchronously indexed." "Any data that has not yet been indexed is still available to search, with a slower exhaustive search of recent data in the log." — [turbopuffer docs: architecture](https://turbopuffer.com/docs/architecture)
- Strongly consistent queries (the default) search all unindexed data exhaustively. Eventually consistent queries search up to 128 MiB of unindexed data. The docs disagree on the staleness bound: one page says up to 60 s, another up to about 1 hour in the worst case. The `unindexed_bytes` and `unindexed_rows` metadata fields expose the backlog. — [turbopuffer docs: concepts](https://turbopuffer.com/docs/concepts); [architecture](https://turbopuffer.com/docs/architecture)
- The SPFresh-based centroid index (vendor) "avoids expensive full index rebuilds." — [turbopuffer docs](https://turbopuffer.com/docs/architecture)

**LanceDB (IVF-PQ, reindexing)**
- "As data is being added and a reindex operation is running, LanceDB will combine results from the existing index with exhaustive/flat search on the new data." `index_stats()` reports unindexed rows. `optimize()` adds new rows to the existing index without retraining it. If the data distribution has shifted substantially, the docs suggest a full rebuild. `fast_search=True` skips unindexed rows, and new rows are then missing from results. — [LanceDB docs: reindexing](https://docs.lancedb.com/indexing/reindexing)
- LanceDB Cloud/Enterprise automatically trigger a new index build when data is added (vendor). — [LanceDB docs: reindexing](https://docs.lancedb.com/indexing/reindexing)
- The docs explicitly tell users to rebuild after distribution shift, because new rows go into partitions trained on old data. This is the documented IVF staleness mechanism.

**Chroma**
- Single-node Chroma keeps an in-memory brute-force buffer in front of hnswlib. `batch_size` (default 100) is the "In-memory bruteforce index size before HNSW flush". `sync_threshold` (default 1000) controls when the index is persisted to disk. — [Chroma docs: configure collections](https://docs.trychroma.com/docs/collections/configure); [Chroma cookbook: configuration](https://cookbook.chromadb.dev/core/configuration/)
- Chroma Cloud and distributed Chroma use SPANN, which is not available on single-node. — [Chroma docs](https://docs.trychroma.com/docs/collections/configure)

**Redis (RediSearch / Redis Query Engine)**
- On Redis 8.8+, when `search-workers` is nonzero, indexing runs in background workers. A document can then exist in the keyspace before a search can see it, which Redis calls "a short indexing delay, not data loss." `--search-workers 0` gives read-after-write. "When you delete a document, the index entry is removed." — [RedisVL docs: search and indexing](https://redis.io/docs/latest/develop/ai/redisvl/0.30.0/concepts/search-and-indexing/)
- I found no Redis doc on HNSW recall over time or rebuild policy.

**MongoDB Atlas Vector Search**
- Atlas Search and Vector Search indexes are built from change streams by the `mongot` process and offer only eventual consistency. A MongoDB staff reply says there are no plans for stronger consistency, and read-your-writes is not supported. — [MongoDB forum: are vector indexes updated in the background?](https://mongodb.com/community/forums/t/are-atlas-search-vector-indexes-updated-in-the-background/272676); [MongoDB forum: delay of index update](https://www.mongodb.com/community/forums/t/delay-of-atlas-search-index-update-when-documents-are-updated/182013)
- If `mongot` falls behind the oplog window it does a full index rebuild automatically, and queries "return stale results" while that runs. — [MongoDB docs: Atlas Search alerts](https://www.mongodb.com/docs/atlas/reference/alert-resolutions/atlas-search-alerts/)

**Azure Cosmos DB (DiskANN, single graph updated in place)**
- DevBlog part 4 (2025-05-07, vendor, Microsoft authors): the index is maintained by two asynchronous processes, one for inserts/replaces/deletes and one for cleaning stale edges. Deletes are handled in place. It was tested on streaming runbooks (MSTuring 1M/10M, Wiki-Cohere 10M, an expiration runbook, and a clustered runbook with 64 k-means clusters over 5 rounds). Recall is shown only as plots. The authors say "many vector databases either skip evaluating deletes, rebuild the index periodically, or ignore data modifications," and that recall degrades when modifications are not handled. They give no numbers for competitors. — [Microsoft DevBlogs: DiskANN part 4](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/)
- The VLDB 2025 paper (Upreti et al.) reports stable recall under updates and <20 ms latency over 10M vectors. The arXiv v1 and VLDB versions give different cost-comparison numbers. — [VLDB PVLDB vol 18 p5166](https://www.vldb.org/pvldb/vol18/p5166-upreti.pdf); [arXiv 2505.05885](https://arxiv.org/abs/2505.05885v1)

### Inferences
- Most systems hide recency bias in the short term because new data goes into a separate buffer or segment that is brute-forced or lightly indexed. Bias for new items appears in three places. (a) Single-graph systems with in-place inserts: pgvector HNSW, Vespa, Redis, Cosmos DiskANN, and Weaviate HNSW after the async queue drains. (b) The largest merged segment in Lucene/ES/OpenSearch: merge copies the largest segment's graph and inserts the other segments' vectors into it, so those vectors are "late inserts" in the merged graph. (c) The largest Qdrant segments, if they are built incrementally. I did not verify whether Qdrant builds a merged segment's HNSW from scratch or incrementally.
- In Milvus, sealed-segment indexes and Pinecone's slab indexes are built in batch over immutable data. Inside one sealed segment, insertion order follows the build order rather than arrival time. Bias by age would then come mostly from segment boundaries, which differ by how many segments and how much ef each segment gets, rather than from in-degree.
- IVF and centroid systems (pgvector IVFFlat, LanceDB, Pinecone large slabs, Turbopuffer/HFresh/SPFresh) have a different form of new-item bias: centroids trained on old data. Vendors document it and offer fixes such as rebuilds and split/merge rebalancing.

### Gaps
- I found no vendor doc measuring recall *for newly inserted vectors versus old ones*, in any of the 13 systems.
- I could not confirm whether Weaviate brute-forces its async queue at query time. Only auto-generated third-party pages claim this, with a "100,000 objects" figure.
- I could not confirm whether Qdrant's optimizer rebuilds HNSW from scratch on merge or reuses graphs, or whether OpenSearch differs from Elasticsearch.
- I did not fetch Zilliz Cloud specifics (Cardinal engine), the OpenSearch docs, or the MongoDB internals (mongot is Lucene-based, but I did not verify that here).

## Q2. Documented complaints and measurements of recall dropping after inserts, deletes or reindexing

### Takeaway
Complaints are common, but nearly all of them are about **deletes and tombstones** (HNSW) or **stale centroids** (IVF). One public, measured user report matches the "late-inserted items are under-linked" mechanism: Chroma issue #7963 (Oct 2026). There, vectors inserted while the collection was small became largely unreachable after a much larger batch arrived, and re-upserting them fixed it. Academic papers measure delete/reinsert degradation in HNSW, and insertion-order effects, more than vendors do.

### Cited Findings
- **Chroma #7963** (opened 2026-10-27, open, no maintainer reply, user report): reported on Chroma 1.5.9 and also on 1.3.7 and 1.0.21, using bge-small 384-d cosine embeddings. Small documents inserted first, then a 4,978-chunk document. Self-recall@6 at ef_search 100: doc A 59/82, doc B 0/7, large doc 4,978/4,978. At ef_search 400: 77/82 and 1/7. Of 31 real questions, 17 lost their target document from the top 6. A synthetic repro (small-first) gave 80–109/120 across seeds. Large-first ordering gave about 115–118/120. Re-upserting the unchanged vectors restored 120/120 and 82/82. A community commenter found that only 38 of 1,717 links into small_a came from the large cluster. The commenter also found that `modify()` of ef_search did not reach the cached index, so part of the "ef doesn't help" result is a separate bug. — [chroma-core/chroma#7963](https://github.com/chroma-core/chroma/issues/7963)
- **pgvector #244** (2023-08-29, closed 2024-10-30): with HNSW, dead tuples from updates and deletes can fill the ef_search candidate list, so a query returns fewer rows than asked for, or none. The maintainer: "I'm not sure there's a way to address this that doesn't affect search speed." It was resolved by iterative index scans in 0.8.0. The maintainer noted the lack of user reports suggests it "doesn't really seem like this was an issue in practice." — [pgvector/pgvector#244](https://github.com/pgvector/pgvector/issues/244)
- **Milvus #52605** (2026-08-18 to 2026-10-19, milestone 3.0.4): on v2.6.22, grouped-search Recall@5 fell from 58/74 to 44/74 after delete/reinsert until compaction, then returned to 58/74. Without grouping it held at 59/74. The root cause was an iterator defect with empty leading chunks, fixed in PR #52663. This is a bug, not a graph-quality effect. — [milvus-io/milvus#52605](https://github.com/milvus-io/milvus/issues/52605)
- **Weaviate #13609** (v1.38.15, user report): HNSW nearVector returned 0–2 hits for 3–9 readable objects. Deleting and reinserting 47 objects restored search, but the cause was not established. — [weaviate/weaviate#13609](https://github.com/weaviate/weaviate/issues/13609)
- **Weaviate #13686**: a delete that arrives before its insert is lost on multi-vector and flat indexes, leaving stale documents in results. — [weaviate/weaviate#13686](https://github.com/weaviate/weaviate/issues/13686)
- **Laurus #1241** (small project): "HNSW builds lose recall when vectors are inserted cluster by cluster." This is the insertion-order effect in a non-major engine. — [mosuka/laurus#1241](https://github.com/mosuka/laurus/issues/1241)
- **Academic (peer-reviewed or preprint):** Xiao et al. (arXiv 2407.07871) find that unreachable points in HNSW grow steadily with repeated delete/reinsert, reaching 3–4% of the dataset after about 3,000 iterations, while recall falls. — [arXiv 2407.07871](https://arxiv.org/html/2407.07871v2)
- FreshDiskANN shows recall deteriorating steadily for graph indexes under streaming updates when nothing is consolidated. — [arXiv 2105.09613](https://arxiv.org/pdf/2105.09613)
- Elliott & Clark (arXiv 2405.17813) find insertion order changes HNSW recall by up to 12.8 percentage points when ordered by local intrinsic dimensionality, and up to 8 pp when ordered by category. — [arXiv 2405.17813](https://arxiv.org/abs/2405.17813v1)
- Another paper reports that indexes are often rebuilt once 10–20% of nodes are deleted. — [arXiv 2502.13826](https://www.arxiv.org/pdf/2502.13826)
- **Code comment** (non-peer-reviewed): the iqdb-hnsw crate says that if a new node gets no reverse links, traversal "cannot reach the new node and recall sags by several percent." — [iqdb-hnsw insert.rs](https://docs.rs/crate/iqdb-hnsw/latest/source/src/insert.rs)
- **Non-peer-reviewed blogs:** the pg_trickle "HNSW recall distribution drift" post explains IVFFlat staleness and HNSW tombstones. Its figures (94%→71% recall, 13.0%/33.6% drift) are illustrative, not measured, and it promotes that product's auto-reindexing. — [pg_trickle blog](https://api.pgxn.org/src/pg_trickle/pg_trickle-0.83.0/blog/hnsw-recall-distribution-drift.md)
- A DEV post ("why your index isn't as healthy as you think") is in the same genre and was not read in full. — [DEV Community](https://dev.to/kenwalger/vector-search-at-scale-why-your-index-isnt-as-healthy-as-you-think-1c19)

### Inferences
- The industry recognizes "recall degrades with churn" as a problem, which is why Cosmos DB, Milvus, Qdrant, Weaviate and Pinecone run tombstone cleanup, compaction or rebuilds. "New items specifically are under-retrieved because of in-degree" is not a recognized vendor-level problem. The Chroma report is the closest public evidence and is unacknowledged by maintainers. This leaves room for a measured contribution. The Chroma case also shows the effect can be severe (0/7) when a small, early cluster is followed by a large batch. That is the reverse direction (old items under-linked), so the bias is really about **late-arriving or under-linked regions**, not only "new".
- Because brute-forced buffers mask the effect for minutes to hours, users mostly see it after the data has been merged into a large graph, when it is hard to attribute.

### Gaps
- I found no public RAG incident report explicitly blaming *ANN* recall (rather than ingestion lag or pipeline staleness) for missing new documents.
- I found no Qdrant-specific GitHub issue on recall degradation after inserts.
- The Chroma #7963 numbers come from one user and have no maintainer confirmation.

## Q3. Which vendors brute-force search the newest data (hiding the bias), and which do not

### Takeaway
Most do: Pinecone, Turbopuffer, LanceDB, Qdrant, Chroma (single-node) and Milvus all search the freshest data exactly, or with a small temporary index. That hides any bias against new items until the data is folded into the main index. Single-graph engines (pgvector HNSW, Vespa, Redis, Cosmos DiskANN, and Weaviate HNSW once the queue drains) put new vectors directly into the main graph, so any in-degree bias against late inserts is visible immediately.

### Cited Findings
- **Brute force or exact for the freshest data:**
  - Pinecone memtable: "exact brute-force scan in memory for the most recent writes (up to ~10k vectors)." — [Pinecone](https://www.pinecone.io/learn/slab-architecture/)
  - Turbopuffer: "slower exhaustive search of recent data in the log" for strongly consistent queries. — [turbopuffer](https://turbopuffer.com/docs/architecture)
  - LanceDB: "exhaustive/flat search on the new data." — [LanceDB](https://docs.lancedb.com/indexing/reindexing)
  - Qdrant: appendable segments "have no vector index" and are scanned. — [Qdrant](https://qdrant.tech/documentation/ops-optimization/optimizer/)
  - Chroma: brute-force buffer of `batch_size` = 100. — [Chroma](https://docs.trychroma.com/docs/collections/configure)
- **Temporary or interim index for the freshest data:**
  - Milvus: the interim index on growing segments is on by default (IVF-style with nlist ≈ sqrt(chunkRows)), with brute force over partial chunks. — [Milvus config](https://milvus.io/docs/configure_querynode.md)
  - Pinecone: L0 slabs use "lightweight indexing", and the freshness layer keeps a "compact fresh index". — [Pinecone](https://www.pinecone.io/learn/slab-architecture/); [Pinecone blog](https://www.pinecone.io/blog/serverless-architecture/)
  - Elasticsearch/Lucene: new documents land in small new per-segment HNSW graphs that every query fans out to. — [Elastic](https://www.elastic.co/search-labs/blog/hnsw-graphs-speed-up-merging)
- **Inserted straight into the main graph:**
  - Vespa: "true real time," one mutable graph. — [Vespa](https://docs.vespa.ai/en/querying/approximate-nn-hnsw)
  - pgvector HNSW: no training step and no segments. — [pgvector](https://github.com/pgvector/pgvector)
  - Cosmos DB DiskANN: in-place insert and delete. — [Microsoft DevBlogs](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/)
  - Redis: background workers, then the HNSW index. — [Redis](https://redis.io/docs/latest/develop/ai/redisvl/0.30.0/concepts/search-and-indexing/)
  - Weaviate HNSW: goes through the async queue, then into the HNSW graph. Whether the queue is brute-forced is unverified. — [Weaviate](https://docs.weaviate.io/weaviate/concepts/vector-index)
- **Opt-outs that expose the bias deliberately:** Qdrant `indexed_only` and LanceDB `fast_search=True` skip unindexed data. Turbopuffer's eventually consistent queries cap the unindexed scan at 128 MiB. — [Qdrant](https://qdrant.tech/documentation/tutorials-operations/prevent-unoptimized-usage/); [LanceDB](https://docs.lancedb.com/indexing/reindexing); [turbopuffer concepts](https://turbopuffer.com/docs/concepts)

### Inferences
- The brute-force buffer only covers roughly the last 10k–100k vectors, or minutes to hours of writes. In a continuously growing collection, last week's inserts already live in a merged graph. If the merge is incremental (Lucene copies the largest graph and inserts the rest), they are late inserts there. So the bias is masked only for the very newest window.
- A fix in a CPU in-memory single-graph engine is most directly relevant to pgvector-, Vespa-, Redis- and hnswlib-style designs (Chroma single-node and Weaviate use hnswlib or their own HNSW), and to Lucene's merge path.

### Gaps
- Not verified: Weaviate's async-queue query path, Qdrant's merge-time HNSW construction, Redis internals.

## Q4. How common is streaming ingestion in RAG and search workloads

### Takeaway
I found no independent survey that quantifies it. The evidence is vendor statements that indexes are "rarely static," plus a large amount of vendor content (Confluent, RisingWave, Oracle) promoting CDC/streaming pipelines for RAG. The fact that nearly every vendor invests in freshness (LSN headers, WAL exhaustive search, async queues) is indirect evidence that continuous growth is the norm.

### Cited Findings
- Pinecone (vendor, 2024-01-16): "Very few of our use cases are primarily static corpora that only tend to serve queries." Also "indexes in Pinecone are rarely static", with Notion cited for varied per-user update frequency. — [Pinecone serverless blog](https://www.pinecone.io/blog/serverless-architecture/)
- Confluent (vendor) argues that nightly batch re-embedding leaves embeddings "inherently stale" and promotes CDC streaming into vector stores. — [Confluent blog](https://www.confluent.io/es-mx/blog/enterprise-knowledge-management-with-rag-for-digital-native-companies/)
- RisingWave (vendor) promotes CDC → stream processing → embeddings → vector store. — [RisingWave blog](https://risingwave.com/blog/streaming-for-rag-real-time-retrieval-augmented-generation/)
- Oracle (vendor) takes a contrary view: operational data should usually stay in SQL rather than be re-embedded on every change. — [Oracle developers blog](https://blogs.oracle.com/developers/real-time-rag-live-sql-incremental-indexing-and-freshness-tests)
- Microsoft (vendor) says streaming runbooks with inserts, deletes and replaces reflect real workloads. Its work builds on the NeurIPS'23 BigANN streaming-track style of runbook. — [Microsoft DevBlogs](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/)
- One page's claim of "40–60% accuracy improvements" from real-time RAG has no source. It is not reliable and should not be used. — [Markaicode](https://www.markaicode.com/build-rag-systems-real-time-data-updates/)

### Inferences
- Continuous growth (append-mostly, with some updates) is very likely the dominant production pattern for RAG and agent-memory workloads, going by the design investment of every vendor. Its exact share is unmeasured in public sources.

### Gaps
- I found no named industry survey (Gartner, a vendor's annual report, an academic user study) giving the share of vector workloads that use streaming ingestion.
- I found no vendor-published distribution of insert rates or of collection growth curves.
