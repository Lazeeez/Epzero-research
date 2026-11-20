# Industry and production evidence on ANN index staleness and maintenance, and the 2026 "When to Repair" paper (adversarial novelty check)

Claim under test: "nobody measures an ANN index's share of retrieval-quality drift over time on real evolving collections, or whether in-place maintenance beats rebuilding by retrieval quality."

Search effort: 13 web searches and 13 page fetches (2 Pinterest fetches returned HTTP 403; the USENIX Quake PDF returned no usable text). Run on 2026-11-12. Verdict labels: **subsumes** (already does what the proposal claims nobody does), **partially overlaps**, **different**.

## 1. The 2026 "When to Repair" paper: full characterisation

### Takeaway
The paper is real, published in 2026, and the closest prior work. But it is a synthetic-churn, recall-only study on SIFT and Fashion-MNIST. It compares fixed-cadence repair with signal-triggered repair and never compares either against a full rebuild. It uses no real evolving collection and no relevance metric, so it does not subsume the claimed gap. Its v2 is a negative result: the signal-triggered trigger does not beat a fixed clock.

### Cited Findings
- **Citation.** Madhulatha Mandarapu and Sandeep Kunkunuru (VaidhyaMegha Private Limited, India; samyama.ai), "When to Repair a Graph ANN Index: A Matched-Budget Negative Result, and the Interpolated-Baseline Trap That Hid It," arXiv:2607.00728v2 [cs.DB], July 2026 (v2 dated 10 July 2026) — [arXiv v2](https://arxiv.org/html/2607.00728), [PDF](https://arxiv.org/pdf/2607.00728)
- **v1 title and claim.** v1 was titled "When to Repair a Graph ANN Index: Navigability-Signal-Triggered Local Repair Protects Tail Recall Under Bursty Churn." It claimed that signal-triggered repair "Pareto-dominates fixed-cadence repair." The claimed tail recall@10 gains at about one consolidation were +0.014 (SIFT) and +0.050 (Fashion-MNIST), with 95% CIs excluding zero over four stream seeds. Mean-recall gains were under 0.005, and the probe signal correlated with true recall at Spearman ρ ≈ 0.95. v1 does not compare against a complete rebuild — [arXiv v1](https://arxiv.org/html/2607.00728v1)
- **Index and setup.** DiskANN Vamana, in-memory, via diskannpy 0.7.0. Builds are single-threaded and deterministic — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Datasets.** SIFT-128 (L2; 20k/10k live set) and Fashion-MNIST-784 (L2; 10k live set), with about 10× turnover. A scale run used a 100k live set — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Churn model.** A synthetic bursty stream alternates delete-heavy and insert-heavy phases. No real evolving collection is used — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Policies.**
  - P0: no repair (lazy tombstoning only).
  - P1: fixed-cadence consolidation every c operations.
  - P2: consolidate when the probe-set recall@10 signal s(t) drops δ below its post-repair baseline.
  - Source: [arXiv v2](https://arxiv.org/html/2607.00728)
- **Metric.** Per-window recall@10 against an exact live-set oracle that is recomputed every window (never a frozen ground truth). The tail metric is the per-window minimum — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Headline result (v2).** At matched repair budget, signal-triggered repair does "not beat fixed-cadence repair on tail recall—on either dataset, at any of three graph degrees, or at a 5× larger index" — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Table 1 numbers.**
  - SIFT-128 (δ=0.030): −0.0005±0.0074, against v1's +0.0134.
  - Fashion-MNIST (δ=0.008): +0.0015±0.0044, against v1's +0.0478.
  - No CI excludes zero.
  - Source: [arXiv v2](https://arxiv.org/html/2607.00728)
- **Graph degree.** "Sparser graphs really do drift more," but no degree yields an advantage. At R=24 the delta is −0.0128±0.0006. At 100k every direct point estimate is negative, the best being −0.0008 — [arXiv v2](https://arxiv.org/html/2607.00728)
- **The interpolated-baseline trap.**
  - v1's P1 grid held only {4, 10, 20, 34} passes, so P1 at one pass was linearly interpolated.
  - The recall-vs-passes curve is sharply concave: one pass captures 53.7% of the total gain, while interpolation credits only 25%.
  - That understates the baseline by about 1.6 recall points, which is more than v1's 1.34-point effect.
  - Source: [arXiv v2](https://arxiv.org/html/2607.00728)
- **Leading-indicator test falsified.** Spearman ρ between the signal and the *subsequent* recall drop is ρ ∈ [0.011, 0.362]. 0 of 16 cells meet the pre-registered |ρ| ≥ 0.6, and 14 of 16 fall in the falsification region |ρ| < 0.3 — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Budget accounting.** Equal pass count is not equal spend: P2 clears 0.77–2.59× different tombstone counts. 28 of 30 matched comparisons fall outside a ±5% parity band, while wall time is 0.70–1.09× — [arXiv v2](https://arxiv.org/html/2607.00728)
- **Negative controls.**
  - NC1, static: passes.
  - NC2, insert-only: the signal fires 28 times with zero deletes, which is 76% of the churn rate. So the signal tracks growth, not delete damage.
  - NC3, budget parity: fails, because parity erases the effect.
  - NC4, oracle: unchurned and fresh indexes agree to 0.0000; an orphaning fixture shows repair restores +0.0735 min-recall.
  - Source: [arXiv v2](https://arxiv.org/html/2607.00728)
- **Stated or evident scope limits.** Two datasets, three degrees, one synthetic churn generator and one signal. No nDCG or relevance metric, no real evolving collection, no full-rebuild arm, and no attribution of drift to the index versus the embeddings or corpus — [arXiv v2](https://arxiv.org/html/2607.00728)

### Inferences
- **Verdict: partially overlaps.** It owns the "repair-timing policy, measured against an exact live-set oracle" question on synthetic data. It does not touch (a) real evolving collections, (b) relevance judgments or nDCG, (c) in-place repair versus full rebuild, or (d) the index's share of retrieval-quality drift. The proposal should cite it as the direct predecessor.
- Its methodology (a live-set oracle recomputed per window, matched budgets run directly rather than interpolated, negative controls) is the bar a new study will be judged against.
- Its negative result is a warning for any "signal-triggered" pitch: on synthetic churn the first consolidation buys most of the recall, so cadence barely matters. A new study should test whether that still holds on real churn.

### Gaps
- The authors' affiliation is a small company. I found no peer-reviewed venue for the paper, and it is arXiv-only.

## 2. Industry papers and engineering blogs on refresh cadence, staleness and online recall

### Takeaway
Industry papers and blogs describe refresh cadences and online ANN-vs-KNN recall tools, but none I found publishes a time series that separates index-induced loss from embedding or corpus staleness, measured by relevance or engagement. Pinterest is the closest: it has an online recall tool and A/B-tests index choices, but publishes no drift-over-time attribution.

### Cited Findings
- **Pinterest, online recall tool.** Pinterest built "an online recall tool by adding functionality in [its] serving cluster to compute both ANN and KNN results," which lets it compute recall at a point in time. It also ran online A/B tests to pick quantizers (PQ vs SQ across HNSW and IVF). This comes from a search snippet; both direct fetches returned 403 — [Pinterest Engineering, "Evolving Pinterest's Embedding Retrieval Platform"](https://medium.com/pinterest-engineering/evolving-pinterests-embedding-retrieval-platform-aede4e831e01)
- **Pinterest, realtime HNSW.** Pinterest runs a realtime HNSW serving system (Manas HNSW Realtime). Its details on deletes, compaction and recall could not be fetched (403). Pinterest is also reported to be exploring cutting item-embedding refresh latency from hours to minutes — [Pinterest Engineering, Manas HNSW Realtime](https://medium.com/pinterest-engineering/manas-hnsw-realtime-powering-realtime-embedding-based-retrieval-dc71dfd6afdd)
- **Instacart.** Product embeddings are pre-computed and the ANN indices are trained offline daily, which is a cadence statement with no drift measurement — [Instacart, SIGIR eCom'22](https://sigir-ecom.github.io/ecom22Papers/paper_8392.pdf)
- **Facebook Search (KDD'20).** Covers ANN parameter tuning in embedding-based retrieval. I did not find a staleness-over-time measurement in the snippet — [Huang et al., KDD 2020](https://arxiv.org/pdf/2006.11632)
- **Airbnb (2026).** Benchmarked IVF vs HNSW in FAISS and chose IVF. The paper notes a high volume of realtime listing updates — [Applying Embedding-Based Retrieval to Airbnb Search, arXiv:2601.06873](https://arxiv.org/pdf/2601.06873)
- **Walmart.** Uses one year of engagement data for evaluating embedding-based retrieval, focused on model relevance rather than index staleness — [arXiv:2408.04884](https://arxiv.org/pdf/2408.04884)
- **"The Staleness Frontier" (ThakiCloud AI Research, Hyojung Han, 2026).**
  - It is embedding staleness of a ~2,000-skill router, not ANN structure.
  - It claims top-1 drops of 2.8 points in 7 days and 9.1 in 30 days, plus a square-root re-embedding cadence law.
  - But the summary states that "no experiments actually ran" and its parameters lack measurement support.
  - It is unreliable and has no arXiv ID.
  - Source: [HF dataset card](https://huggingface.co/datasets/thaki-AI/daily-paper-2026-09-15-staleness-frontier-frozen-index-re-embedding)
- **indexical.dev blog (Travis Jordan).**
  - It reports hybrid BM25+dense recall@10 of 72.1% → 57.9% after 90 days on 10 repositories.
  - That loss is from embedding-to-content drift (stale chunks), not ANN structure.
  - The underlying "Stanford drift study" is not linked to any primary source, so treat it as unverified SEO-grade content.
  - Source: [indexical.dev](https://indexical.dev/blog/reindex-beats-rerank-past-seven-days-of-index-drift.php)

### Inferences
- **Verdict on industry papers: different or partially overlapping.** Industry reports cadence (daily rebuilds, minute-level embedding freshness) and point-in-time online recall. They do not report decomposed drift (index vs embeddings vs corpus vs queries) or in-place versus rebuild compared by relevance over time. Pinterest is the most likely to have this data internally but, in what I could access, has not published it.
- Blog-sphere figures on "staleness" nearly always mean content or embedding staleness, not ANN graph degradation. The proposal can use this confusion as motivation for separating the two.

### Gaps
- I could not read Pinterest's posts in full (403), so they may contain recall-over-time numbers.
- I did not find KDD ADS, WWW, RecSys, SIRIP or VLDB industry papers from Spotify, LinkedIn, Meta, Amazon, Alibaba, Uber or Shopify that report index-refresh impact on engagement. This is absence of evidence from limited searches, not proof of absence. LinkedIn's "within 1 minute" embedding freshness appeared only via a secondary aggregator snippet and is not cited as fact.

## 3. Vector database vendors: quantified drift and live recall monitoring

### Takeaway
Live recall monitoring by sampling exact search is established prior art: turbopuffer measures 1% of production queries, and Pinterest, Vespa and Vertex all offer some form of exact-vs-approximate comparison. So "monitoring recall" alone is not novel. Vendors publish streaming-recall stability charts (Cosmos DB/DiskANN) and anecdotal churn degradation, but no relevance-level attribution on real evolving collections.

### Cited Findings
- **turbopuffer (2024-09-04, Morgan Gallant).**
  - It "automatically samples 1% of all queries" and runs both ANN and exhaustive search.
  - It targets 90–95% recall@10 including filtered queries, with monitors on thresholds.
  - It claims: "To our knowledge, continuous recall measurement is not something any other search engine or database does."
  - It publishes no drift-over-time numbers. It also has a recall-evaluation API endpoint.
  - Sources: [turbopuffer blog](https://turbopuffer.com/blog/continuous-recall), [turbopuffer docs: recall](https://turbopuffer.com/docs/recall)
- **Vespa.** Its docs prescribe measuring the recall loss by running `{approximate:false}` exact search, comparing it with approximate results, and computing overlap@k with distance-tie handling. This is a manual tool, not a time series — [Vespa docs, ANN with HNSW](https://docs.vespa.ai/en/querying/approximate-nn-hnsw.html)
- **Google Vertex AI Vector Search.** The official monitoring page exposes only two IndexEndpoint metrics: shard count and active replicas. I found no built-in recall monitor. Brute-force indexes exist for ground truth, and recall-vs-brute-force is a user-run practice — [Vertex AI Vector Search monitoring](https://cloud.google.com/vertex-ai/docs/vector-search/monitor?hl=pt-br), [third-party guide (oneuptime)](https://oneuptime.com/blog/post/2026-02-17-how-to-configure-vertex-ai-vector-search-indexes-for-billion-scale-similarity-search/view)
- **Azure Cosmos DB with DiskANN, Part 4 (2025-05-07; Codella, Singh, Upreti).**
  - Recall is measured over time under streaming runbooks:
    - MSTuring-1M and Wiki-Cohere-10M expiration runbooks, with lifespans of 100/50/10 steps in 1:2:10 proportions.
    - MSTuring-10M clustered runbook: 64 k-means clusters, 5 rounds of cluster-wise insert and delete.
  - It compares in-place delete against basic deletion and claims stable recall with no full rebuild (IndexBuilder plus IndexVacuum).
  - Numbers are only in charts, which I could not read.
  - Source: [Microsoft Cosmos DB blog](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/)
- **Third-party Qdrant churn anecdote.** Text recall was 0.960 at 50 candidates when fresh. After update churn it was 0.925 even at 400 candidates, and a 0.95 target was "unreachable without a rebuild." This is from a GitHub PR benchmark of unknown rigor — [GitHub PR](https://github.com/muzakkirhussain011/needtoknow/pull/68)
- **Practitioner guidance (not evidence).**
  - "Running recall measurement on a schedule—hourly, or triggered by index update volume thresholds" — [Ken Walger](https://www.kenwalger.com/blog/ai/vector-search-at-scale-why-your-index-isnt-as-healthy-as-you-think/)
  - Rebuild if corpus change exceeds 15%, recall drops more than 2–3 points, or p99 grows more than 20%. These are uncited heuristics — [indexical.dev](https://indexical.dev/knowledge/what_are_the_best_vector_index_rebuild_strategies_for_enterprise_ai_retrieval_systems_in_2026.php)

### Inferences
- **Verdict.**
  - Live recall monitoring partially overlaps (turbopuffer and Pinterest do it in production).
  - Cosmos DB/DiskANN partially overlaps: recall over time under streaming, in-place vs naive delete, but synthetic runbooks, recall only and no rebuild arm visible.
  - None subsumes the claim, because all measure recall@k against exact search, not retrieval quality against relevance judgments, and none decomposes drift.
- The proposal must not claim novelty for "monitoring recall by sampling exact search." Its novelty has to rest on connecting index recall to relevance (nDCG) on a longitudinal IR collection and attributing the drift.

### Gaps
- Not covered by targeted searches (budget): Elastic/OpenSearch recall-estimation features, Pinecone, Zilliz/Milvus, Weaviate, Qdrant official docs and LanceDB. No quantified vendor drift numbers were found for them. This is absence of evidence only.

## 4. 2024–2026 research on maintenance and rebuild policies for dynamic vector indexes

### Takeaway
Systems research thoroughly covers in-place update versus global rebuild, but measures it by recall@k against exact neighbours, on synthetic or semi-synthetic workloads. Quake is the strongest counterexample: it uses a Wikipedia-derived evolving workload and shows baseline recall degrading over time. SPFresh explicitly compares in-place update against global rebuild. None uses relevance judgments.

### Cited Findings
- **Quake (OSDI 2025).**
  - Authors: Mohoney, Sarda, Tang, Chowdhury, Pacaci, Ilyas, Rekatsinas, Venkataraman.
  - It uses adaptive multi-level partitioning driven by a cost model, plus a recall-estimation model that sets query parameters to meet recall targets.
  - Against SVS, DiskANN, HNSW and ScaNN it reports 1.5–38× lower query latency and 4.5–126× lower update latency.
  - It builds "a Wikipedia vector search workload and ... a workload generator."
  - Per the search snippet, "Quake maintains a stable recall of near 90%, while LIRE's recall degrades over time as it uses a static nprobe."
  - Sources: [arXiv:2506.03437](https://arxiv.org/abs/2506.03437), [USENIX OSDI'25](https://www.usenix.org/conference/osdi25/presentation/mohoney)
- **SPFresh / LIRE (SOSP 2023).** Incremental in-place rebalancing of partitions. It reports "superior query latency and accuracy to solutions based on global rebuild" on a billion-scale disk index with a 1% daily update rate, using about 1% of the DRAM and under 10% of the cores at peak — [arXiv:2410.14452](https://arxiv.org/html/2410.14452), [MSR](https://www.microsoft.com/en-us/research/publication/spfresh-incremental-in-place-update-for-billion-scale-vector-search/)
- **Ada-IVF (arXiv:2411.00970).** Tracks per-cluster size, centroid drift and access "temperature," and triggers localized re-clustering when thresholds are crossed. It reports 2–5× update throughput over rebuild or naive incremental baselines. This is a drift-triggered maintenance policy for IVF — [arXiv:2411.00970](https://arxiv.org/abs/2411.00970)
- **DeDrift (ICCV 2023, arXiv:2308.02752).** Robust similarity search under content drift: IVF/PQ codebook refresh as the data distribution shifts over time — [arXiv:2308.02752](https://arxiv.org/pdf/2308.02752)
- **LSM-VEC (arXiv:2505.17152).** Recall@10: Balanced workload 88.4% (vs SPFresh 75.5%, DiskANN 82.0%); Delete-heavy 77.4% (vs DiskANN 61.0%). DiskANN update latency is 11.86 ms against LSM-VEC's 4.60 ms. It is out-of-place and needs no global reconstruction — [arXiv:2505.17152](https://arxiv.org/pdf/2505.17152)
- **Pathak, "Coverage Is Not Redundancy" (arXiv:2608.16043, 2026-08-17).**
  - It studies admission control for retrieval hubs on HNSW, IVF-Flat and IVF-PQ under workload drift.
  - The exposure floor "worsen[s] as index recall declines."
  - It is validated on real workload drift. A search snippet attributes a Wikipedia clickstream replay across COVID-19 emergence (Nov–Dec 2019 → Mar–Apr 2020) to this or a related paper; the attribution is unconfirmed.
  - It is implemented in pgvector with 0.33% overhead.
  - Source: [arXiv:2608.16043](https://arxiv.org/abs/2608.16043)
- **2026 bachelor thesis, University of Piraeus.** Compares SPFresh, DeDrift and Quake maintenance policies plus a contextual bandit on IVF, with fixed-recall evaluation — [GitHub](https://github.com/kostasb1/bitakos-e20108-index-maintenance)
- **Elliott & Clark (arXiv:2405.17813).** Shows that data, insertion ordering and intrinsic dimensionality affect HNSW recall. This is relevant because in-place-grown and rebuilt indexes differ in insertion order — [arXiv:2405.17813](https://arxiv.org/pdf/2405.17813)
- **LongEval-Sci diagnostic (arXiv:2607.04088).** Temporal BM25 and Qwen3 dense baselines over three snapshots by nDCG@10. It does not isolate approximate vs exact or incremental vs rebuild — [arXiv:2607.04088](https://arxiv.org/abs/2607.04088)

### Inferences
- **Verdicts.**
  - Quake: partially overlaps. It is the strongest threat because it has a Wikipedia-based evolving workload and recall over time, but it measures recall against exact search, not relevance, and does not decompose drift.
  - SPFresh: partially overlaps. It answers "in-place vs global rebuild" by recall and latency, not by retrieval quality.
  - Ada-IVF and DeDrift: partially overlap, since they are drift-triggered maintenance policies.
  - LSM-VEC: different. It is a systems design compared by recall.
  - Pathak 2026: different, on hub admission.
  - LongEval-Sci diagnostic: different. It is longitudinal IR without index attribution.
- **Overall.** No item found subsumes the claim. The gap survives in narrowed form: "on a real longitudinal IR collection with relevance judgments, (1) attribute nDCG drift to ANN approximation versus model, corpus or query change by running exact and ANN side by side on each snapshot, and (2) compare in-place maintenance versus rebuild by nDCG, not only recall@k." The proposal should drop any blanket "nobody measures index recall over time" wording, since Quake, Cosmos DB/DiskANN, turbopuffer, Pinterest and "When to Repair" all do.
- **Risk.** On real collections the ANN share may be tiny (recall around 0.95 typically moves nDCG very little). This could make the headline a "negligible share" finding. That is still publishable but should be pre-registered.

### Gaps
- I could not extract Quake's Wikipedia-12M workload details (time span, monthly insert volumes, page-view query sampling) or its per-baseline recall-over-time numbers: the PDF fetch failed. These need direct reading of osdi25-mohoney.pdf §Evaluation.
- Greator and "self-tuning vector index" papers were not specifically located in this pass.
- FreshDiskANN (2021) streaming recall stability is known prior art from background knowledge but was not re-verified this session.
- I did not verify whether Big-ANN NeurIPS'23 streaming-track runbooks (the source of the Cosmos DB runbooks) include any relevance metric. I believe they do not.
