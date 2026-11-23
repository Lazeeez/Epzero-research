# Staleness of partition-based (IVF) ANN indexes and vector quantizers under growth and drift

Scope note: compiled 2026-11-23. "PR" = peer-reviewed; "preprint" = arXiv only, no venue found. Numbers marked (text) were read from the paper text. Numbers that appear only in figures could not be read and are listed as gaps. PDFs for DeDrift, HAKES and PQR-Search were downloaded and converted to text with pdftotext, so quotes from them are first-hand.

**Headline answer to the overarching question:** No paper found measures retrievability broken down by item age or insertion time for IVF or quantized indexes. Every system paper reports aggregate recall, QPS or latency over time. Measured IVF staleness is always tied to a distribution change, either real temporal drift (DeDrift, Quake's Wikipedia workload) or synthetic cluster-ordered insertion (Ada-IVF, SPFresh skew, CoDEQ, PQR-Search). Under i.i.d. or random insertion, IVF recall effects are small (HAKES: "degrades slightly"; MicroNN: "drifts only slightly"). Quake/Ada-IVF report IVF staleness mostly as a cost (latency/QPS) problem, not a recall problem. Recall loss shows up when nprobe is held fixed.

## 1. Systems for updatable partition indexes: what drift they measure, how, and with what numbers

### Takeaway
SPFresh/LIRE, Ada-IVF, Quake, Ubis, DeDrift, MicroNN and PQR-Search all handle IVF maintenance (split, merge, reassign, recenter). They measure partition imbalance, reconstruction error, latency/QPS and aggregate recall over update steps. Their drift is mostly synthetic cluster-ordered streams. The only real-time-ordered workloads are DeDrift's VideoAds/YFCC and Quake's Wikipedia-12M. None breaks recall down by item age.

### Cited Findings
**SPFresh / LIRE (Microsoft Research et al.; SOSP 2023, PR; arXiv 2410.14452 posted Oct 2024)**
- LIRE is a "lightweight incremental rebalancing protocol". It splits partitions and reassigns boundary vectors, and is built on the premise that one update to a well-balanced index only changes a local region — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1); [MSR PDF](https://microsoft.com/en-us/research/uploads/prod/2023/08/SPFresh_SOSP.pdf)
- Motivating measurement (text, Fig. 2, SPANN on SIFT): a static 2M index compared with 1.5M base + 0.5M in-place appended vectors. "Updating one-third of the vectors degrades the query recall by more than one point and increases tail latency by 4X." — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)
- Workloads (text): SPACEV and SIFT 100M subsets at a 1% daily update rate for 100 days, with random 1% deletes and 1% inserts from a separate pool. The billion-scale stress test runs 20 days. There are no real timestamps. "Skew" datasets are synthetic — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)
- SPANN+ (append without split/reassign): P99.9 latency went "from 4ms to more than 10ms". The accuracy gap to SPFresh "increases over time" because "the index quality of SPANN+ degrades as partition distributions skew over time". In the stress test SPFresh recall stayed "higher than 0.862 for the uniform dataset and 0.807 for the skew dataset" (top-64 postings). Per-day recall values are only in figures — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)
- Rebalancing is rare: about 0.4% of inserts triggered it, with 2 splits on average (max 160) and about 0.1% merges. Each reassignment evaluated 5,094 vectors on average and moved 79 — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)
- No breakdown by item age — [arXiv 2410.14452](https://arxiv.org/html/2410.14452v1)

**Ada-IVF — "Incremental IVF Index Maintenance for Streaming Vector Search" (Mohoney et al., arXiv 2411.00970, 1 Nov 2024; preprint, no venue found)**
- Ada-IVF tracks global partition-size std (σ) and reconstruction error (MSE to the assigned centroid). Per partition it tracks size deviation, centroid drift (relative change of the partition mean) and "read temperature" (access frequency). It re-clusters locally — [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)
- Workloads: an industrial recommendation trace on MSTuring10M, BigANN-SS (30M MSTuring, cluster-based insert/delete batches) and a synthetic cluster-sampling generator. There is no real temporal drift — [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)
- Table 3, QPS at recall 0.9 normalised to full Rebuild:
  - Internal trace: Frozen 0.81, Update 0.84, LIRE 0.83, Ada-IVF 0.89
  - BigANN-SS: Frozen 0.27, Update 0.54, LIRE 0.89, Ada-IVF 0.85. The text says Frozen gets "3.5×" less QPS than Rebuild.
  - Update throughput vs Rebuild: Frozen 20–33×, Ada-IVF 6.7–10×, LIRE 4.2–4.7×

  [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)
- Frozen centroids "is only effective when the distribution of vectors and queries is uniform". Frozen's decent result on the Internal trace is attributed to the large initial dataset giving a representative clustering. Fig. 3 shows Frozen's reconstruction error drifting above Rebuild's as inserts accumulate — [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)
- Abstract: "an average of 2× and up to 5× higher update throughput" than the best dynamic IVF baselines. Internally the paper is inconsistent, stating "1.5−5×" in one place and "2−5×" in another — [arXiv 2411.00970](https://arxiv.org/abs/2411.00970v1)
- No age or insertion-time breakdown. "Temperature" is access recency, not data age — [arXiv 2411.00970](https://arxiv.org/html/2411.00970v1)

**Quake — "Quake: Adaptive Indexing for Vector Search" (Mohoney et al., OSDI 2025, pp. 153–169, PR; arXiv 2506.03437)**
- Hierarchical, cost-model-driven partition maintenance plus APS (adaptive partition scanning), which sets nprobe per query from a recall estimator. OSDI abstract: 1.5–38× lower query latency and 4.5–126× lower update latency vs SVS, DiskANN, HNSW and ScaNN. arXiv v1 said 1.5–22× and 6–83× — [USENIX](https://www.usenix.org/conference/osdi25/presentation/mohoney); [arXiv v2](https://arxiv.org/html/2506.03437v2)
- **Real-temporal workload, Wikipedia-12M:** built from monthly Wikipedia page additions and page-view frequencies, April 2013–December 2021. It grows from 1.6M to 12M vectors over 103 monthly updates (~100K per update), each followed by 100K searches sampled by page views. Vectors are DistMult graph embeddings with inner-product metric — [arXiv 2506.03437v2](https://arxiv.org/html/2506.03437v2)
- Results on Wikipedia-12M:
  - "LIRE's recall degrades over time as it uses a static nprobe". Table 3 marks LIRE as "unable to meet the recall target with static query parameters".
  - Faiss-IVF without maintenance needed 165.8 h of search time. The paper attributes this to skew-driven latency growth.
  - DeDrift recall "stays relatively constant" but its latency "increases significantly with time".
  - Recall std on this workload: 0.005–0.008 with APS vs 0.025 without.
  - No per-age breakdown.

  [arXiv 2506.03437v2](https://arxiv.org/html/2506.03437v2)
- APS vs fixed nprobe (static SIFT1M, 1000 partitions, target 90%): APS 91.2% recall at nprobe 20.2, 0.48 ms, 0 s tuning. Fixed: 90.3%, nprobe 19, 0.44 ms, 330 s tuning. Auncel: 98.1%, 1.29 ms — [arXiv 2506.03437v2](https://arxiv.org/html/2506.03437v2)

**UBIS — "Updatable Balanced Index for Stable Streaming Similarity Search over Large-Scale Fresh Vectors" (Lai, Huang, Wang, Wuhan Univ.; arXiv 2602.00563, Jan/Feb 2026; preprint)**
- Cluster-based index built on SPFresh/SPANN. It targets "update congestion and imbalanced data distribution" under streaming and relaxes balance constraints for skew. It claims "up to 77% higher search accuracy with 45% higher update efficiency" than state-of-the-art indexes (self-reported) — [arXiv 2602.00563](https://arxiv.org/html/2602.00563v1)

**MicroNN (Apple; arXiv 2504.05573, 8 Apr 2025; preprint)**
- On-device IVF with a delta store that every query scans. Flushes assign vectors to the nearest centroid and then update centroids. A full rebuild fires when average partition size grows 50%. In a growing-collection run (start at 50% of data, +3% per epoch) the authors say recall "drifts only slightly" from a full-rebuild baseline. Numbers are only in Fig. 10b. Maintenance costs under 2% of rebuild I/O — [arXiv 2504.05573](https://arxiv.org/html/2504.05573)

**HAKES (Hu et al., PVLDB 18(9):3049–3062, 2025, PR)**
- Drift-tolerance experiment (text, Fig. 11, MBNET-1024):
  - Setup: a 1M ImageNet-derived index with 1/10 of categories held out. Four batches of 200K inserts and 1K queries are drawn with a mixing ratio of held-out categories from 0 to 0.8, so inserts and queries both drift.
  - Result: "As more data are added the recall degrades slightly. The result showed the robustness of IVF and HAKES-Index training process against moderate drifts for embeddings from the same model." RSNET-2048 gave similar results. For embeddings from different models or sources they "recommend building different indexes".

  [PVLDB PDF](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)
- Compressing new vectors with search-side learned parameters lowers recall (Fig. 12): "the true neighbors in the added data can have significantly lower approximated similarity than those neighbors in the original dataset". This is the closest statement found to "new items are disadvantaged" in a quantized IVF. It is caused by HAKES's learned (decoupled) compression, not by IVF centroids — [PVLDB PDF](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)
- The paper also says that as a dataset grows considerably the index "should be rebuilt with a larger number of IVF partitions". It notes that embedding models are often retrained daily in recommenders, which forces rebuilds — [PVLDB PDF](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)

**Engines**
- Lucene/Elasticsearch int8 scalar quantization keeps per-segment quantiles. On merge it takes a weighted average and re-quantizes only if the merged quantiles diverge sharply from the originals. This is a built-in SQ staleness guard — [Elastic blog](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene); [Lucene ScalarQuantizer javadoc](https://lucene.apache.org/core/10_3_2/core/org/apache/lucene/util/quantization/ScalarQuantizer.html)
- Milvus builds indexes per sealed segment ("each segment carrying its own index"), so later data is indexed with structures trained on later data — [Milvus docs](https://milvus.io/docs/data_processing.md). The docs did not confirm that IVF centroids are trained per segment; see Gaps.
- pgvector IVFFlat assigns new rows to existing centroids, and REINDEX re-runs k-means. The sources are vendor/practitioner posts, not measurements — [pg_trickle blog](https://pgxn.org/dist/pg_trickle/0.36.0/blog/incremental-pgvector.html)

### Inferences
- Every IVF maintenance paper treats staleness as partition imbalance plus centroid–data mismatch, and uses QPS at fixed recall or latency as the headline metric. Without maintenance, IVF mainly gets slower: posting lists grow unevenly, and at fixed nprobe more vectors are scanned or neighbors fall in unprobed lists. That is consistent with a frozen-centroid IVF showing no new-vs-old miss bias on a growing, mildly drifting corpus. The only inputs are centroid geometry and the query, so with no strong drift, a new item in a dense region is no worse off than an old one.
- HAKES (PR) gives direct peer-reviewed support: IVF recall is "robust ... against moderate drifts for embeddings from the same model". Ada-IVF concedes Frozen is fine when the distribution is stable and the initial sample is large.

### Gaps
- No paper reports recall or miss-rate split by item age, insertion time or new-vs-old neighbors for any IVF system: SPFresh, Ada-IVF, Quake, UBIS, MicroNN, HAKES, DeDrift or PQR-Search. This was checked in each text and is explicitly absent.
- Exact per-step recall curves (SPFresh Figs. 7/9/10, Ada-IVF Figs. 2/3, Quake Fig. 4, MicroNN Fig. 10b) are figure-only and could not be extracted.
- "CTR" in the brief is not an IVF maintenance system. It is the Certify-then-Rectify recall-certification method for HNSW; see section 4.
- Not verified whether Milvus/Knowhere trains IVF centroids per segment. No Faiss wiki guidance on retraining under drift was found. Faiss provides no automatic rebalancing; Faiss-IVF is used as the "no maintenance" baseline in Quake.

## 2. DeDrift in detail (datasets, measured loss from stale centroids/codebooks)

### Takeaway
DeDrift (ICCV 2023, PR) is the main real-temporal-drift study. On VideoAds (86M) and YFCC (84M), IVF trained on the first months and then indexing later months loses up to ~11 points of 10-recall@10 at tight search budgets (IVF16384: 0.842→0.732). The loss shrinks to about 1 point at generous budgets. PQ/OPQ codec loss is under 1% relative. The protocol is a sliding window where the whole database post-dates training, so it does not test old vs new items coexisting in one index.

### Cited Findings
- Title "DEDRIFT: Robust Similarity Search under Content Drift". Authors Baranchuk, Douze, Upadhyay, Yalniz. ICCV 2023, pp. 11026–11035 (PR); arXiv 2308.02752, 5 Aug 2023 — [CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Baranchuk_DEDRIFT_Robust_Similarity_Search_under_Content_Drift_ICCV_2023_paper.html); [arXiv](https://arxiv.org/abs/2308.02752v1)
- Datasets:
  - **VideoAds:** 86M L2-normalised 512-d embeddings of ad videos published Oct 2020–Oct 2022 on a large social media site (RegNetY + Linformer, MoCo). Proprietary; no release statement found.
  - **YFCC:** 84M Flickr images, Jan 2007–Dec 2013, 384-d DINO ViT-S/16 embeddings, randomly rotated and 8-bit quantized.

  [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752); [arXiv PDF](https://arxiv.org/pdf/2308.02752)
- Drift evidence (§3.4):
  - For queries of month i, exact top-10 neighbors from the previous m months (m=12 VideoAds, m=36 YFCC): "recent content occurs more often in the results". YFCC shows a seasonal trend (snow, foliage, holidays).
  - "If the vectors were temporally i.i.d. search results would be equally likely to come from all previous m months."
  - Daily, weekly and monthly similarity matrices show drift over years.

  [arXiv PDF](https://arxiv.org/pdf/2308.02752)
- Protocol (§4.1):
  - The index is trained on months {i..i+m−1}, and months {j..j+m−1} are indexed, with m=3. "This is the maximum retention period of user data for many sharing platforms."
  - In-domain (ID): i=j. Out-of-domain (OOD): i=0 (the first months).
  - The window slides monthly, deleting the oldest month and inserting the next. Queries are 10,000 random vectors from month j+m.
  - Search budget is in distance computations (DCS). Results are reported for the month with the largest ID–OOD gap.

  [arXiv PDF](https://arxiv.org/pdf/2308.02752)
- Table 2, 10-recall@10, ID vs OOD (stale coarse quantizer, uncompressed):

  | Dataset | Index | 6000 DCS | 12000 DCS | 30000 DCS | 60000 DCS |
  |---|---|---|---|---|---|
  | VideoAds | IVF16384 | 0.842 vs 0.732 | 0.914 vs 0.845 | 0.966 vs 0.938 | 0.985 vs 0.973 |
  | VideoAds | IVF65536 | 0.914 vs 0.835 | | | 0.993 vs 0.985 |
  | VideoAds | RCQ10×4 | 0.651 vs 0.531 | | | |
  | YFCC | IVF4096 | 0.796 vs 0.744 | 0.892 vs 0.858 | 0.960 vs 0.945 | 0.983 vs 0.977 |
  | YFCC | IVF16384 | 0.894 vs 0.851 | | | 0.993 vs 0.989 |

  IMI and RCQ coarse quantizers also lose 1–12 points. "The content drift has a significant impact" — [arXiv PDF](https://arxiv.org/pdf/2308.02752)
- Table 1, codecs only, ID vs OOD (VideoAds | YFCC):

  | Codec | VideoAds | YFCC |
  |---|---|---|
  | PCA128 | 0.709 vs 0.698 | 0.625 vs 0.622 |
  | PCA256 | 0.844 vs 0.835 | 0.867 vs 0.864 |
  | PQ16 | 0.237 vs 0.237 | 0.164 vs 0.160 |
  | PQ32 | 0.441 vs 0.441 | 0.380 vs 0.377 |
  | OPQ16 | 0.457 vs 0.446 | 0.302 vs 0.294 |
  | OPQ32 | 0.609 vs 0.601 | 0.484 vs 0.477 |

  The paper calls this "acceptable". Codecs are "relatively robust to content changes", and the stated reason is small codebooks that can hardly overfit. Residual PQ encoding (relative to IVF centroids) is more drift-sensitive than direct encoding — [arXiv PDF](https://arxiv.org/pdf/2308.02752); [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752)
- DeDrift variants. There is no "Lite" variant.
  - **Split:** re-k-means a few clusters (k=8 YFCC, k=64 VideoAds). 160× (YFCC) and 250× (VideoAds) faster than Full.
  - **Lazy:** move centroids to the mean of their assigned vectors without reassigning, then recompute PQ look-up tables with historical centroids. 70× and 170× faster than Full.
  - **Hybrid:** Lazy plus Split.
  - Gains on YFCC are "5.4%, and 1.3%" at 6000 and 30000 DCS. The method "almost reaches Full on YFCC" and "significantly reduces the gap" on VideoAds.
  - Update cost for OPQ32 (Table 7): YFCC Split 2.1 s, Lazy 8.1 s, Hybrid 10.4 s, Full 29.8 s. VideoAds 10.8 s, 24.1 s, 33.2 s and 430.8 s.

  [arXiv PDF](https://arxiv.org/pdf/2308.02752); [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752)
- Outlier bursts (Table 3, YFCC with placeholder "removed image" vectors restored, IVF4096, Sept 2012):

  | Method | 6000 DCS | 12000 DCS | 20000 DCS | 30000 DCS | 60000 DCS |
  |---|---|---|---|---|---|
  | None | 0.726 | 0.865 | 0.917 | 0.950 | 0.976 |
  | Lazy | 0.720 | 0.796 | 0.832 | 0.868 | 0.946 |
  | Split | 0.796 | | | | |
  | Hybrid | 0.824 | 0.900 | 0.937 | 0.959 | 0.980 |
  | Full | 0.824 | 0.904 | 0.942 | 0.963 | 0.984 |

  Lazy degrades below no-reindexing — [arXiv PDF](https://arxiv.org/pdf/2308.02752)
- Evolving k-means (Appendix I) gives "slightly higher recall" but costs about as much as a full rebuild. In Lazy, more centroid-update iterations hurt because centroids move "too far away from the vectors already stored" — [arXiv PDF](https://arxiv.org/pdf/2308.02752)
- The paper does not separate query drift from database drift. It runs no shuffled/i.i.d. control and only argues that k-means would be optimal if data were temporally i.i.d. It gives no per-age breakdown — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752)

### Inferences
- DeDrift's OOD condition is "index entirely from the future, centroids entirely from the past". A growing collection (old plus new, centroids from the first snapshot) is a milder condition, because the old half is in-domain by construction. So DeDrift's 1–11 point losses are an upper-envelope estimate for growth scenarios, and they are largest only at small nprobe/DCS budgets.
- DeDrift's evidence that the coarse quantizer, not the codec, is the drift-sensitive part matches the PQR-Search and Quake framing (residual growth, imbalance).

### Gaps
- No public release statement for VideoAds was found. YFCC100M is public but DeDrift's exact 84M filtered subset and embeddings were not confirmed as released.
- No result in DeDrift isolates "recall of items indexed after training vs items indexed before" inside the same index.

## 3. Quantizer staleness (PQ/OPQ codebooks, SQ ranges, RaBitQ, learned quantizers)

### Takeaway
Evidence is thin and mostly preprint. DeDrift (PR) found PQ/OPQ codecs nearly drift-immune (under 1% relative loss), while residual-PQ inherits centroid drift. Newer preprints (CoDEQ 2512.18335; IVF-TQ 2605.17415) claim frozen PQ degrades under streaming, one even under i.i.d. ingestion, but give few text-readable numbers. SQ range overflow is only documented as engine mechanics: clipping and per-segment quantile recomputation. No measurement was found. RaBitQ's guarantee is data-independent in its error bound, but its centroid normalisation and IVF partitioning are still data-fitted.

### Cited Findings
- DeDrift (PR): PQ16/PQ32/OPQ16/OPQ32 codecs show ID–OOD gaps ≤0.011 absolute (Table 1 above). Residual encoding is more sensitive, and DeDrift-Lazy on residual OPQ32/VideoAds gains +3.7% at 6000 DCS and +1.4% at 30000 DCS over no update — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752)
- "Quantization for Vector Search under Streaming Updates" (Aden-Ali, Ferhatosmanoglu, Greaves-Tunnell, Mishra, Wagner; arXiv 2512.18335, 20 Dec 2025; preprint, cs.DS)
  - Abstract: data-dependent quantizers "cannot handle insertions and deletions", so quality degrades or a global rebuild is needed. The paper proves static data-dependent quantization can be made "dynamically consistent" with bounded disk I/O per update.
  - Experiments: synthetic drift on Deep/BigANN/Text2Image 100M (initial clusters, then inserts from other clusters). Baselines are FrozenPQ, DeDriftPQ, RebuildPQ and LSHForest; the proposed method is CoDEQ.
  - Fig. 5: the FrozenPQ–RebuildPQ gap widens over 20 iterations and grows faster when queries favour recent data (α>0). Exact numbers are not in the text.
  - The text says RaBitQ's preprocessing stays frozen on original data and that disk-free baselines including RaBitQ "show substantially worse recall performance".
  - No age breakdown.

  [arXiv abs](https://arxiv.org/abs/2512.18335); [HTML](https://arxiv.org/html/2512.18335v1)
- "IVF-TQ: Calibration-Free Streaming Vector Search via a Codebook-Free Residual Layer" (single author; arXiv 2605.17415, 17 May 2026, v2 22 May 2026; preprint, unreplicated)
  - Claims IVF-PQ degrades even under shuffled i.i.d. ingestion ("no shift at all"), by −3.8pp at sub-matched bit budgets. Per-batch PQ retraining "never recovers the streaming gap".
  - IVF-TQ uses a fixed random rotation plus a Lloyd-Max SQ and stays within [−0.80, +0.56]pp.
  - A search snippet cited a "2.31 points" SIFT-10M figure; this was not verified on the abstract page.

  [arXiv 2605.17415](https://arxiv.org/abs/2605.17415)
- RaBitQ (Gao & Long, SIGMOD 2024, PR; extended version SIGMOD 2025, arXiv 2409.09913): D-bit codes from a random orthogonal rotation of vectors normalised relative to a centroid (dataset or IVF cluster), with an unbiased estimator and a high-probability error bound — [arXiv 2405.12497](https://arxiv.org/abs/2405.12497v1); [arXiv 2409.09913](https://arxiv.org/html/2409.09913v1)
- PQR-Search (preprint) builds on this. Its guarantee requires that centroids and stage configuration be chosen independently of the rotation (condition C1/C5). Section 5.1 notes that residuals grow under updates as vectors drift from their centroids, which shrinks pruning margins and raises cost but not failure probability. Maintenance (recentering, split, dissolution) is chosen by a cost-gain rule (§5.3) — [arXiv 2611.07579](https://arxiv.org/pdf/2611.07579)
- SQ range staleness:
  - NVIDIA cuVS SQ learns a global float range and clips values outside it, with a quantile knob — [cuVS docs](https://docs.nvidia.com/cuvs/user-guide/api-guides/preprocessing-guide/scalar-quantizer)
  - Lucene recomputes per-segment quantiles on merge only when they diverge — [Elastic blog](https://www.elastic.co/search-labs/blog/scalar-quantization-in-lucene)
  - RSLM (arXiv 2608.30384, preprint) proposes per-vector scaled SQ to remove offline training and "guarantees robustness against out-of-distribution data shifts" (author claim) — [arXiv 2608.30384](https://arxiv.org/pdf/2608.30384)
- HAKES (PR): learned compression parameters applied to new inserts cause true neighbors among added data to get "significantly lower approximated similarity than those neighbors in the original dataset". This is the only found statement that is effectively a new-item disadvantage, and it is caused by a learned quantizer — [PVLDB PDF](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)

### Inferences
- For a quantizer, the RaBitQ family's error-bound guarantee holds regardless of data distribution. A stale centroid only inflates residual norms, which widens the error and costs more re-ranking, so the cost moves to latency rather than systematic misses. A global-range SQ8 is the one quantizer with an obvious age-dependent failure mode: new vectors outside the trained range get clipped. No measurement of this was found, so a quick SQ8 range-overflow check on late-arriving data would be novel evidence.
- The strongest peer-reviewed quantizer-staleness evidence (DeDrift) says codecs are not the problem. The contrary claims are preprints with mostly figure-only results.

### Gaps
- No measurement of SQ8 range overflow or clipping rate for late-arriving or OOD vectors was found.
- OPQ rotation staleness was measured only in DeDrift (small effect). No study of learned quantizers (e.g. ScaNN anisotropic, learned rotations) under temporal drift was found beyond HAKES.
- CoDEQ's exact recall gaps are figure-only, and its datasets have no real timestamps.

## 4. PQR-Search (arXiv 2611.07579) and search-parameter staleness / auto-tuning

### Takeaway
PQR-Search (preprint, ICDE 2027 submission) gives the most quantitative evidence that parameters tuned at build time go stale under updates. For IVF-Flat, HNSW, FreshDiskANN and Quake, the share of queries below 0.99 recall roughly doubles over five 10% update steps, and per-query worst cases collapse (Quake worst query 0.73→0.42). This is parameter staleness (fixed nprobe/ef) measured in aggregate. There is no age breakdown. LAET, DARTH, Ada-ef, Quake APS and CTR predict or certify recall but assume calibration and test queries come from the same distribution. None measures calibration drift as the collection grows.

### Cited Findings
- Title "Per-Query Recall Guarantees for Approximate Nearest Neighbor Search in Dynamic Vector Databases". Authors Woong-Kee Loh (Gachon Univ.) and Suan Lee (Semyung Univ.). arXiv 2611.07579v2, 11 Nov 2026. Extended version of an ICDE 2027 submission; preprint — [arXiv abs](https://arxiv.org/abs/2611.07579); [HTML](https://arxiv.org/html/2611.07579)
- Fig. 1(a), static, average-recall target 0.99, k=100:
  - 5.0–19.2% of in-distribution queries (SIFT, GIST, arxiv) fall below 0.99 for IVF-Flat, HNSW and IVF-RaBitQ.
  - 51.4–99.6% fall below for real OOD queries (yandex, laion, hotpotqa).
  - Tuning on a query log gives 4.5–18.7%.
  - IVF-Flat on laion: average recall 0.37 at k=10, and 60% of queries below 0.5.

  [arXiv PDF](https://arxiv.org/pdf/2611.07579)
- Fig. 1(b), updates on yandex (k=100). The index is built on 50% of data and tuned to 0.99. Then five steps each insert 10% "arriving in cluster order" and delete the oldest 10%, with no retuning. Fraction of queries below 0.99: IVF-Flat 13.5%→22.3%, FreshDiskANN 13.8%→20.0%, hnswlib 11.1%→16.7%, Quake 7.9%→17.5% — [arXiv PDF](https://arxiv.org/pdf/2611.07579)
- Fig. 1(c): CTR left up to 16.4% of queries below τ across 27 settings — [arXiv HTML](https://arxiv.org/html/2611.07579)
- §6.4: 11 workloads, each starting at 50% and adding 5×10%, under three patterns (random insertion, cluster-ordered insertion = "distribution shift", and a cluster-ordered sliding window) on arxiv, SIFT, laion, GIST and yandex.
  - IVF-Flat on the SIFT sliding window: 6.1%→14.6% (k=10) and 15.8%→31.6% (k=100).
  - IVF-Flat on GIST: 9.6%→21.3% (k=100).
  - IVF-Flat on arxiv: "queries with recall 0 appeared".
  - "Graph indexes degraded gradually even under random insertion alone": on SIFT k=100, HNSW 21.0%→26.9% and FreshDiskANN 17.5%→24.0%.
  - Quake on yandex: worst-query recall 0.73→0.42.

  [arXiv PDF](https://arxiv.org/pdf/2611.07579)
- NeurIPS'23 streaming runbook (msturing-10M-clustered delete_runbook: 32 inserts, 10 deletes, 33 searches, live set 260K–5.07M). IVF-Flat with nprobe fixed at 2612/4096 so that 99% of queries were exact at the start missed exactness on 1.0% of queries at the first search and 1.6–4.2% at later steps, "exceeding its target by up to 4.2×" — [arXiv PDF](https://arxiv.org/pdf/2611.07579)
- The PQR "arxiv" dataset uses held-out documents as queries (from an embedding-benchmark suite, ref [1]). The paper does not describe it as temporally ordered. Update streams are cluster-ordered, not time-ordered. No breakdown by item age or new-vs-old neighbors appears anywhere in the text — [arXiv PDF](https://arxiv.org/pdf/2611.07579)
- LAET: "Improving Approximate Nearest Neighbor Search through Learned Adaptive Early Termination" (Li, Zhang, Andersen, He; SIGMOD 2020, pp. 2539–2554, PR). Per-query GBDT termination for IVF/HNSW/IMI in Faiss — [paper PDF](https://conglongli.github.io/paper/ann-sigmod2020.pdf); [dblp](https://dblp1.uni-trier.de/rec/conf/sigmod/LiZAH20.html)
- DARTH: "Declarative Recall Through Early Termination for Approximate Nearest Neighbor Search" (Chatzakis, Papakonstantinou, Palpanas; SIGMOD 2026, PR; arXiv 2505.19001). Reports speedups of up to 14.6× on HNSW and up to 41.8× on IVF. DARTH+ (HAL 2026) adds quality guarantees — [arXiv](https://arxiv.org/html/2505.19001v2); [DARTH+](https://helios2.mi.parisdescartes.fr/~themisp/publications/hal26-darthplus.pdf)
- CTR, Certify-then-Rectify (Li, Mittal, Rana, Shetiya, Das, Koudas; arXiv 2607.02338, Jul 2026; preprint). Applies Conformal Risk Control plus Learn-then-Test to HNSW. Its guarantee assumes "independent queries from the same workload" — [arXiv 2607.02338](https://arxiv.org/pdf/2607.02338)
- PQR-Search's own related-work classification: LAET, DARTH, Ada-ef and Quake APS "only predict recall and do not guarantee it" — [arXiv PDF](https://arxiv.org/pdf/2611.07579)
- Quake reports static nprobe causing LIRE to miss its recall target on the real-temporal Wikipedia-12M workload — [arXiv 2506.03437v2](https://arxiv.org/html/2506.03437v2)

### Inferences
- PQR's Fig. 1(b) and §6.4 are the best available numbers for "parameters fitted to past data go stale as the collection changes", and they apply to every index family tested. The workload mixes cluster-ordered distribution shift and deletion, though. Under "random insertion alone" the text names only graph indexes as degrading. That suggests IVF-Flat parameter staleness is much weaker without shift, which matches a no-bias IVF result on organic growth. The IVF-Flat random-insertion numbers themselves are not in the text.
- Conformal and learned termination methods (LAET, DARTH, CTR) all calibrate under exchangeability. A growing, drifting collection breaks that assumption by construction, but no paper was found that measures their calibration error versus collection age or size.

### Gaps
- IVF-Flat's per-step numbers under random insertion only (no shift) in PQR §6.4 were not found in the text; they may be in an appendix table.
- No study was found that measures LAET/DARTH/Ada-ef/APS calibration drift as a collection grows over real time.

## 5. Datasets with real timestamps usable for temporal ANN evaluation

### Takeaway
Only two ANN-specific workloads with real time order were found: DeDrift's YFCC (Flickr 2007–2013, public source) and VideoAds (proprietary), and Quake's Wikipedia-12M (monthly page additions 2013–2021 with page-view query weights). Mainstream ANN benchmarks use synthetic cluster-ordered streams or none: VIBE, Big-ANN NeurIPS'23 streaming (MSTuring "clustered" runbooks), SPFresh, Ada-IVF, CoDEQ and PQR-Search.

### Cited Findings
- YFCC (DeDrift): 84M images, Jan 2007–Dec 2013, upload timestamps, DINO 384-d. Shows a monthly and seasonal drift signal — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752)
- VideoAds (DeDrift): 86M, Oct 2020–Oct 2022, internal ad videos — [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.02752)
- Wikipedia-12M (Quake): monthly page additions and page views Apr 2013–Dec 2021, 1.6M→12M over 103 updates, DistMult embeddings — [arXiv 2506.03437v2](https://arxiv.org/html/2506.03437v2)
- Big-ANN NeurIPS'23 streaming track (results paper arXiv 2409.17424; NeurIPS 2025 Datasets & Benchmarks, PR): MS Turing 30M with predefined insert/delete/search runbooks and an 8 GB, 1 h limit, ranked by average recall across checkpoints. The "clustered" runbooks are synthetic, and expiration-time runbooks assign simulated lifespans — [arXiv 2409.17424](https://arxiv.org/html/2409.17424v1); [NeurIPS proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/63092d79154adebd7305dfd498cbff70-Abstract-Datasets_and_Benchmarks_Track.html); [Azure Cosmos DB blog](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/)
- VIBE (arXiv 2505.17810) explicitly excludes streaming settings — [arXiv 2505.17810](https://arxiv.org/pdf/2505.17810)
- Slipstream (arXiv 2606.02992, preprint) compares original stream order with shuffled order for graph construction. This is evidence that real arrival order has locality — [arXiv 2606.02992](https://arxiv.org/pdf/2606.02992)
- HAKES' drift workload is synthetic (held-out ImageNet categories) — [PVLDB PDF](https://www.vldb.org/pvldb/vol18/p3049-ooi.pdf)

### Inferences
- A growing scientific-document collection with real arrival dates and relevance judgments is rarer than anything used in this literature. Every system paper above uses either synthetic cluster order or a dataset without per-item relevance judgments. A per-arrival-time retrievability analysis would be a genuine gap-filler for IVF as well as graphs.

### Gaps
- Not verified in this session: arXiv metadata with submission dates (e.g. the Kaggle arXiv snapshot), MS MARCO time splits, LongEval (CLEF) web snapshots, Wikipedia revision dumps, news corpora or Amazon Reviews timestamps as ANN benchmarks. None was found used in an IVF/quantizer staleness study. No pre-built "temporal ANN benchmark" with real ingestion timestamps was found (searched explicitly).
- Quake's Wikipedia-12M availability (code or data release) was not confirmed.
