# Sweep re-run: ANN / vector indexes x evolving corpora x relevance effectiveness (web search, 2026-11-12)

Method: general web search (WebSearch, standard mode) plus WebFetch of arXiv / ACL Anthology / CEUR pages. 27 web searches and 16 fetches. The Semantic Scholar, arXiv and DBLP APIs were not used. Reviewer stance: try to find prior work that would make the proposal not novel. The proposal is: separate the ANN index's contribution from the model's in longitudinal (LongEval-style) effectiveness; compare rebuild vs incremental vs in-place maintenance by nDCG; use a nearest-neighbour-distance signal to detect query drift over time.

## Q1: Keyword query sweep (hits per query)

### Takeaway
None of the queries found a paper that measures how much an ANN index (HNSW/IVF/PQ/DiskANN) contributes to effectiveness on relevance labels across time-ordered snapshots, or that compares rebuild vs incremental vs in-place maintenance by nDCG. The work that comes closest splits into two groups. ANN-systems papers measure drift by recall against exact kNN, not by relevance labels. IR/continual-learning papers measure nDCG over time but treat the index as fixed, or avoid re-indexing for model reasons.

### Cited Findings
Per-query record. "Relevant" means at least partial overlap.

1. "approximate nearest neighbor" temporal drift retrieval effectiveness evolving corpus arxiv. Relevant hits:
   - DeDrift (Baranchuk, Douze et al., ICCV 2023, arXiv 2308.02752). Already known. It adapts IVF/PQ indexes to content drift and reports that it recovers 95-99% of recall at about 100x lower cost than a rebuild. **Verdict: partially overlaps.** It covers drift plus index maintenance, but quality is kNN recall against exact search, with no relevance labels and no index-vs-model split. — [arXiv](https://arxiv.org/pdf/2308.02752)
   - CANDY benchmark (arXiv 2406.19651). Continuous ANN with dynamic ingestion and distribution drift, testing HNSW, IVFPQ and others. **Partially overlaps.** It is drift-aware ANN benchmarking, but quality is kNN recall only. — [arXiv](https://arxiv.org/pdf/2406.19651)
   - "Approximate Nearest Neighbour Search on Dynamic Datasets: An Investigation" (arXiv 2404.19284). Evaluates ScaNN, IVFPQ, k-d tree, ANNOY and HNSW, including update cost. **Partially overlaps.** It is ANN-only, with no relevance labels. — [arXiv](https://www.arxiv.org/abs/2404.19284)
   - Drift-Adapter (arXiv 2509.23471). A transformation layer that avoids re-encoding and rebuilding the ANN index when the embedding model changes. **Different.** The drift it handles is a model-version change, not a change in the corpus or queries over time. — [arXiv](https://www.arxiv.org/pdf/2509.23471)
   - Query Drift Compensation (Goswami et al., CoLLAs 2026, PMLR 330:324-340; arXiv 2506.00037). Continual document retrieval over sequential BEIR tasks. It reports nDCG@10 and avoids re-indexing by compensating query embeddings. Reported FT+KD+QDC average nDCG@10 is 54.1%, matching joint training. **Partially overlaps.** It uses nDCG together with a "no re-index" maintenance choice. However, the "drift" is embedding-model drift across tasks, there is no ANN structure effect and no real time axis. — [PMLR](https://proceedings.mlr.press/v330/goswami26a.html), [arXiv](https://arxiv.org/html/2506.00037v2)
2. "index staleness" dense retrieval. The academic hits use "staleness" for the encoder/index mismatch during training (ANCE-style refresh). The rest are practitioner blogs (tianpan.co and similar) about RAG freshness. **Different / grey literature.** — [Survey 2211.14876](https://arxiv.org/pdf/2211.14876); [blog example](https://tianpan.co/blog/2026-04-20-rag-knowledge-base-freshness-index-rot)
3. "dynamic corpus" "dense retrieval" ANN index update effectiveness. Relevant hits:
   - **Embedding Surgery** (Amendola, Mallia, Perego; arXiv 2609.05110, 4 Sep 2026). Corrects rankings with convex-optimisation edits to document embeddings. Claims the edits "can be applied safely and efficiently to scalable ANN indexes via simple in-place overwriting, without requiring costly index reconstruction". The search snippet states that "no consistent degradation in retrieval effectiveness is observed" with in-place updates (TREC DL / MS MARCO). **Partially overlaps.** It is an effectiveness-measured in-place vs rebuild comparison on an ANN index. However, the updates are targeted embedding edits, not corpus evolution over time, and it has no longitudinal snapshots or drift detection. A reviewer could cite it against a narrow claim that "nobody has measured in-place ANN updates by nDCG". — [arXiv](https://arxiv.org/abs/2609.05110)
   - **MURR** (Yang, Tonellotto, Lawrie, MacAvaney, Mayfield, Oard, Miller; arXiv 2504.10250, Apr 2025). Model updating with regularized replay so that new models stay compatible with already-indexed vectors. The setup is four simulated document/query streams from LoTTE in five sessions, with a FAISS PQ index and Success@5. **Partially overlaps (closest IR-side hit).** It has a streaming corpus, an ANN index, effectiveness labels over sessions, and compares avoiding a re-index against re-indexing. It does not isolate the ANN structure's error from model error (the index is a fixed PQ setting), does not compare HNSW in-place vs rebuild, and has no NN-distance drift detector. — [arXiv](https://arxiv.org/html/2504.10250v1)
   - ANCE (2007.00808) and FreshDiskANN (2105.09613) came up again. Both are already known. — [ANCE](https://arxiv.org/pdf/2007.00808), [FreshDiskANN](https://arxiv.org/abs/2105.09613)
4. LongEval dense retrieval HNSW temporal persistence. LongEval overviews and Keller et al. replicability work came back. No hit mentions HNSW or ANN-structure effects. **Nothing relevant beyond known items.** — [LongEval 2025 overview](https://arxiv.org/abs/2503.08541v1); [Keller et al. 2407.06716](https://ar5iv.arxiv.org/html/2407.06716)
5. "search error" "dense retrieval" approximate exact evaluation conclusions. Relevant hits:
   - Macdonald and Tonellotto, "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval" (CIKM 2021, arXiv 2108.11480). Exact vs approximate (FAISS) scoring, with the reported drop in recall of relevant documents on TREC 2019 (0.77 to 0.63). **Partially overlaps** on separating index error from model error by relevance labels. It is static, with no time axis. — [arXiv](https://arxiv.org/pdf/2108.11480)
   - "Semantic Recall for Vector Search" (Kuffo et al., arXiv 2604.20417, Apr 2026). Proposes a recall metric restricted to relevant items, plus "tolerant recall", said to be "more robust for monitoring frequently-updated datasets". **Partially overlaps** on metric design. It is not longitudinal and does not compare maintenance strategies. — [arXiv](https://arxiv.org/html/2604.20417v1)
6. "incremental index" ColBERT streaming corpus. PLAID SHIRTTT (2405.00975) only, already known. It reports that PLAID degrades in streaming because centroids come from early tokens, and it fixes this with hierarchical sharding. **Partially overlaps (known).** — [arXiv](https://arxiv.org/abs/2405.00975)
7. ceur-ws Vol-4283 paper379 (lookup). Gave the volume identity: **CLEF 2026 Working Notes**, CEUR Vol-4283, 509 papers. — [CEUR Vol-4283](https://ceur-ws.org/Vol-4283/)
8. "re-indexing" dense retrieval temporal LongEval rebuild index. Relevant hits:
   - Yang and Wu, "Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci" (arXiv 2607.04088, Jul 2026). Uses BM25, RM3, a cross-encoder, RRF and a Qwen3 dense baseline. Includes "qualitative weekly IR-system update-monitoring analysis based on ingestion velocity and stale-coverage drift" and suggests that "simple collection-drift signals could indicate when an IR system should be refreshed". **Partially overlaps** on the drift-to-refresh idea. The signal is collection or ingestion based and qualitative, not NN-distance based, and there is no ANN-structure analysis. — [arXiv](https://arxiv.org/abs/2607.04088)
   - CLEF 2026 LongEval participant paper 379, "Temporal Stability in Scientific Information Retrieval: A Three-System Analysis for CLEF LongEval 2026". Uses BM25, RM3 and a *pre-built* Qwen3-Embedding-4B dense index fused with RRF. Temporal_Fusion reports mean stability 0.704 and mean nDCG@10 0.235 across snapshots. Snippets say a fixed pre-built dense index "cannot capture documents added after index construction time". **Different.** It uses a frozen index and does not study the ANN structure or maintenance. — [CEUR paper379](https://ceur-ws.org/Vol-4283/paper379.pdf)
   - Paper 384, "Candidate-Constrained RAG for LongEval-RAG". A search snippet mentions retrieval "from an HNSW index built over the same corpus" (attribution to a specific paper is unverified). Snapshot sizes are about 87K, 1.3M and 1.7M documents. **Different** (HNSW used only as infrastructure). — [CEUR paper384](https://ceur-ws.org/Vol-4283/paper384.pdf)
9. "vector database" drift relevance evaluation nDCG snapshots. Only practitioner blogs and vendor guides. They recommend tracking nDCG@k on a fixed query set and week-to-week "nearest-neighbor stability" (85-95% overlap is called stable). **Grey literature, not peer reviewed.** It does show that the monitoring idea is folklore among practitioners, so the novelty claim cannot rest on the idea alone. — [tianpan.co](https://tianpan.co/blog/2026-04-19-embedding-drift-long-lived-rag-systems); [dev.to](https://dev.to/dev-zc/detecting-embedding-drift-the-silent-killer-of-rag-accuracy-17md)
10. "index drift" retrieval nearest neighbor distance query drift detection. DeDrift plus blogs. **No academic hit** uses NN distance to detect query drift over time against relevance outcomes.
11. "streaming" "dense retrieval" evaluation index rebuild incremental. Relevant hits:
    - Streaming RAG (arXiv 2508.05662). Incremental prototype index upsert. **Different** (systems focus). — [arXiv](https://arxiv.org/pdf/2508.05662)
    - IncDSI (arXiv 2307.10323) and Hierarchical Corpus Encoder (arXiv 2502.18877). Incrementally updated retrieval with effectiveness, compared against DPR. **Partially overlaps** on incremental-update effectiveness. The index here is generative or learned, not an ANN graph. — [IncDSI](https://arxiv.org/pdf/2307.10323), [HCE](https://arxiv.org/pdf/2502.18877)
    - Ada-IVF (arXiv 2411.00970). Incremental IVF maintenance, measured by recall and throughput. **Partially overlaps (ANN-only).** — [arXiv](https://pdf.arxiv.org/abs/2411.00970)
    - Azure Cosmos DB DiskANN blog, "stable vector search recall with streaming data". Recall only. **Grey literature.** — [Microsoft devblog](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/)
12. "distribution shift" "ANN index" "retrieval quality". Relevant hits:
    - OOD-DiskANN (arXiv 2211.12850). Reports an order-of-magnitude latency gap for OOD queries. Recall only. **Partially overlaps** on query-shift effects on ANN, with no time axis. — [arXiv](https://arxiv.org/pdf/2211.12850)
    - Bao et al., "Alleviating Performance Degradation Caused by OOD Issues in Embedding-Based Retrieval" (Findings of EMNLP 2025). Query/base embedding mismatch degrades ANN, fixed with a distribution regularizer. **Partially overlaps.** It ties query-distribution shift to ANN effectiveness, but it is not temporal and has no NN-distance drift detector. — [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.340)
    - VIBE (arXiv 2505.17810). In-/out-of-distribution ANN benchmark. **Different (known).** — [arXiv](https://arxiv.org/pdf/2505.17810)
13. "Temporal Stability ... CLEF LongEval 2026" (title lookup). Confirmed paper 379 details (see 8).
14. LongEval 2026 dense FAISS HNSW Vol-4283. Returned CLEF 2026 papers 374, 375, 376, 379, 383, 384, 385 and 394. Snippets mention HNSW and FAISS-flat used as plain infrastructure. **No evidence any analyses index effects** (full texts were not readable, see Gaps).
15. "exact search" approximate system ranking evaluation conclusions dense reproducibility. Relevant hits:
    - "ANN Search: Recall What Matters" (Dimitropoulos and Mamoulis, arXiv 2606.04522, Jun 2026). Reports that downstream quality stays "highly stable even when Recall@k drops significantly" and proposes 1/Ratio@k. **Partially overlaps** on the point that ANN recall is a poor proxy for effectiveness. It is static. — [arXiv](https://arxiv.org/abs/2606.04522)
    - Hypencoder Revisited (SIGIR 2026 reproducibility, arXiv 2604.27037). Compares exhaustive vs efficient search. **Different.** — [arXiv](https://arxiv.org/abs/2604.27037)
    - PLAID reproducibility (arXiv 2404.14989) and Anserini dense (2304.12139). **Different (known).**
16. "longitudinal" "vector search" evaluation. Keller et al., "Simplified Longitudinal Retrieval Experiments" (arXiv 2509.17440, Sep 2025), an ir_datasets extension with no ANN. DS@GT at LongEval (2507.08360), sparse only. **Different.** — [2509.17440](https://arxiv.org/abs/2509.17440v1), [2507.08360](https://arxiv.org/abs/2507.08360)
17. "index maintenance" "retrieval effectiveness" dense vector. Ada-IVF plus vendor docs (Oracle IVF online rebuild, AlloyDB "maintain vector indexes"). Recall and QPS only. **Different / grey.** — [Oracle docs](https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/inverted-file-flat-vector-indexes-online-rebuild.html)
18. "temporal persistence" "dense retrieval" index. LongEval papers 379 and 384, MURR, TempRetriever (2502.21024), Re3 (2509.01306) and Generative Retrieval on Dynamic Corpora. Only MURR is relevant (see 3).
19. ReNeuIR 2025 workshop dense index update. Only workshop pages came back, with no individual paper on index maintenance over time. **Nothing relevant found.** — [ReNeuIR](https://reneuir.webis.de/)
20. thesis/dissertation dynamic vector search temporal dense. **No thesis found.** Returned LiveVectorLake (Prajapati, arXiv 2601.05270, Nov 2025): a versioned vector KB with 10-15% re-processing vs 100% for a full re-index, latency only, on a 100-document corpus with 5 versions and no relevance labels. **Different.** Also returned Kim et al. EMNLP 2024 (see 22), IP-DiskANN (2502.13826) and streaming quantization (2512.18335). — [LiveVectorLake](https://www.arxiv.org/abs/2601.05270)
21. SPFresh / FreshDiskANN in-place recall vs nDCG MS MARCO temporal. Returned IP-DiskANN (arXiv 2502.13826, "stable recall over lengthy update patterns") and SPFresh (2410.14452). Recall only. **No hit evaluates these with relevance labels.** — [IP-DiskANN](https://arxiv.org/abs/2502.13826v1)
22. "Exploring the Practicality of Generative Retrieval on Dynamic Corpora" (Kim, Yoon, Lee, Jang, Yang, Seo; EMNLP 2024, arXiv 2305.18952). Compares dual encoders vs generative retrieval on StreamingQA with "indexing-based updates" (index only) vs "training-based updates". **Partially overlaps.** It compares update strategies by effectiveness over a time-evolving corpus. However, the dual-encoder "index" update means adding embeddings, the ANN structure is not studied, and the contrast is model family. — [ACL Anthology](https://aclanthology.org/2024.emnlp-main.755.pdf)
23. NN distance query drift detection monitoring. Blogs and glossaries only (nearest-neighbour churn, re-embedding distance). **No academic hit.**
24. Xiao 2407.07871 cited by. Returned **GradNormIR**: Ko et al., "When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR" (Findings of ACL 2025, arXiv 2506.01877). It uses gradient norms to detect OOD *corpora* before indexing, on BEIR. **Partially overlaps** with the drift-detection component. It is the closest published "when to update" detector, but the signal is gradient based rather than NN distance, it detects corpus drift rather than query drift, and it is not tied to ANN structure. — [arXiv](https://arxiv.org/abs/2506.01877)
25. TempWeb 2024/2025 vector temporal. TempWeb 2025 was held at WWW 2025 in Sydney and included "Temporal Aspects of CLIP Embeddings". **Nothing on ANN maintenance or effectiveness.** — [TempWeb](https://temporalweb.net/)
26. ECIR 2026 reproducibility ANN HNSW. Returned "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" (ECIR 2026 Reproducibility II) and Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?" (ACL 2025 Industry; this is 2409.06464, already known). Both are static. **Different.** — [ECIR 2026 session](https://ecir2026.dryfta.com/programme/abstract-archive/program/detail/18/reproducibility-ii-retrieval); [ACL](https://preview.aclanthology.org/setup/2025.acl-industry.61)
27. continual dense streaming nDCG time slices re-encode. QDC again (see 1). Nothing new.

### Inferences
- The proposal's components each have precedents. Missing is any work that combines them:
  - Longitudinal effectiveness: LongEval, Keller et al.
  - ANN drift by recall: DeDrift, CANDY, Ada-IVF, SPFresh, IP-DiskANN.
  - Avoiding re-indexing judged by effectiveness: MURR, QDC, Embedding Surgery, Kim et al. 2024.
  - Separating ANN error from model error on static data: Macdonald and Tonellotto 2021, "Recall What Matters" 2026, Semantic Recall 2026.
  - Deciding when to update: GradNormIR, Yang and Wu 2026.
- The gap is still there: an index-vs-model decomposition across real time-ordered snapshots, with rebuild vs incremental vs in-place HNSW maintenance compared by nDCG, and with query-side NN-distance drift detection. The novelty case is only as strong as its claim of combining these. Each component alone has precedent.
- The strongest reviewer objections would cite MURR (streaming plus PQ index plus effectiveness), Embedding Surgery (in-place ANN overwrite judged by effectiveness), GradNormIR (when-to-update detection) and DeDrift (drift-aware index maintenance). The proposal should cite and contrast all four.
- The NN-distance monitoring idea already circulates in practitioner writing (blogs on "nearest-neighbour stability"). The novelty claim therefore has to rest on validating the signal against relevance labels over time, not on the idea itself.

### Gaps
- One assigned query, "evolving corpus" HNSW nDCG, was folded into query 1 and not run separately.
- The full texts of CLEF 2026 papers 379 and 384 could not be parsed: the PDFs came back as binary and no PDF tools are installed. It is unverified whether either analyses HNSW or ANN effects beyond using them. The snippets suggest they do not.

## Q2: Forward citations of anchor papers

### Takeaway
I could not run a direct forward-citation crawl. Google Scholar redirected (blocked), and the Semantic Scholar web page for 2405.00975 returned 404. Web search around the anchors turned up GradNormIR, MURR, IP-DiskANN and Embedding Surgery. None of these connects ANN dynamics with IR effectiveness over time in the way the proposal does.

### Cited Findings
- Xiao et al. 2024 (2407.07871) is "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation" (code ICPADS-MN-RU). It is an ANN systems paper about the unreachable-points phenomenon. — [arXiv](https://arxiv.org/html/2407.07871v1)
- MURR (2504.10250) comes from the same HLTCOE group as PLAID SHIRTTT and continues the streaming-dense-retrieval line. It probably cites SHIRTTT; this was not verified from its reference list. — [arXiv](https://arxiv.org/html/2504.10250v1)
- IP-DiskANN (2502.13826) succeeds FreshDiskANN and is evaluated by recall only. — [arXiv](https://arxiv.org/abs/2502.13826v1)

### Inferences
- No anchor-adjacent paper found measures ANN-structure maintenance with relevance labels over time. This is absence of evidence from an incomplete crawl, not evidence of absence.

### Gaps
- Blocked: Google Scholar "cited by" (redirect to sgsync, treated as a block) and the Semantic Scholar web page (404). Connected Papers was not attempted because it is rendered with JavaScript.
- Forward citations of FreshDiskANN, SPFresh, Lin 2024, Keller ICTIR 2024 (2407.01373) and DeDrift remain unenumerated. Recommended manual step: open the Google Scholar "Cited by" pages for DeDrift, PLAID SHIRTTT and Keller 2024 in a normal browser.

## Q3: Workshops and theses

### Takeaway
ReNeuIR, TempWeb, ECIR 2026 reproducibility and the CLEF 2026 LongEval working notes contained nothing that studies ANN index maintenance by effectiveness over time. I found no thesis or dissertation on the topic.

### Cited Findings
- ReNeuIR 2025 (4th edition, SIGIR 2025) focused on efficiency benchmarking and a shared task. Its pages listed no paper on index updates over time. — [ReNeuIR](https://reneuir.webis.de/)
- TempWeb 2025 was held at WWW 2025 (Sydney) and TempWeb 2024 at WWW 2024 (Singapore). Topics were web archives and temporal analytics, plus one talk on CLIP embedding temporality. — [TempWeb](https://temporalweb.net/)
- The ECIR 2026 reproducibility track includes the HNSW "hubs" paper, which is static. — [ECIR 2026](https://ecir2026.dryfta.com/programme/abstract-archive/program/detail/18/reproducibility-ii-retrieval)
- In the CLEF 2026 LongEval working notes (CEUR Vol-4283), participant systems use pre-built dense indexes, HNSW or FAISS-flat as infrastructure. — [CEUR Vol-4283](https://ceur-ws.org/Vol-4283/)
- SIGIR 2026 reproducibility: Hypencoder Revisited, with an exhaustive vs efficient search comparison, static. — [arXiv](https://arxiv.org/abs/2604.27037)

### Inferences
- LongEval participants have the ingredients (snapshots, dense indexes, nDCG) but treat the index as a black box. This supports the proposal's framing that the index is an unexamined variable in longitudinal IR evaluation.

### Gaps
- Not checked individually: ReNeuIR 2023, 2024 and 2026 paper lists, SIGIR 2025 and 2026 short-paper lists, and thesis repositories (ProQuest, EThOS, NDLTD). The thesis search used web search only and found none, which is weak evidence.
- Totals: 27 web searches and 16 fetches. Blocked or failed: 4 fetches (Google Scholar redirect, Semantic Scholar 404, two CEUR PDFs unreadable). Queries with nothing relevant beyond known items: 2 (staleness), 4 (LongEval HNSW), 9 (vector DB drift, grey only), 10 (index drift), 16, 17, 19 (ReNeuIR), 20 (theses), 23, 25 (TempWeb), 26, 27.
