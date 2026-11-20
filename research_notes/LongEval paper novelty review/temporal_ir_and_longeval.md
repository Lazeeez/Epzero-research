# Temporal IR, CLEF LongEval 2023-2026, and the index-over-time question: novelty review

Scope: does any prior work study how the search index (ANN type, incremental maintenance, rebuild policy) affects relevance-judged effectiveness over time? Target paper: "Does the Index Matter Over Time? Approximate Search, Index History and Collection Growth in Longitudinal Dense Retrieval" (LongEval-Sci 2026 data; exact vs HNSW/IVF; incremental vs rebuilt vs shuffled rebuild; held-out test queries; pre-registered hash-frozen protocol).

Method note: about 18 tool calls. I read the full text of three CLEF 2026 CEUR papers (the extended overview, Kumaran et al., Wu and Yang) with pdftotext and grepped them for faiss|hnsw|ivf|approximate|nearest|exact search|re-index|rebuild|dense|embedding. For the 2025 overview I read only the arXiv HTML. For 2023 and 2024 I did not do a team-by-team full-text sweep (see Gaps).

## Q1. LongEval 2023-2026 overviews and working notes: did anyone use an ANN index, maintain it across snapshots, or compare it to exact search?

### Takeaway
No. No LongEval overview or working note I could check compares exact and approximate search, or studies whether a dense index is maintained or rebuilt across snapshots. Dense retrieval in LongEval is a black-box first stage or a reranking signal. In 2026 it is mostly the organisers' pre-built per-snapshot Qwen3 dense index. One 2026 working note states the gap outright: "we cannot assess the impact of online dense index updates". That sentence is the best motivating quote for the new paper.

### Cited Findings
- **2026 extended overview (CEUR Vol-4283, paper374, pp. 5144-5173; Bouaraba, Breuer, Cancellieri, ..., Keller, ..., Schaer, Schwab).** The corpus is cumulative: "the collection snapshots are strictly cumulative. Each successive snapshot is a superset of the preceding one ... This yielded corpus sizes of approximately 87K, 1.3M, and 1.7M documents for Snapshot 1, Snapshot 2, and Snapshot 3, respectively." — [CEUR Vol-4283 paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- 2026 overview, official dense baseline: "The second baseline also used the abstract collection which is embedded with Qwen3 4B [30] at the default temperature. It was implemented with PyTerrier-Dense and a vLLM backed [31]." The overview names no index type (flat, HNSW or IVF) and says nothing about exact vs approximate search or rebuild policy. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- 2026 overview ir_metadata table: "Dense Neural 3 17" (3 of 20 submissions had a dense component), "Lexical 20 0", "Sparse Neural 0 20". — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- 2026 overview, queries: "Snapshot 1 contains 149 queries, Snapshot 2 100, and Snapshot 3 136. 12 queries are present in all snapshots." Training is "documents from Snapshot 1, 100 queries"; "The test datasets are composed of documents from Snapshot 2 and Snapshot 3, and 381 queries ... No overlap between this train split, the test split, or the LongEval 2025 collection exists." — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- 2026 overview: the only "approximate nearest neighbor" mention is in the Task 4 (RAG) data construction, not in retrieval evaluation: "paper abstracts were encoded using the SPECTER25 classification model ... These embeddings were then used to identify semantically similar papers through approximate nearest neighbor (ANN) search." — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- 2026 participants with dense components, per the overview: Lexora ("a hybrid dense-sparse pipeline combining BM25 with RM3, and Qwen3 embeddings via Reciprocal Rank Fusion"), dense-fusion (an LTR reranker using "embedding representations of the abstracts"), and the air-benders and nlp-hashiras "bm25-dense-rerank" runs. None of these studies the index. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- **Kumaran D, Janani Vr, Beulah A, Priyadharsini R, "Temporal Stability in Scientific Information Retrieval: A Three-System Analysis for CLEF LongEval 2026" (CEUR Vol-4283 paper379, pp. 5217-5230).** This is the closest LongEval paper, and it marks the gap:
  - "Dense indexes for each snapshot are loaded from HuggingFace Hub under the identifiers jueri/longeval-2026-snapshot-{1,2,3}-index-dense. These indexes were constructed externally and are used as provided; our contribution is the retrieval and fusion strategy rather than dense index construction."
  - "Dense retrieval interfaces are provided by the pyterrier_dr extension."
  - "the pre-built dense indexes are static: they capture document representations computed at a fixed time and cannot adapt to newly added documents in subsequent snapshots."
  - Limitations: "the QwenTemporalIR dense indexes are externally provided and fixed; we cannot assess the impact of online dense index updates." — [CEUR Vol-4283 paper379](https://ceur-ws.org/Vol-4283/paper379.pdf)
  - Caveat: their "static" claim is confusing. The organisers released one index per snapshot, so each snapshot's corpus is fully indexed. They seem to mean that encodings and index structure are not updated online. Either way, they did not test it.
- **Wu and Yang, "Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci" (CEUR Vol-4283 paper384; arXiv 2607.04088).** This paper raises update policy as a question but handles it only with monitoring heuristics, not index experiments:
  - "when should a retrieval system be updated as the scientific collection continues to grow?"
  - "Streaming vector-index maintenance studies similar problems for dense search [19]. We treat update triggers as a separate system-maintenance layer rather than mixing them into the ranking-model discussion." Its ref [19] is Mohoney et al., "Incremental IVF Index Maintenance for Streaming Vector Search", arXiv 2411.00970.
  - "because full-text and dense indexes are expensive to construct, while run-level and reranking overlays ..." and "None of these steps requires building a new index."
  - It uses "staleness_rate" and "coverage_gap" ("how far the indexed anchor has drifted from the live collection size") as update triggers.
  - It finds that "reindexing during March-May mostly adds later distractors rather than newly judged-relevant answers".
  - The dense run is the official Qwen3-Embedding-4B baseline over title+abstract (nDCG@10 0.270 / 0.236 / 0.140 across the three snapshots, per its table). — [arXiv 2607.04088 HTML](https://arxiv.org/html/2607.04088v1); [CEUR paper384](https://ceur-ws.org/Vol-4283/paper384.pdf)
  - This is the most likely "already known" objection on the update-policy side. The answer: it measures corpus-growth monitoring signals, never index type or index history, and its "reindexing" means corpus inclusion, not ANN structure.
- **2026 LongEval papers in CEUR Vol-4283 (paper374-385):** the overview, OpenWebSearch, two CIR papers, Cheng et al., Kumaran et al., Díaz Martínez et al., DS@GT ARC, Orejuela Cueter et al., VerbaNexAI, Wu and Yang (two papers). No title mentions dense, embedding, vector, index, ANN, FAISS or HNSW. — [CEUR Vol-4283 index](https://ceur-ws.org/Vol-4283/)
- **2025 overview (LongEval at CLEF 2025, LNCS; arXiv 2509.17469).**
  - Dense retrieval with cross-encoder reranking did best on Sci, and "No team submitted a retrieval model that was directly trained or fine-tuned on prior snapshots."
  - Table 2 counts "Dense Neural" in 59 runs.
  - The text never mentions FAISS, HNSW, ANN, vector index, per-snapshot index rebuild, or exact search.
  - Its temporal measures are RI, DRI and ER against a BM25 pivot, ARP, MARP, standardized nDCG@10, and Kendall's tau / Pearson across snapshots. — [arXiv 2509.17469](https://arxiv.org/html/2509.17469v1); [LNCS DOI](https://dl.acm.org/doi/10.1007/978-3-032-04354-2_20)
- 2025 teams with dense components, per the overview:
  - EAIiIB/AGH: "dense vector-based retrieval, and hybrid approaches"; MiniLM-L6-v2 and E5-large-v2.
  - RAND: "multi-qa-MiniLM-L6-cos-v1" for reranking.
  - RACOON: an SBERT cosine reranker.
  - 3DS2A: "reranking based on sentence embeddings".
  - DataHunter: SPLADE, which is sparse, plus CamemBERT. — [arXiv 2509.17469](https://arxiv.org/html/2509.17469v1)
  - The AGH working note: "initial retrieval using a dense bi-encoder model followed by reranking"; "dense + cross-encoder reranking outperforms all lexical and hybrid configurations" (search-snippet quotes; index type not checked). — [CEUR Vol-4038 paper_276](https://ceur-ws.org/Vol-4038/paper_276.pdf)
- 2025 DS@GT LongEval (web): a BM25 plus cross-encoder two-stage pipeline, not dense first stage (search-result summary; not read in full). — [Pith 2507.08360](https://pith.science/paper/2507.08360)
- 2024: a ListT5 listwise-reranking working note on LongEval 2024 (about 1.5M documents) mentions FAISS only in passing, per a snippet (unverified). — [arXiv 2407.06716](https://arxiv.org/pdf/2407.06716)

### Inferences
- In every LongEval edition I checked, the dense index is held fixed: rebuilt from scratch per snapshot by organisers or participants, and never varied. Its type (flat/exact vs ANN) is not reported. For 2026 the official index came through PyTerrier-Dense (pyterrier_dr), whose default flat index is, I believe, exact brute-force scoring (unverified; check the jueri/longeval-2026-snapshot-*-index-dense model card).
- So the paper's novelty within LongEval rests on three things nobody there has done: (a) a matched exact baseline, (b) index history (incremental vs rebuilt vs shuffled rebuild), and (c) separating index-induced effects from collection- and query-induced temporal change.
- Kumaran et al.'s limitation sentence and Wu and Yang's "streaming vector-index maintenance ... separate layer" framing should be quoted as the open problem the paper closes.

### Gaps
- I did not sweep the 2023 (CEUR Vol-3497) and 2024 (CEUR Vol-3740) LongEval team papers in full text for FAISS/HNSW. Search found no such paper, but a few teams (2023 dense/ColBERT runs, the 2024 ListT5 note) may have used FAISS HNSW/IVF as an unremarked implementation detail. None is known to compare against exact search or to vary maintenance. A reviewer is unlikely to raise these, but a quick grep of those volumes is advised.
- I did not read the 2025 CEUR extended overview (Vol-4038) in full, only the LNCS/arXiv version.
- I did not verify the type of the official 2026 dense index (flat vs HNSW/IVF).

## Q2. Other temporal / evolving-collection IR work: does anything decompose temporal effectiveness change, or include the index as a component?

### Takeaway
Keller, Breuer and Schaer (ICTIR 2024) is the reference decomposition. It breaks temporal change down by test-collection component (documents, queries, qrels) using CRUD operations. The retrieval system, including its index, is treated as constant, so the index is not one of the components. The new paper should present itself as adding a fourth, system-side component (index state and history) to that framework.

The streaming dense retrieval line (PLAID SHIRTTT, MURR, GradNormIR) studies staleness of the encoder or of the index's quantisation codebook. Of these, PLAID SHIRTTT is the work a reviewer is most likely to cite as "already known".

### Cited Findings
- **Keller, Breuer and Schaer, "Evaluation of Temporal Change in IR Test Collections", ICTIR 2024 (DOI 10.1145/3664190.3672530; arXiv 2407.01373).**
  - They classify "the change in the essential components according to the create, update, and delete operations of persistent storage known from CRUD."
  - They evaluate five systems on TripClick, TREC-COVID and LongEval.
  - Finding: "Both average retrieval performance of single systems and relative system performance are strongly affected by the components that change and the extent of those changes."
  - Code: github.com/irgroup/Temporal-Change. — [arXiv 2407.01373](https://arxiv.org/abs/2407.01373)
  - The components are the Cranfield components (documents, topics, qrels); the index is not one of them, per the abstract.
- The same group presented temporal replicability measures at CLEF 2024 (slides; not read in full). — [CLEF 2024 slides](https://ir.web.th-koeln.de/files/24-CLEF-temporal-replicability-slides.pdf)
- **Kuissi, Subrahmanyan, Thakur, Lin, "Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks" (arXiv 2603.04532, March 2026).**
  - FreshStack LangChain corpus snapshots, Oct 2024 vs Oct 2025.
  - System rankings stay stable: "correlation of up to 0.978 Kendall tau at Recall@50."
  - Its subject is benchmark reliability under corpus drift, not the index. — [arXiv 2603.04532](https://arxiv.org/abs/2603.04532)
- **Lawrie, Kayi, Yang, Mayfield, Oard, "PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval", SIGIR 2024 short (DOI 10.1145/3626772.3657964; arXiv 2405.00975). This is the closest "already known" work.**
  - Abstract: PLAID's "performance degrades in streaming settings where documents arrive over time."
  - Cause: "the vocabulary in the new documents moves away from that in the documents from which the cluster centroids were built."
  - NeuCLIR Chinese nDCG@20 by fraction of the stream used to build centroids: 0.047 (5%), 0.336 (75%), 0.384 (90%), 0.440 (100%). Also: "Even waiting until 75% of the stream has arrived still yields rather poor results."
  - "One solution is to periodically re-compute cluster centroids. Doing so results in new centroids, thus requiring all prior documents to be re-indexed."
  - SHIRTTT reaches 0.431 vs 0.448 nDCG′@20 against an "Oracle Shard Model" on ClueWeb09.
  - It attributes some Recall@100 gaps to "variations in the lossy vector compression". — [arXiv 2405.00975 HTML](https://arxiv.org/html/2405.00975)
  - Differences from the target paper:
    - It concerns ColBERT/PLAID late interaction, where centroids act as the residual-compression codebook as well as the candidate generator. It does not cover single-vector HNSW or IVF.
    - The baseline is an oracle-centroid PLAID, not exact search, so it cannot separate approximation (search) error from representation error.
    - There is no history or shuffle-order control.
    - The data is ClueWeb09 and NeuCLIR, not a longitudinal click-qrel benchmark like LongEval, and it reports no temporal-persistence measures (RI, ER).
- **MURR, "Model Updating with Regularized Replay for Searching a Document Stream" (HLTCOE/JHU; arXiv 2504.10250, 2025).**
  - It updates the encoder "to ensure the model can still faithfully search existing documents without reprocessing", on simulated streams.
  - It notes that systems "keep the same model over the lifespan of the index, or re-index the entire collection when a new model is trained."
  - Encoder-side, not ANN-side. — [arXiv 2504.10250](https://arxiv.org/abs/2504.10250)
- **GradNormIR, "When Should Dense Retrievers Be Updated in Evolving Corpora?" (Findings of ACL 2025; arXiv 2506.01877).**
  - "indexing newly emerging documents can degrade retrieval performance for future queries".
  - It detects out-of-distribution corpora to decide when to update the retriever. Encoder-side. — [ACL Anthology](https://aclanthology.org/2025.findings-acl.1334.pdf)
- Generative and incremental retrieval over dynamic corpora: IncDSI (arXiv 2307.10323), and a replication study (arXiv 2504.17519) in which "BM25 and DPR exhibit stable performance in maintaining retrieval effectiveness for initial documents" (snippet). These concern the model or DSI index, not ANN structure. — [IncDSI](https://arxiv.org/html/2307.10323v2); [Replication 2504.17519](https://arxiv.org/pdf/2504.17519)
- AURORA / "Neuro-Symbolic Continual Indexing for Evolving RAG" (Findings of ACL 2026): "recovers up to +26.9% Recall@10" under distribution shift (snippet only; baseline and setup unverified). — [ACL Anthology 2026.findings-acl.495](https://aclanthology.org/2026.findings-acl.495.pdf)
- ANCE (Xiong et al., ICLR 2021) refreshes its ANN index periodically during training because re-encoding is expensive. This is training-time index staleness, not evaluation over time. — [arXiv 2007.00808](https://arxiv.org/pdf/2007.00808)

### Inferences
- Taxonomy for related work (my synthesis):
  1. Test-collection-side temporal change: Keller et al. 2024, LongEval overviews, "Still Fresh?".
  2. Encoder-side staleness: MURR, GradNormIR, continual dense or generative retrieval.
  3. Index-codebook staleness for late interaction: PLAID SHIRTTT.
  4. ANN-structure maintenance measured only by neighbour recall: FreshDiskANN, Mohoney et al., BigANN streaming (see Q3).
  The target paper sits at the uncovered join: ANN-structure state and history, with a fixed encoder, measured by judged effectiveness and temporal-persistence measures, against an exact-search control.
- PLAID SHIRTTT must be cited and set apart explicitly. Suggested wording: "Lawrie et al. show that PLAID centroids learned early in a stream degrade nDCG; we ask the analogous question for single-vector ANN indexes (HNSW, IVF) with an exact-search control that isolates approximation error, and add index-history and insertion-order controls on a longitudinal benchmark."
- IVF is the direct analogue of PLAID's stale centroids: frozen coarse quantiser vs re-trained quantiser. A reviewer will expect a link to Lawrie et al. for the IVF arm. HNSW's insertion-order and history dependence has no counterpart in the IR literature I found.

### Gaps
- I did not read Chen et al., "Continual learning for generative retrieval", or Cai et al. on continual dense retrieval in full. Both are encoder-side by their titles and abstracts.
- I did not locate a paper titled exactly "Exploring the practicality of dense retrieval under document streams". The nearest matches are MURR and PLAID SHIRTTT from the same HLTCOE group.
- Alonso / Kanoulas temporal IR surveys and TemporalWiki / StreamingQA are about temporal query intent or knowledge, not index structure, from prior knowledge. Not re-verified here.

## Q3. Any study joining evolving corpora, ANN index quality, and relevance-judged effectiveness (2019-2026), including industry or vendor work?

### Takeaway
I found none. ANN-maintenance work (FreshDiskANN, IP-DiskANN, Mohoney et al.'s incremental IVF, BigANN streaming runbooks, vendor blogs) measures recall against exact k-NN ground truth over time, never relevance-judged effectiveness. Static IR work comparing HNSW with flat search measures judged effectiveness, but only on a fixed corpus. PLAID SHIRTTT comes closest to the join, but for late interaction and without an exact control.

### Cited Findings
- **FreshDiskANN (Singh et al., 2021; arXiv 2105.09613).**
  - Reports that under delete-and-reinsert cycles, recall "stabilizes at a lower recall value compared to the static index it starts out with, due to the use of approximate distances" (snippet).
  - Its metric is k-recall@k against the true k nearest neighbours. — [arXiv 2105.09613](https://arxiv.org/pdf/2105.09613)
- Soft deletes: "query quality degrades over time, and indices are typically rebuilt after, say, 10-20% of nodes are deleted" (IP-DiskANN, arXiv 2502.13826, snippet). — [arXiv 2502.13826](https://www.arxiv.org/pdf/2502.13826)
- **Mohoney et al., "Incremental IVF Index Maintenance for Streaming Vector Search" (arXiv 2411.00970, 2024).** Wu and Yang (LongEval 2026) cite it as the dense-side analogue, which shows the LongEval community knows the ANN-maintenance literature but has not tested it on judged effectiveness. Content not read in full; by its abstract it is a systems paper with neighbour-recall metrics. — [arXiv 2411.00970](http://arxiv.org/abs/2411.00970)
- **BigANN streaming track / Azure Cosmos DB DiskANN (industry).** Runbooks of inserts and deletes let you "plot the recall of vector index over time and thus evaluate the resilience of the index to streaming updates". Again, recall against ground-truth neighbours. — [Microsoft Cosmos DB blog, May 2025](https://devblogs.microsoft.com/cosmosdb/azure-cosmos-db-with-diskann-part-4-stable-vector-search-recall-with-streaming-data/); [TDS: Please use streaming workload to benchmark vector databases](https://towardsdatascience.com/please-use-streaming-workload-to-benchmark-vector-databases-bb0154480263)
- **Static exact vs HNSW on judged effectiveness.** Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?" (arXiv 2409.06464, 2024), and Anserini HNSW integration (arXiv 2304.12139): "compared to Faiss FlatIP (brute-force nearest neighbor search), we see that the HNSW implementation successfully retains retrieval effectiveness while improving retrieval efficiency" (snippet). Single static corpus (MS MARCO / BEIR). — [arXiv 2409.06464](https://arxiv.org/pdf/2409.06464); [arXiv 2304.12139](https://arxiv.org/pdf/2304.12139)
- **"ANN Search: Recall What Matters" (Dimitropoulos, arXiv 2606.04522, 2026).** "quality loss is bounded within a narrow interval even when Recall degrades by 60%" on RAG tasks. Static, single preprint. Useful support for the claim that neighbour recall is a poor proxy for judged effectiveness, which motivates measuring nDCG directly. — [arXiv 2606.04522](https://arxiv.org/html/2606.04522v1)
- I found no Pinecone or Weaviate vendor study measuring relevance-judged effectiveness of index freshness over time. Vendor material I found reports neighbour recall only.

### Inferences
- The defensible novelty claim: "To our knowledge, no prior study measures how ANN index type and maintenance history affect relevance-judged effectiveness and its temporal persistence on a longitudinal benchmark, against an exact-search control." Hedge with "to our knowledge".
- Reviewer risk: a systems reviewer may say incremental HNSW and IVF recall degradation is known (FreshDiskANN, Mohoney et al.). The answer is that neighbour recall does not translate linearly into judged effectiveness (Dimitropoulos 2026; Lin 2024 static), so the IR-level effect, and whether it changes LongEval temporal conclusions (RI, ER, system rankings), is untested.
- A null result ("the index does not matter over time at standard settings") is publishable in this framing. It licenses LongEval participants to ignore the index, which they currently do without evidence.

### Gaps
- I did not fully read the Mohoney et al. or IP-DiskANN evaluation sections. Some systems papers may include a downstream task metric. Worth checking before submission.
- I did not systematically search vendor blogs (Pinecone, Weaviate, Milvus, Qdrant).

## Q4. Search error vs model error (Stahlberg and Byrne 2019): has it been applied to dense retrieval indexes?

### Takeaway
Stahlberg and Byrne set up the canonical decomposition in NMT. Exact search exposes model errors, and search errors can hide them ("just the right amount of beam search errors"). I found no IR or dense-retrieval paper that applies this framing to ANN indexes with an exact-search control, let alone over time. The usual dense-IR treatment is a single "ANN drop vs exact" number on static data. That makes the framing an available, citable lens.

### Cited Findings
- **Stahlberg and Byrne, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?", EMNLP-IJCNLP 2019, pp. 3356-3362 (DOI 10.18653/v1/D19-1331).**
  - Using exact search on WMT15 En-De, beam search often fails to find the global best-scoring translation, even at beam 100.
  - "for more than 50% of the sentences, the model in fact assigns its global best score to the empty translation."
  - "search errors ... paradoxically, often prevent the decoder from suffering from a frequent but very serious model error."
  - "vanilla NMT in its current form requires just the right amount of beam search errors." — [ACL Anthology D19-1331](https://aclanthology.org/D19-1331/); [arXiv 1908.10090](https://arxiv.org/abs/1908.10090)
- Dense retrieval surveys frame retrieval as nearest-neighbour search and note that "most of previous studies adopt the Faiss library" without analysing approximation effects. — [Zhao et al. survey, arXiv 2211.14876](https://arxiv.org/pdf/2211.14876)
- A Faiss-based study reports "the performance drop of the approximation search with Faiss from the exact search is only 0.0098 on MRR" (search snippet; source paper not identified; static). This is the typical one-number treatment.
- A targeted search for "search error" plus dense retrieval plus a model-error decomposition found only ANN-theory error bounds (δ-EMG, arXiv 2511.16921) and distance-approximation error (FINGER). Neither is a search-error vs model-error decomposition of effectiveness. — [δ-EMG](https://arxiv.org/pdf/2511.16921)

### Inferences
- Positioning: define index-induced change as exact-vs-ANN deltas in nDCG and RI/ER (search error), and exact-search temporal change as model error plus collection change. The analogue of Stahlberg's paradox is worth testing: does approximation ever raise judged effectiveness, for example by missing near-duplicate distractors as the corpus grows? Wu and Yang's observation that reindexing "mostly adds later distractors rather than newly judged-relevant answers" makes this plausible.
- Combining this with Keller et al.'s CRUD decomposition gives the paper's conceptual contribution: temporal change in effectiveness = collection/query/qrels change (Keller) + index-state change (new), with exact search as the control that isolates the latter.

### Gaps
- I did not find any paper that explicitly borrows "search error / model error" for learned sparse or dense retrieval. This is an absence claim from limited searching (2 queries). A final Google Scholar check on "search errors" + "dense retrieval" + "exact" is advised before claiming firsthood for the framing.
