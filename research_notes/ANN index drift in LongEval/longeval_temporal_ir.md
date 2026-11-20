# CLEF LongEval (2023–2026) and temporal IR evaluation: is the ANN index ever treated as a factor in effectiveness drift?

Research date: 2026-11-12. About 18 tool calls. Full-text PDFs were read locally (pdftotext + grep) for: Keller et al. 2023 (arXiv 2308.10549), Keller et al. ICTIR 2024 (arXiv 2407.01373), the LongEval 2024 extended overview (CEUR Vol-3740 paper-213), Galápagos Tortoise 2024 (Vol-3740 paper-219), DS@GT 2025 (Vol-4038 paper_274), AGH IR 2025 (Vol-4038 paper_276), and FADERIC 2023 (Vol-3497 paper-188). Everything else comes from abstracts, HTML pages, or search snippets, and is marked that way.

**Warning about one tool result:** a WebFetch summary of the 2024 overview said that participants used "FAISS" and "HNSW". When I grepped the full text of that PDF, neither word appears. I treat that summary as a hallucination and did not use it.

---

## Q1. Which LongEval participant papers used dense retrieval, and did any use ANN indexes or treat the index as a factor in temporal drift?

### Takeaway
Dense retrieval in LongEval mostly means exact or brute-force search (FAISS flat, or cosine over precomputed embeddings), BM25 followed by neural reranking, or ColBERT. In the papers I read, no participant reported ANN parameters (HNSW M/ef, IVF nprobe, PQ settings) or recall against exact search. None compared index rebuild with incremental updates, and none named the index as a possible source of temporal drift. The index is either exact or not mentioned.

### Cited Findings (evidence that X was done)
- **Keller, Breuer, Schaer (TH Köln), CLEF 2023 working notes, "Evaluating Temporal Persistence Using Replicability Measures"** ([arXiv 2308.10549](https://arxiv.org/abs/2308.10549), CEUR Vol-3497):
  - They submitted five systems: RRF, ColBERT, monoT5, Doc2Query, and E5.
  - For E5 the paper says: "The models E5small and E5base are used in a zero-shot fashion… A Faiss flat index was created from all embeddings, and L2 was used to score the query document similarity." This is explicitly exact (brute-force) search.
  - ColBERT was used through PyTerrier's `pyterrier_colbert`, both "as a first-stage retriever, where the whole corpus is converted to embeddings" and as a reranker of the BM25 top 1000. The paper does not describe the ANN stage of first-stage ColBERT. [Source](https://arxiv.org/pdf/2308.10549)
  - E5 first-stage runs were "not significantly different from the baseline", and ColBERT first-stage "impairs the performance further". [Source](https://arxiv.org/pdf/2308.10549)
- **Keller, Breuer, Schaer, ICTIR 2024, "Evaluation of Temporal Change in IR Test Collections"** ([arXiv 2407.01373](https://arxiv.org/abs/2407.01373)): the systems are BM25 (pivot), BM25+monoT5, BM25+ColBERT reranking, d2q, and RRF. Every one uses BM25 as first stage, so no ANN first stage is involved. [Source](https://arxiv.org/pdf/2407.01373)
- **Keller, Breuer, Schaer, CLEF 2024 conference paper, "Replicability Measures for Longitudinal IR Evaluation"** ([arXiv 2409.05417](https://arxiv.org/html/2409.05417); [Springer LNCS](https://link.springer.com/chapter/10.1007/978-3-031-71736-9_16)): the systems are BM25, colBERT, monoT5, RRF, d2q, and E5_base. The paper does not discuss the index or approximate search. This comes from the HTML page, and the index type for E5 was not confirmed in the text I extracted. [Source](https://arxiv.org/html/2409.05417)
- **Team Galápagos Tortoise (Jena), LongEval 2024, "Neural Re-Ranking and Rank Fusion for Temporal Stability"** (Gründel, Weber, Franke, Merker; [CEUR Vol-3740 paper-219](https://ceur-ws.org/Vol-3740/paper-219.pdf)):
  - Lexical first stage (PyTerrier PL2/BM25), then a sparse cross-encoder, ColBERT, and RankZephyr reranking, combined with ranx.fuse.
  - The full text contains no mention of FAISS, HNSW, or approximate search.
- **LongEval 2024 extended overview** ([CEUR Vol-3740 paper-213](https://ceur-ws.org/Vol-3740/paper-213.pdf)):
  - "most of the solutions… apply a multi-stage retrieval approach. Often, the first stage involves a lexical-based retrieval (e.g., BM25)… Reranking is done either using neural-based methods or sentence based transformers."
  - Runs are tagged only as "neural yes/no".
  - Grepping the full text for faiss, hnsw, approximate, nearest, and dense returned no hits.
- **DS@GT 2025, "DS@GT at LongEval: Evaluating Temporal Performance in Web Search Systems and Topics with Two-Stage Retrieval"** (Miyaguchi, Afrulbasha, Pramov; [CEUR Vol-4038 paper_274](https://ceur-ws.org/Vol-4038/paper_274.pdf); [arXiv 2507.08360](https://arxiv.org/pdf/2507.08360)):
  - They submitted four sparse systems: BM25, BM25+Gemini query expansion, BM25+CamemBERT cross-encoder rerank, and expansion plus rerank. Pyserini was used for keyword search only. Preprocessing ran on Georgia Tech PACE.
  - Dense retrieval appears only as future work: "Future work would involve incorporating dense retrieval into the pipeline… each date split takes about a day of Nvidia v100 GPU-time". They measured 21 GPU-hours to embed the 2022-08 split with nomic-embed-text-v2-moe.
  - They also suggest that "graph theoretic analysis techniques also apply to the semantic k-NN graph of a dense embedding model… in the context of the temporal evolution of systems".
  - Their temporal analysis is topic modelling (LDA/NMF). They attribute an anomaly before October 2022 to data or query factors, possibly non-French documents or event-specific queries.
- **AGH IR 2025, "Improving Scientific IR with Dense Representations and Cross-Encoder Re-ranking"** (Stryszewski, Prosowicz, Kawiak, Jaśkowiec; [CEUR Vol-4038 paper_276](https://ceur-ws.org/Vol-4038/paper_276.pdf)), SciRetrieval task:
  - Models: E5-base-v2 / MiniLM-L6-v2 bi-encoder plus ms-marco-MiniLM-L-12-v2 cross-encoder. Best reported nDCG@10 = 0.7448.
  - The background section describes ANN, HNSW, and Faiss in general terms.
  - The described method is "Embeddings are pre-computed and stored… Cosine similarity is used to identify top-k candidates". No ANN index or parameters are reported.
  - Some development experiments filtered the corpus down to about 4,262 qrel documents (or qrels plus 50K–200K random documents) because encoding the full corpus with E5-large would take about 70 hours on an RTX 2070.
  - The paper has no temporal or index-drift analysis.
- **LongEval 2025 working notes** (CEUR Vol-4038, LongEval section): 13 papers.
  - Overview and extended abstract: [paper_264](https://ceur-ws.org/Vol-4038/paper_264.pdf).
  - Participants: DataHunter (Boolean + CamemBERT), OpenWebSearch, RAND ("semantic… components"), BASETTE, CIR ("temporal sensitivity"), 3DS2A, SARD ("hybrid queries"), RISE, RACOON, DS@GT, CIR_cluster, AGH IR.
  - [Volume index](https://ceur-ws.org/Vol-4038/)
- **LongEval 2025 overview** ([arXiv 2509.17469](https://arxiv.org/html/2509.17469v1)), per its HTML: participant neural methods included E5-large-v2, MiniLM, SBERT, monoT5, CamemBERT, and SPLADE. A WebFetch summary reported no mention of FAISS, HNSW, ANN, index rebuild, or incremental indexing. I did not confirm this with a full-text grep.
- **LongEval 2026 working notes** ([CEUR Vol-4283](https://ceur-ws.org/Vol-4283/), CLEF 2026, Jena, 21–24 Sept 2026): 12 papers.
  - Extended overview (paper 374).
  - OpenWebSearch (LLM relevance judgments as relevance feedback).
  - CIR (sci retrieval, topic extraction, user simulation).
  - CIR (topic creation).
  - Multi-agent similarity-guided judgment (Cheng, Qi, Yi).
  - "Temporal Stability in Scientific IR: A Three-System Analysis" (Kumaran D et al.).
  - Qrels-grounded agentic LLM query expansion.
  - DS@GT ARC (citation integrity in scientific QA).
  - Hybrid evidence-aware user simulation.
  - VerbaNexAI (local temporal evidence selection for RAG).
  - Wu & Yang, "Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci".
  - Yang & Wu, candidate-constrained RAG.
  - [Source](https://ceur-ws.org/Vol-4283/)

### Searched and found nothing
- None of the full texts I read (2308.10549, 2407.01373, Vol-3740 paper-213, paper-219, Vol-4038 paper_274, paper_276) describes an HNSW or IVF/PQ index with parameters, ANN recall against exact search, index rebuild versus incremental update across snapshots, or the index as a cause of temporal drift.
- A search for "LongEval FAISS HNSW dense retrieval" turned up only the AGH background text.

### Inferences
- In LongEval practice, the index is either exact (FAISS flat, cosine) or hidden inside a toolkit. Possible hidden cases are ColBERT first stage through pyterrier_colbert, which in its standard end-to-end mode uses a FAISS IVF-PQ candidate-generation stage (see Macdonald & Tonellotto, Q5), and possibly "semantic"/"hybrid" components in 2025 Padua team papers.
- So the proposal's premise, that the ANN index is treated as exact or ignored, matches every paper inspected.
- The ColBERT case matters: Keller et al. 2023 report that first-stage ColBERT has worse temporal persistence. If that run used IVF-PQ candidate generation, part of its drift might come from the index. Nobody has separated that out. This is a hypothesis, not a finding.
- Heavy embedding costs (about 1 GPU-day per snapshot for DS@GT, 70 h for AGH) push participants to sample corpora or skip dense first stage. This also makes "re-encode and rebuild per snapshot" the implicit default whenever dense retrieval is used.

### Gaps
- I did not read the full text of the 2025 RAND, SARD, DataHunter, CIR, and CIR_cluster papers, or of any 2026 paper. In particular, "Temporal Stability in Scientific IR: A Three-System Analysis" (Vol-4283 paper 379) and Wu & Yang's "Diagnostic Analysis of Full-Text Temporal Retrieval" (paper 384) are the most likely places to find dense or ANN components and should be checked.
- I did not read the 2023 participant list beyond TH Köln and FADERIC. FADERIC is Lucene/BM25 only; its "approximate search" means fuzzy term matching.
- I did not verify what index the E5_base runs in 2409.05417 used. The 2023 paper used FAISS flat.

---

## Q2. LongEval overview papers 2023–2026: tasks, collections, snapshots, qrels, and stated causes of performance drop

### Takeaway
LongEval has moved from French web search with English translations (Qwant, 2023–2024), to French-only Qwant monthly snapshots plus CORE scientific search (2025), to CORE-only with four tasks (2026: SciRetrieval, TopEx, USim, RAG). Qrels come from click models (Cascade, later simplified DBN), with some manual assessments. The organisers attribute effectiveness changes to collection evolution (falling document overlap) and query or topic change. The system's index is never named as a cause.

### Cited Findings
- **2023 (1st edition).**
  - Qwant data with English automatic translations.
  - Three sub-collections: WT (within time, June 2022), ST (short-term, July 2022), and LT (long-term, September 2022, a two-month gap from ST).
  - More than 1.5M documents, of which 1,011,613 are core documents present in all three. Document content changes across snapshots.
  - Only 124 query strings appear in all collections.
  - Qrels are simulated with the Cascade Click Model on a 3-grade scale, with a mean of 14 qrels per topic. Highly relevant labels are rare (about 1 per topic).
  - Many more documents were removed between ST and LT than between WT and ST.
  - [Keller et al. 2023](https://arxiv.org/pdf/2308.10549)
- **2024 (2nd edition).**
  - Training set: January 2023, 599 queries, 9,785 qrels, 2,049,729 pages.
  - Test sets: Lag6 (June 2023: 1,790,028 documents, 407 queries) and Lag8 (August; the paper says "August 2024", apparently a typo for 2023: 2,531,614 documents, 1,518 queries).
  - Qrels come from the Qwant click model. More than 25K manual assessments were collected with Doctag over 60 queries, but were not yet used for scoring.
  - Main measure: Relative nDCG Drop, RnD = (nDCG_Lag6 − nDCG_Lag8)/nDCG_Lag6.
  - Findings:
    - The most robust systems (low RnD) are not the best ones.
    - Run correlation across lags is high: 0.99 on nDCG.
    - The correlation between absolute rank and the change ranking is about 0.07 / −0.05.
    - Maximum nDCG is about 0.4 on Lag6 and about 0.3 on Lag8.
    - Original French data beat the translations.
  - Text-classification task: Relative Performance Drop (RPD).
  - [CEUR Vol-3740 paper-213](https://ceur-ws.org/Vol-3740/paper-213.pdf)
- **2025 (3rd edition).**
  - WebRetrieval: French-only Qwant data. 19M documents from June 2022 to February 2023 for training, plus 14M documents in 7 monthly snapshots from March to August 2023. 33M documents and 182,757 queries in total.
  - SciRetrieval: CORE. About 2M documents in the first snapshot (Nov–Dec 2024) and more than 1M in the second (January 2025). 984 queries.
  - Qrels for both tasks come from a simplified DBN (sDBN) click model.
  - 19 teams submitted 45 Web runs and 23 Sci runs.
  - Measures: nDCG@10 and nDCG@1000, RI, ΔRI (normalising against BM25 to "dampen" environment change), and ER.
  - Findings:
    - Web: stable over short spans (March to May 2023) and drops over longer spans (March to August). Pearson 0.954 / Kendall 0.930 short-term, 0.965 / 0.893 long-term.
    - Sci: weaker agreement (Pearson 0.708, Kendall 0.500), and the best system changed between snapshots.
    - The organisers point to "decreasing overlap suggests a shifting document distribution over time" (document-overlap matrix).
  - Sources: [arXiv 2509.17469](https://arxiv.org/html/2509.17469v1); [Springer LNCS](https://link.springer.com/chapter/10.1007/978-3-032-04354-2_20); [ECIR 2025 lab paper, arXiv 2503.08541](https://arxiv.org/pdf/2503.08541); [CEUR extended abstract](https://ceur-ws.org/Vol-4038/paper_264.pdf)
- **2026 (4th edition).**
  - Four tasks: (1) ad-hoc scientific retrieval, (2) topic extraction from query logs (TopEx), (3) user simulation of query (re)formulations (USim), and (4) RAG over the evolving collection.
  - Data: CORE snapshots, the `longeval-sci-2026` collection in ir_datasets (tags `clef-2026/sci`, `clef-2026/rag`).
  - 104 runs were evaluated on effectiveness and on "robustness measures that quantify changes in retrieval effectiveness over time".
  - Sources: [Springer overview LNCS](https://link.springer.com/chapter/10.1007/978-3-032-39150-6_32); [ECIR 2026 lab paper](https://link.springer.com/chapter/10.1007/978-3-032-21321-1_45); [clef-longeval.github.io](https://clef-longeval.github.io/tasks/); [LongEval-Sci at CLEF 2026, arXiv 2607.04088](https://arxiv.org/html/2607.04088v1); [ir-datasets-longeval](https://github.com/clef-longeval/ir-datasets-longeval/tree/main)

### Inferences
- The causes the organisers name are all about the evaluation environment: documents (overlap and content change), queries (low query overlap), and qrels (click-model artefacts). The system side is reduced to "model robustness". There is no slot for "index-induced change", so the proposal fills a real gap in how drift is attributed.
- The 2026 RAG task over an evolving collection is a natural home for an index-effect study, because retrieval in RAG pipelines is almost always ANN-based (an inference; not verified in 2026 participant papers).

### Gaps
- I did not obtain the full text of the 2026 extended overview (Vol-4283 paper 374; the guessed URL returned 404) or of arXiv 2607.04088. The exact 2026 snapshot dates, sizes, qrel type (LLM judgments?), and robustness measures are unverified.
- I did not read the 2023 overview itself (Springer LNCS / CEUR Vol-3497). The 2023 details above come from Keller et al. 2023.
- Use of LLM relevance judgments in 2026 is suggested only by the OpenWebSearch 2026 title ("LLM Relevance Judgments as Relevance Feedback"). I did not confirm that the organisers' qrels are LLM-based.

---

## Q3. Temporal persistence measures, and how an "index component" could be expressed in them

### Takeaway
LongEval uses RnD/RPD (relative drop), Result Delta (RΔ), RI/ΔRI, Effect Ratio (ER), and ARP. All of them compare one system's score across evaluation environments (EEs), or compare a system with a pivot such as BM25. An index component fits in directly if you treat exact search as the pivot, or as an extra "environment" axis.

### Cited Findings
- Definitions from [Keller et al., arXiv 2409.05417](https://arxiv.org/html/2409.05417) (CLEF 2024 LNCS):
  - Result Delta: RΔ = (M^EE(S) − M^EE′(S)) / M^EE(S). Closer to 0 means more persistent.
  - Relative Improvement: RI = (M^EE(S) − M^EE(P)) / M^EE(P), where P is the pivot (BM25).
  - ΔRI = RI − RI′. Ideal value is 0.
  - Effect Ratio: ER = mean per-topic improvement over the pivot in EE′ divided by the same in EE. ER = 1 means the effect fully replicates.
  - ARP is mean effectiveness across EEs.
  - Finding: "the most effective systems are not necessarily the ones with the most persistent performance", and effectiveness generally improved over the three 2023 sub-collections.
- The 2023 paper first cast longitudinal evaluation as a replicability task ("Evaluation Environment" in the ACM sense) and used ΔRI and ER from repro_eval. [Source](https://arxiv.org/pdf/2308.10549)
- The 2024 overview defines RnD as the relative nDCG drop from Lag6 to Lag8. [Source](https://ceur-ws.org/Vol-3740/paper-213.pdf)
- The 2025 overview uses RI, ΔRI, and ER, and says ΔRI dampens "changes introduced by the evolving search setting". [Source](https://arxiv.org/html/2509.17469v1)

### Inferences (proposed mapping; my own construction, not from literature)
- Let E_t be exact-search effectiveness at snapshot t, and A_t^m be approximate-index effectiveness under maintenance policy m (rebuild, incremental, or in-place).
  - Model/data/query effect: RΔ_exact = (E_0 − E_t)/E_0.
  - Index effect at time t: I_t^m = E_t − A_t^m. In relative terms: RI with exact as pivot, i.e. (A_t^m − E_t)/E_t.
  - The observed drop splits exactly: A_0 − A_t = (E_0 − E_t) + (I_t − I_0).
- In ΔRI or ER form, use exact search as the pivot P instead of BM25. Then ΔRI(approx vs exact) = RI_t − RI_0 isolates how the index's penalty changes over time. Per-topic ER with an exact pivot shows whether the index penalty "replicates" across snapshots.
- An ARP-style summary of I_t^m across snapshots, for each maintenance policy, gives a single "index persistence" number.

### Gaps
- No published work found that uses exact search as the pivot system in RI, ΔRI, or ER.

---

## Q4. Other work decomposing IR effectiveness change into components; temporal drift in dense retrievers

### Takeaway
The closest prior work is Keller, Breuer, Schaer (ICTIR 2024). They decompose change by which evaluation component changes (documents, topics, qrels, using a CRUD view), but they keep the system fixed and do not separate the system's internals (encoder versus index). I found no decomposition that isolates an index or search-error term.

### Cited Findings
- Keller, Breuer, Schaer, "Evaluation of Temporal Change in IR Test Collections", ICTIR 2024 (Washington DC, July 2024):
  - Changes are categorised as CRUD on documents, topics/queries, and qrels.
  - Finding: system effectiveness and system rankings are "strongly affected by the components that change and to what extent these components changed".
  - All systems use a BM25 first stage.
  - [arXiv 2407.01373](https://arxiv.org/abs/2407.01373); [OpenReview](https://openreview.net/forum?id=ZnLa0TkhRg)
- "Simplified Longitudinal Retrieval Experiments: A Case Study on Query Expansion and Document Boosting" (2025) presents an ir_datasets extension for dynamic test collections. [arXiv 2509.17440](https://arxiv.org/html/2509.17440). I saw only the title and snippet.
- In the 2024 overview, the organisers cite Lovón-Melgarejo et al., "Studying catastrophic forgetting in neural ranking models" (ECIR 2021), as showing that "transformers-based [systems] are not always very sensitive to corpus evolution". [Source](https://ceur-ws.org/Vol-3740/paper-213.pdf)
- A 2026 systematic multi-domain evaluation of document retrievers appeared in search results. [arXiv 2609.29455](https://arxiv.org/pdf/2609.29455). I saw the title only, and its relevance is unverified.

### Searched and found nothing
- I found no paper that decomposes longitudinal effectiveness change into encoder/model versus ANN-index components.
- I did not search StreamingQA, TemporalWiki, or time-aware dense retrieval specifically, because of the tool-call budget. These are model-side temporal-drift benchmarks and, from prior knowledge, do not isolate index error. Verify before citing.

### Gaps
- I did not search the "Validating synthetic usage data" line of work or Breuer et al.'s other evolving-collection work.

---

## Q5. "Search error" vs "model error" for dense retrieval (ANN vs exact)

### Takeaway
Some IR work measures how ANN approximation affects downstream effectiveness on static collections, and the ANN community is debating which measures matter. Separately, the vector-database community has documented recall degradation in HNSW and graph indexes under streaming inserts and deletes. Nobody has connected the two inside a longitudinal IR evaluation. That intersection is the proposal's novelty.

### Cited Findings
- **Macdonald & Tonellotto, "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval"** (CIKM 2021; [arXiv 2108.11480](https://arxiv.org/pdf/2108.11480)):
  - Studies how ColBERT's ANN (FAISS) candidate-selection stage affects final effectiveness.
  - A search snippet reports only moderate agreement (Spearman ρ = 0.57) between exact and approximate scoring ranks, and says approximate scores alone are too inaccurate for high MRR/nDCG/MAP.
  - This is the closest IR "search-error" study. It uses a static collection.
- **Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?"** (2024; [arXiv 2409.06464](https://arxiv.org/pdf/2409.06464)): compares HNSW and flat indexes in Lucene/Anserini on BEIR on both effectiveness and efficiency, on static collections. I saw only the search result. The detailed finding that "flat is fine for small corpora, HNSW costs a small nDCG loss" comes from my prior knowledge and needs verifying.
- **"Are We There Yet? A Decision Framework for Replacing Term Based Retrieval with Dense Retrieval Systems"** ([arXiv 2206.12993](https://arxiv.org/pdf/2206.12993)): reports dense retrieval in both full and HNSW-approximate (128 neighbours) settings, and finds HNSW DR Pareto-dominant on nDCG@10 versus latency. This is a static evaluation.
- **Dimitropoulos & Mamoulis, "ANN Search: Recall What Matters"** (arXiv 2606.04522, June 2026): argues Recall@k is the wrong target. In downstream tasks (label precision, semantic similarity, BERTScore, LLM-graded quality), quality "remain[s] highly stable even when Recall@k drops significantly". They propose 1/Ratio@k instead. Static datasets only, with no nDCG-on-qrels or temporal aspect. [Source](https://arxiv.org/abs/2606.04522)
  - This cuts against the proposal: index error may barely affect nDCG. The proposal should treat that as a possible null result.
- **Index degradation under updates** (systems literature, without an IR-effectiveness framing):
  - FreshDiskANN (Singh et al. 2021) shows that simple graph update rules degrade index quality over streams of inserts and deletes for HNSW and NSG, and proposes batched delete consolidation. [arXiv 2105.09613](https://arxiv.org/pdf/2105.09613)
  - "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation" (2024) documents unreachable points under HNSW updates. [arXiv 2407.07871](https://arxiv.org/pdf/2407.07871)
  - "In-Place Updates of a Graph Index for Streaming ANN Search". [ResearchGate](https://www.researchgate.net/publication/389168199_In-Place_Updates_of_a_Graph_Index_for_Streaming_Approximate_Nearest_Neighbor_Search)
  - LSM-VEC: dynamic disk-based vector search. [arXiv 2505.17152](https://arxiv.org/pdf/2505.17152)
  - These measure recall@k against exact search on synthetic or benchmark streams, never on IR qrels over real temporal snapshots.

### Searched and found nothing
- I found no IR paper using the NMT-style "search error vs model error" decomposition (exact minus approximate effectiveness) for dense retrieval over time.
- I found no paper that evaluates FreshDiskANN-style or HNSW maintenance policies on a longitudinal IR test collection with relevance judgments.

### Inferences
- The proposal's novelty claim holds up against LongEval participant and organiser work (2023–2026, as inspected) and against the ICTIR 2024 decomposition. To position itself, it should:
  - extend Keller et al.'s CRUD/component view with a system-internal component;
  - use exact search as the pivot in RI/ΔRI/ER;
  - bring the streaming-ANN degradation literature (FreshDiskANN and the in-place-update papers) into IR evaluation.
- Main risk: on LongEval's sparse click-based qrels (mean of about 14 qrels per topic in 2023, few highly relevant), small recall losses may not show up as significant nDCG change. That is consistent with 2606.04522. The study should report recall@k against exact search alongside nDCG, and use per-topic ER-style analysis.

### Gaps
- I did not verify the full text of 2108.11480 or 2409.06464. The specific numbers attributed to them come from snippets or prior knowledge.
- I have not checked Pyserini/Anserini's own HNSW regression docs for any temporal results. They are unlikely to have any.
