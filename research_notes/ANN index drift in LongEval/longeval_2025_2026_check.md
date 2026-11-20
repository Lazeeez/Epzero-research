# LongEval 2025 and 2026 papers: does any study ANN indexing under temporal drift?

Method: on 2026-11-12 I downloaded the full-text PDFs of every LongEval paper in CEUR-WS Vol-4283 (CLEF 2026 Working Notes, papers 374–385) and every LongEval paper in Vol-4038 (CLEF 2025 Working Notes, paper_264–paper_276). I extracted the text with pdftotext and searched all of it for: HNSW, FAISS, PLAID, nearest neighbo(u)r, approximate, ANN, IVF, brute force, exact search, exhaustive, IndexFlat/FlexIndex, vector database/store/index, Qdrant/Milvus/Chroma/Weaviate/pgvector, kNN, ColBERT, dense retrieval, embedding, re-index, incremental. Then I read the passages around each hit. **Everything below comes from the full text unless it is marked [abstract-only] or [overview-only].** I also read the organisers' dense-baseline code on GitHub and the pyterrier-dr source it depends on.

## Q1. CLEF 2026 LongEval working notes (CEUR-WS Vol-4283): every paper, its pipeline, and whether ANN or indexing appears

### Takeaway
None of the 12 LongEval 2026 papers uses ANN search for dense retrieval, reports an ANN index configuration, compares approximate with exact search, or attributes effectiveness change to the vector index. The only dense first-stage retrieval in Task 1 used the organisers' pre-built Qwen3-Embedding-4B indexes, which are brute-force (exact) NumPy indexes built separately for each snapshot. The phrase "approximate nearest neighbor" appears only once in the 2026 corpus: the overview uses it while describing how the Task 4 RAG questions were built, not for retrieval. The closest prior work is two papers that raise index maintenance qualitatively. Paper 379 lists "online dense index updates" as an unassessed limitation and as future work. Paper 384 proposes a weekly policy for when to trigger an index refresh. It does not measure effectiveness under that policy and cites Incremental IVF (Mohoney et al. 2024) only as related work.

### Cited Findings
**Volume:** CLEF 2026 Working Notes are CEUR-WS Vol-4283. Its LongEval section is papers 374–385 — [CEUR Vol-4283](https://ceur-ws.org/Vol-4283/)

Paper-by-paper (full text searched):

- **374 – Extended Overview of the CLEF 2026 LongEval Lab** (Bouaraba, Breuer, Cancellieri, El-Ebshihy, Fröbe, Galuščáková, Goeuriot, et al.), pp. 5144–5173 — [PDF](https://ceur-ws.org/Vol-4283/paper374.pdf)
  - Task 1 had "20 submitted retrieval runs from eight teams". All 20 runs use a lexical ranker at some point; 3 runs use a dense neural component; 15 are multi-stage. All eight teams used PyTerrier. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
  - Baselines: BM25 over abstracts, and "the abstract collection which is embedded with Qwen3 4B … implemented with PyTerrier-Dense and a vLLM backed". The overview gives no index type and does not mention exact or approximate search. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
  - The only ANN mention: for the Task 4 RAG collection, "paper abstracts were encoded using the SPECTER2 … model … These embeddings were then used to identify semantically similar papers through approximate nearest neighbor (ANN) search". This is dataset construction, not a retrieval system. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
  - Task 3 mentions "k-nearest-neighbor retrieval" only in a query-generation component of VerbaNexAI's user simulator. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- **375 – Team OpenWebSearch at LongEval: LLM Relevance Judgments as Relevance Feedback for Scientific Search** (Alexander, Fröbe, Hendriksen, Hagen, Hiemstra, Potthast, de Vries), pp. 5174–5178 — [PDF](https://ceur-ws.org/Vol-4283/paper375.pdf)
  - Runs: keyquery reformulation with RM3 over BM25, Dirichlet and PL2, plus "put-to-front" over BM25. Lexical only. No ANN, dense or vector terms found.
- **376 – CIR at LongEval 2026: Ad-Hoc Scientific Retrieval, Topic Extraction From Query Logs, and User Simulation** (Bakirci, Brückner, Christian, et al.; student teams Time4Citation and DenseFusion), pp. 5179–5195 — [PDF](https://ceur-ws.org/Vol-4283/paper376.pdf)
  - Time4Citation: BM25 top-1000, re-ranked with citation and recency boosts.
  - DenseFusion: BM25 top-500, re-ranked with LightGBM LambdaRank. One feature is the cosine similarity of all-MiniLM-L6-v2 query and abstract embeddings. Dense vectors appear only as a re-ranking feature over BM25 candidates, so there is no vector index and no ANN. No ANN terms found.
- **377 – CIR at LongEval 2026: Manual Topic Creation and Concept Induction-Based Topic Generation** (Breuer, Keller, Bernard, Schaer), pp. 5196–5205 — [PDF](https://ceur-ws.org/Vol-4283/paper377.pdf)
  - Topic creation. Students used the ChatNoir longeval-sci-2026-snapshot-3 index. Qwen3-4B embeddings are used for topic generation. No retrieval-index study and no ANN terms.
- **378 – A Multi-Agent Approach with Similarity-Guided Judgment for Scientific Knowledge Retrieval and Answer Generation** (Cheng, Qi, Yi), pp. 5206–5216 — [PDF](https://ceur-ws.org/Vol-4283/paper378.pdf)
  - RAG task. The paper computes pairwise cosine similarity with all-MiniLM-L12-v2 over candidate documents. In Task 4 each question comes with its own 10 candidate documents, so no index is built. No ANN terms.
- **379 – Temporal Stability in Scientific Information Retrieval: A Three-System Analysis for CLEF LongEval 2026** (Kumaran D, Janani Vr, Beulah A, Priyadharsini R; team Lexora), pp. 5217–5230 — [PDF](https://ceur-ws.org/Vol-4283/paper379.pdf)
  - Three systems: RM3_Conservative; Temporal_Fusion (BM25 + RM3 + a cross-snapshot prior); QwenTemporalIR (BM25 + RM3 + Qwen3-Embedding-4B dense run fused with RRF, each stage TOP_K = 200).
  - Dense component: "Dense indexes for each snapshot are loaded from HuggingFace Hub under the identifiers jueri/longeval-2026-snapshot-{1,2,3}-index-dense. These indexes were constructed externally and are used as provided; our contribution is the retrieval and fusion strategy rather than dense index construction." Queries go through "the pre-built index's retriever interface" in pyterrier_dr. The paper names no index type (flat, HNSW or IVF) and makes no exact-vs-approximate comparison.
  - The paper attributes dense temporal behaviour to the index being static: "the pre-built dense indexes are static: they capture document representations computed at a fixed time and cannot adapt to newly added documents in subsequent snapshots". Its limitations section says "the QwenTemporalIR dense indexes are externally provided and fixed; we cannot assess the impact of online dense index updates". Its future work lists "Online dense index construction, where the dense index is rebuilt incrementally as new documents arrive".
  - Results: Temporal_Fusion has the best stability (Jaccard 0.704) and the best mean nDCG@10 (0.235). QwenTemporalIR has the best snapshot-3 effectiveness of any system (ARP 0.160, RC 0.395), but its stability (0.607) is below BM25's (0.620).
  - None of the search terms (HNSW, FAISS, ANN, approximate, IVF, brute, exact search) appears.
- **380 – Qrels-Grounded and Agentic LLM Query Expansion with Post-Processing Optimization for Longitudinal Scientific Topic Extraction** (Díaz Martínez, Serrano Castañeda, Puertas, Martínez-Santos; VerbaNexAI), pp. 5231–5239 — [PDF](https://ceur-ws.org/Vol-4283/paper380.pdf)
  - Task 2 topic extraction, using an in-memory PyTerrier BM25 index for pseudo-relevance feedback. Its one "exact search" hit refers to lexical matching, not vectors. No ANN.
- **381 – DS@GT ARC at LongEval: Citation Integrity and Factual Grounding in Scientific QA** (Michaels, Johnson), pp. 5240–5251 — [PDF](https://ceur-ws.org/Vol-4283/paper381.pdf)
  - RAG task. Hybrid BM25 + bge-base-en-v1.5 dense retrieval over chunks of the provided candidate documents, then a cross-encoder filter and Gemma-4-31B generation. Index and search type are not given. No ANN terms.
- **382 – Hybrid Evidence-Aware User Simulation for Scientific Search Sessions at LongEval 2026 Task 3** (Orejuela Cueter, Serrano Castañeda, Martinez-Santos, Puertas), pp. 5252–5261 — [PDF](https://ceur-ws.org/Vol-4283/paper382.pdf)
  - User simulation. A recency-weighted all-MiniLM-L6-v2 session vector is "used both for nearest-neighbor retrieval in the transition memory and for ranking candidate next queries". This is kNN over a query-transition memory, not document retrieval. No index configuration or exact-vs-approximate comparison.
- **383 – VerbaNexAI at LongEval 2026 Task 4: Fully Local Temporal and Pairwise Evidence Selection for Scientific RAG** (Sánchez Sánchez, Serrano Castañeda, Puertas, Martínez-Santos), pp. 5262–5268 — [PDF](https://ceur-ws.org/Vol-4283/paper383.pdf)
  - RAG task. Splits documents into 250-word windows and scores them with the bge-reranker-v2-m3 cross-encoder. No vector index and no ANN.
- **384 – Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci** (Haijian Wu, Yingdong Yang; team GLHF), pp. 5269–5283 — [PDF](https://ceur-ws.org/Vol-4283/paper384.pdf). Also on arXiv as [2607.04088](https://arxiv.org/abs/2607.04088), dated 5 July 2026, with the author order listed as Yang, Wu.
  - Pipeline: compares the official BM25 and Qwen3 dense baselines (title+abstract) with full-text BM25 plus additive, router, temporal and citation variants, RM3, a MiniLM cross-encoder, and RRF (for example RRF(FT BM25, Dense)). The dense run is the official baseline-qwen3-4b run. The authors stress that fusion "operates over existing run files and does not create a new index", and that "full-text and dense indexes are expensive to construct".
  - Section 2.5, "Dynamic Index Maintenance", cites incremental inverted indexing (Brown/Callan/Croft 1994; Asadi & Lin 2013; Moffat & Mackenzie 2023) and "Streaming vector-index maintenance … for dense search", citing Mohoney et al., *Incremental IVF Index Maintenance for Streaming Vector Search* (arXiv:2411.00970). They state: "We treat update triggers as a separate system-maintenance layer rather than mixing them into the ranking-model discussion."
  - Section 6 proposes a weekly update-trigger policy. It fires an incremental refresh when ingestion velocity exceeds 3× the baseline, or when staleness_rate ≥ 0.80 and coverage_gap ≥ 0.03. That yields 2/5/1 updates across the three snapshot windows. The authors call it "scheduling/monitoring analysis, not … a causal claim that more frequent updates improve current-qrel effectiveness". They also note "reindexing during March–May mostly adds later distractors rather than newly judged-relevant answers". The policy is not tied to a vector-index type and measures no recall or effectiveness effect of updating.
  - None of HNSW, FAISS, ANN, approximate, brute or exact-search appears. "IVF" appears only in the title of reference [19].
- **385 – Candidate-Constrained Retrieval-Augmented Generation for LongEval-RAG: System Design and Empirical Analysis** (Yang, Wu), pp. 5284–5298 — [PDF](https://ceur-ws.org/Vol-4283/paper385.pdf)
  - RAG over the provided candidates: rule-based chunking, query expansion, pseudo-relevance feedback, BM25 with RRF, and late MiniLM sentence reranking. No vector index and no ANN.

**Task 1 teams without a working-notes paper:** the overview's results tables list the runs b-bm25-dense-rerank and b-bm25-dense-rerank-v2 (team air-benders), bm25-dense-rerank (nlp hashiras) and ce-reranked (air-02). No CEUR paper exists for these teams, and the overview's team summaries do not describe them. Going by run names alone, they appear to be BM25 first-stage runs with dense re-ranking, which would not use an ANN first stage. [overview-only; inferred from run names] — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)

### Inferences
- In 2026, the only dense *first-stage* retrieval in Task 1 was the organisers' Qwen3 baseline and Lexora's reuse of it (paper 379). Both ran exact brute-force search (see Q3). This means ANN-induced effects were absent from the dense runs by construction, and nobody measured them.
- Papers 379 and 384 together are the closest prior art. 379 names "online dense index updates" as an open question. 384 frames index refresh as a maintenance policy, citing streaming IVF maintenance. Neither builds an ANN index, varies its parameters, compares it with exact search, or compares rebuilding with incremental updates for effectiveness. The novelty claim should cite both and set itself apart from them.
- Paper 379 claims the dense index "cannot adapt to newly added documents in subsequent snapshots". This contradicts the paper's own setup, which loads a separate index for each of snapshots 1, 2 and 3, and the organisers' code, which builds one index per snapshot. Their explanation of why dense retrieval underperformed is therefore probably wrong. That itself is evidence that index effects were not studied rigorously.

### Gaps
- I did not check the LNCS (Springer) condensed overview of CLEF 2026 LongEval. It cannot be accessed openly. The CEUR extended overview should be a superset of it.
- Teams air-benders, nlp hashiras and air-02 have no paper, so their pipelines cannot be verified beyond the run names.

## Q2. Unread 2025 LongEval papers (CEUR-WS Vol-4038): RAND, SARD, DataHunter, CIR, CIR_cluster

### Takeaway
None of the five papers uses ANN or vector-index search, and none reports an index configuration or studies index updates. Their pipelines are Lucene/PyTerrier BM25 or Boolean retrieval, with neural models used only for re-ranking or for computing similarities. Searching the full text of *all* 13 LongEval 2025 papers (paper_264–276) for ANN terms found a hit in only one: AGH IR (paper_276). It discusses ANN, HNSW and FAISS as background, but its implementation section describes cosine similarity over vectors cached in NumPy files, which is apparently exact.

### Cited Findings
- **paper_265 – DataHunter at LongEval: Temporal Stability Analysis of Boolean and CamemBERT-Based Retrieval Systems** (University of Padua course team) — [PDF](https://ceur-ws.org/Vol-4038/paper_265.pdf)
  - Boolean retrieval vs CamemBERT neural reranking over six monthly web snapshots (March–August 2023), compared with two-way ANOVA and Tukey HSD. Both systems drop in August, which the authors attribute to "collection shift".
  - Their one "dense retrieval" hit is in related work. CamemBERT is a reranker, so no vector index is used and there is no ANN.
- **paper_267 – Team RAND at LongEval: Composable Information Retrieval with Semantic and Language-Aware Components** — [PDF](https://ceur-ws.org/Vol-4038/paper_267.pdf)
  - Java/Lucene parse–index–search pipeline with synonym query expansion and filters. An optional semantic re-ranking (DL model or LLM) runs only when the top Lucene score falls below a threshold. No vector index and no ANN.
- **paper_269 – CIR at LongEval 2025: Exploring Temporal Sensitivity in Web Retrieval** (Braun, Busch, et al.) — [PDF](https://ceur-ws.org/Vol-4038/paper_269.pdf)
  - PyTerrier BM25 with:
    - LLM-classified time-dependent query boosting;
    - tf-idf relevance feedback from older sub-collections;
    - boosting of documents judged relevant in earlier snapshots, matched across versions by Sentence-BERT (all-MiniLM-L6-v2) pairwise cosine similarity or Jaccard similarity;
    - neural re-ranking with semantic clustering.
  - The embeddings compare document versions pairwise. No ANN index is used and no ANN terms appear.
- **paper_271 – SARD at LongEval: On Longitudinal Evaluation of IR Systems by Using Query Rewriting and Hybrid Queries** — [PDF](https://ceur-ws.org/Vol-4038/paper_271.pdf)
  - Apache Lucene with varied analyzers, language detection, Boolean and phrase queries, synonym expansion, and GPT-4 Turbo query rewriting. Indexing was multithreaded for speed. "Indexing configurations" here means lexical analyzer choices. No vectors and no ANN.
- **paper_275 – CIR_cluster at LongEval 2025: Clustering Query Variants for Temporal Generalization** (Muanza Ndiema, et al.) — [PDF](https://ceur-ws.org/Vol-4038/paper_275.pdf)
  - BM25 followed by re-ranking from the previous snapshot's qrels, extended to query variants found by clustering query embeddings (Lajavaness/bilingual-embedding-large; its references include k-means). Clustering did not improve average effectiveness. No ANN index or ANN terms.
- **paper_276 – AGH IR at LongEval: Improving Scientific Information Retrieval with Dense Representations and Cross-Encoder Re-ranking** (Stryszewski, Prosowicz, Kawiak, Jakowiec). This is outside the five named papers but surfaced by the corpus-wide search. — [PDF](https://ceur-ws.org/Vol-4038/paper_276.pdf)
  - Its keywords include "Approximate Nearest Neighbor (ANN)". Its background section (§3.3) describes ANN, LSH, HNSW and "The Faiss library".
  - Its implementation, however, lists "Embedding Storage: NumPy files to cache vectors. Query Evaluation: cosine similarity search, re-ranking by cross-encoder", with e5-large-v2 / e5-base-v2 / all-MiniLM-L6-v2 bi-encoders and Top-K = 200.
  - It gives no FAISS or HNSW parameters, makes no approximate-vs-exact comparison, and does not study index updates.
- Searching the other 2025 papers (264 extended abstract, 266 OpenWebSearch, 268 BASETTE, 270 3DS2A, 272 RISE, 273 RACOON, 274 DS@GT) for the ANN regex (HNSW|FAISS|PLAID|nearest neighbo|approximate nearest/search|ANN|IVF|brute force|exhaustive search|vector database/store/index|Qdrant|Milvus|Chroma|Weaviate|pgvector|knn) found **no hits**. The only exception is a single "ANN" token in paper_266 that did not match the stricter regex. — [Vol-4038 index](https://ceur-ws.org/Vol-4038/)

### Inferences
- AGH IR (2025) is the only LongEval paper that names ANN or FAISS. It is background text, and its NumPy-based implementation points to exact search. It should be cited as "acknowledged ANN, did not use or evaluate it". The NumPy reading is my interpretation of their implementation list; the paper never says "exact".

### Gaps
- I did not fetch the code of AGH IR or the other 2025 teams to confirm exact vs ANN.

## Q3. The 2026 overview's findings on what drives effectiveness change, and the organisers' dense baseline (exact or ANN?)

### Takeaway
The overview attributes falling effectiveness to the collection growing, not to the index. It says temporal robustness "does not seem to depend on any particular approach class". The organisers' Qwen3-Embedding-4B baseline uses pyterrier-dr `FlexIndex` with its default retriever, which is `NumpyRetriever`: an exact brute-force matrix-product search. It builds a fresh index from scratch for each snapshot. ANN therefore played no part in the official dense baseline, and rebuild-vs-incremental was never at issue.

### Cited Findings
- On drivers of change, the overview says: "For almost all systems, the effectiveness measured on the DCTR qrels decreases over time. Meanwhile, the collection size increases, which might explain this observation." It also finds that "effectiveness and robustness seem to be independent goals" and that "temporal robustness does not seem to depend on any particular approach class". — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- Further, it reports that "document recency does not appear to be a driving factor of relevance on this testbed", that citation data had "only a marginal positive effect", and that "the ranking of systems is more dynamic" than in 2025, judged by lower rank correlation. — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- On the synthetic GPT-OSS-120b qrels, "the Qwen3 baseline performs better and is consistently outperformed by only four approaches over time". — [paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- The baseline code (`clef26/scientific-retrieval/baseline-pyterrier-dense/baseline.py`) does the following — [GitHub baseline.py](https://github.com/clef-longeval/longeval-code/blob/main/clef26/scientific-retrieval/baseline-pyterrier-dense/baseline.py):
  - builds `FlexIndex(index_directory / "my_index.flex")` from a vLLM-served `Qwen/Qwen3-Embedding-4B` encoder;
  - retrieves with `vllm_encoder >> index.retriever()`;
  - runs `process_dataset(snapshot, index / snapshot.get_snapshot(), …)` once for each sub-collection, so one independent index per snapshot, rebuilt from scratch.
- The run's ir-metadata records `retrieval: - name: NumpyRetriever`, `indexing: dense: vLLM Encoder with pyterrier-dr`, with pyterrier-dr 0.7.0 and pyterrier 1.0.4, and "We use Qwen3-Embedding-4B and leave everything to the defaults of the model and PyTerrier." — [GitHub ir-metadata.yml](https://github.com/clef-longeval/longeval-code/blob/main/clef26/scientific-retrieval/baseline-pyterrier-dense/baseline-qwen3-embedding-4b/ir-metadata.yml)
- In the pyterrier-dr source, `FlexIndex.retriever = _np_retriever # default retriever`. The `_np_retriever` docstring reads "Return a retriever that uses numpy to perform a brute force search over the index" (scores = `query_vecs @ doc_batch`). The README separately offers `faiss_hnsw_retriever()` and states that `np_retriever` "performs an exhaustive (exact) search using numpy". — [pyterrier_dr flex/np_retr.py](https://github.com/terrierteam/pyterrier_dr/blob/master/pyterrier_dr/flex/np_retr.py); [pyterrier_dr README](https://github.com/terrierteam/pyterrier_dr)
- The pre-built indexes are published as a HuggingFace collection ("longeval-2026-baseline-dense", e.g. `jueri/longeval-2026-snapshot-1-index`). Paper 379 used these, so its QwenTemporalIR dense runs were most likely exact too. — [baseline README](https://github.com/clef-longeval/longeval-code/tree/main/clef26/scientific-retrieval/baseline-pyterrier-dense); [paper379](https://ceur-ws.org/Vol-4283/paper379.pdf)

### Inferences
- The official dense baseline is exact and rebuilt per snapshot, and Lexora reused it. So no LongEval 2026 dense result contains ANN error, and none involves incremental index maintenance.
- This makes the proposal's experiment a clean addition: swap the exact FlexIndex for HNSW/IVF (pyterrier-dr already offers `faiss_hnsw_retriever`), and compare rebuild-per-snapshot with incremental insertion, with the organisers' exact runs as ground truth. The released per-snapshot exact indexes can serve directly as the exact reference.
- Paper 379's reasoning about a "static" index is mistaken in light of this code (see Q1). It should be cited carefully, if at all.

### Gaps
- I could not confirm that the HuggingFace artifacts loaded by paper 379 (`…-index-dense`) are the same as the `…-index` ones named in the README. The naming differs slightly, and I did not open the HuggingFace pages.
- Paper 379 does not name the retriever method it called. "Exact" for its runs is inferred from the pyterrier-dr default.
