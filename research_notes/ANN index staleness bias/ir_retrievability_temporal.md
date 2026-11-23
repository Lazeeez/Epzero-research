# IR framing: retrievability, exposure, ANN approximation and temporal evaluation (for an "ANN index arrival-time bias" claim)

Research date: 2026-11-23. About 24 tool calls. "Preprint" means arXiv only, with no peer-reviewed version confirmed in this session. Where a venue is inferred and not confirmed, the entry says so.

## Q1. Retrievability and its follow-ups (measures, dense retrievers, document-side biases)

### Takeaway
Retrievability is the standard IR concept for "does a system make some documents harder to reach regardless of relevance". It is usually reported as a Gini coefficient or Lorenz curve over r(d) scores computed from large synthetic query sets. Since 2023 there has been a small, active line of work applying it to dense or neural retrievers (CIKM 2023, CIKM 2025, EMNLP 2026 Findings). None of that work looks at document arrival time or at the ANN index as a source of bias.

### Cited Findings
- **Origin.** Azzopardi & Vinay, "Retrievability: an evaluation measure for higher order information access tasks", CIKM 2008, pp. 561–570, DOI 10.1145/1458082.1458157. The paper frames retrievability as a measure of how a system shapes access to documents, beyond effectiveness and efficiency, and tests it on two TREC collections using query-based sampling. Peer-reviewed. — [Strathclyde record](https://pureportal.strath.ac.uk/en/publications/retrievability-an-evaluation-measure-for-higher-order-information/); [DBLP](https://dblp.org/rec/conf/cikm/AzzopardiV08.html)
- **Applications.** Retrievability has been used to detect search-engine and algorithmic bias, tune IR systems and assess collection quality (a tertiary summary). — [Wikipedia: Retrievability](https://en.wikipedia.org/wiki/Retrievability)
- **Follow-up on integrated systems.** "Retrievability in an Integrated Retrieval System: An Extended Study" (arXiv 2303.15036) restates that the measure comes from Azzopardi & Vinay 2008 and uses query-based sampling. — [arXiv 2303.15036](https://arxiv.org/pdf/2303.15036)
- **Time-related retrievability (closest temporal precedent).** Samar et al., "Quantifying retrieval bias in Web archive search", *International Journal on Digital Libraries* (2018; DOI 10.1007/s00799-017-0215-9). They measure retrievability by document timestamp in a web archive and report that it varies across time frames at 1-year granularity. Documents with more versions get higher retrievability scores, and users prefer older documents. Peer-reviewed. The bias comes from the ranker and the collection, not from an ANN index. — [Springer](https://link.springer.com/article/10.1007/s00799-017-0215-9); [PDF](https://d-nb.info/1131140923/34)
- **Bias against new content in a streaming system.** "How UMass-FSD Inadvertently Leverages Temporal Bias" (arXiv 2208.01347) finds that the novelty scores of newly arriving documents are lowered when they are relevant to older, already-known topics, and traces this to an implementation choice. This precedent fits the framing "an implementation detail, not the model, disadvantages new arrivals". Preprint status not re-checked. — [arXiv 2208.01347](https://arxiv.org/pdf/2208.01347)
- **Dense and PLM rankers, CIKM 2023.** "Retrievability Bias Estimation Using Synthetically Generated Queries" (CIKM 2023, DOI 10.1145/3583780.3615221; Leiden) compares the retrievability bias (Gini) of a BERT bi-encoder and a cross-encoder re-ranker against BM25, using docT5query-generated queries. It says no earlier work evaluated PLM rankers for retrievability bias, and that BM25 was among the least biased models in prior work. — [ACM DL](https://dl.acm.org/doi/10.1145/3583780.3615221); [Leiden PDF](https://scholarlypublications.universiteitleiden.nl/access/item:3718746/download)
- **Training data effects.** "Improving Content Retrievability in Search with Controllable Query Generation" (arXiv 2303.11648) reports that a dense model trained on click data and a pre-trained bi-encoder both show significantly more retrievability bias than BM25 (Gini +9.2% and +10%). Fine-tuning on synthetic queries reduces the bias. Venue not confirmed in this session. — [arXiv 2303.11648](https://arxiv.org/pdf/2303.11648)
- **Topic-local retrievability, CIKM 2025.** Chang, Meng & Ganguly, "T-Retrievability: A Topic-Focused Approach to Measure Fair Document Exposure in Information Retrieval", CIKM 2025 (Seoul, 10–14 Nov 2025; arXiv v1 29 Aug 2025). They argue that uneven r(d) can reflect topical differences rather than exposure bias, so they compute retrievability within groups of topically related documents and then aggregate. They apply this to neural rankers. Peer-reviewed. — [arXiv 2508.21704](https://arxiv.org/abs/2508.21704); [ACM DL](https://dl.acm.org/doi/10.1145/3746252.3760820)
- **Modern embedding models.** "From Topology to Retrieval: Decoding Embedding Spaces with Unified Signatures" (arXiv 2511.22150) computes Gini over top-100 retrievability for dense embedding models. It reports that decoder-based embedders show more exposure inequality on every dataset except ArguAna. Preprint; numbers not checked against the full text. — [arXiv 2511.22150](https://arxiv.org/pdf/2511.22150)
- **Learned document priors in bi-encoders.** Valentini, Altszyler & Fajcik, "Do Neural Retrievers Prefer Certain Documents? Evidence of Learned Relevance Priors" (arXiv 2606.02814; v1 1 Jun 2026, v2 2 Nov 2026; camera-ready for EMNLP 2026 Findings per the arXiv comments). Supervised bi-encoders learn a query-independent document-level relevance prior. Low-prior documents are harder to retrieve even when relevant, and the gap is weaker for BM25. The priors relate to content (comprehensive, mainstream versus niche or technical) and not to document age. — [arXiv 2606.02814](https://arxiv.org/abs/2606.02814)
- **Other document-side biases in neural retrievers.**
  - "Source bias": neural retrievers favour LLM-generated text, and re-rankers more so than first-stage retrievers (Dai et al., KDD 2024 per search snippet). — [arXiv 2310.20501](https://arxiv.org/pdf/2310.20501)
  - Brevity, position, literal-match and repetition biases in Dragon+ and Contriever ("Collapse of Dense Retrievers", ACL 2025 per search listing). — [arXiv 2503.05037](https://arxiv.org/abs/2503.05037)
  - "Search-by-Answer Bias in Retrieval Models" (preprint, Nov 2026). — [arXiv 2611.11396](https://arxiv.org/abs/2611.11396)
  - "Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking" (preprint, 2025), a *recency* bias at the reranker level. — [arXiv 2509.11353](https://arxiv.org/pdf/2509.11353)

### Inferences
- The arrival-time finding fits as a retrievability or exposure claim with a new cause (the ANN index) and a new grouping variable (arrival snapshot). The precedents group r(d) by document type, topic (T-Retrievability), source (LLM vs human) or timestamp in a static archive (Samar et al.). None groups by when the document entered a growing index.
- Reviewers who know this literature will expect r(d) with Gini or Lorenz, or at least a group-wise comparison. A natural contribution is to report ΔGini or Δr(d) between exact and HNSW retrieval, split by arrival cohort. The T-Retrievability critique (differences may be topical) means new and old documents should be compared *within topic* or with query-matched controls.
- The Valentini et al. prior finding is a confound to rule out. If newly arrived documents differ in content (new topics, more niche), a dense model may rank them lower under exact search as well. The ANN-specific claim must be stated relative to exact search with the same embeddings.

### Gaps
- I did not verify the Wilkie & Azzopardi ECIR 2015 comparison of inequality measures (Gini vs alternatives) or the classic Lorenz-curve papers in this session. Look them up on DBLP before citing.
- I did not verify the venues of arXiv 2303.11648 and 2208.01347.
- **NOT FOUND:** any retrievability study that varies the index (exact vs ANN) while holding the embedding model fixed, and any that groups documents by arrival time in a growing collection.

## Q2. Exposure fairness and fair ranking: anything at the index or ANN level?

### Takeaway
Exposure fairness is well established (Singh & Joachims 2018, Biega et al. 2018, Diaz et al. 2020, TREC Fair Ranking 2019–2022). It is defined on rankings and policies, not on candidate generation or the ANN index. Fairness has been studied in near-neighbour search theory, but in a different sense: uniform sampling among valid neighbours. One 2026 recommender preprint links deterministic ANN to long-tail exposure. I found no IR paper on exposure unfairness caused by ANN approximation error.

### Cited Findings
- **Singh & Joachims, "Fairness of Exposure in Rankings"** (KDD 2018; arXiv 1802.07281). Fairness constraints on rankings are expressed as exposure allocation, with algorithms that maximise utility under those constraints. The KDD venue is widely known but was not confirmed by this session's search results. — [arXiv 1802.07281](https://arxiv.org/pdf/1802.07281)
- **Diaz, Mitra, Ekstrand, Biega & Carterette, "Evaluating Stochastic Rankings with Expected Exposure"** (CIKM 2020, pp. 275–284). They define expected exposure as the average attention an item gets over repeated samples of the same query, and state the equal-expected-exposure principle for items of equal relevance. Peer-reviewed. — [arXiv 2004.13157](https://arxiv.org/pdf/2004.13157); [NSF PAR](https://par.nsf.gov/biblio/10199451-evaluating-stochastic-rankings-expected-exposure)
- **TREC Fair Ranking.**
  - 2019: fair exposure of authors, using Semantic Scholar data. — [TREC 2019 overview](https://arxiv.org/pdf/2003.11650)
  - 2020: systems return multiple rankings per query and are scored on expected group exposure. — [TREC 2020 overview](https://arxiv.org/pdf/2108.05135)
  - 2021: Wikipedia, with expected exposure against a target distribution plus a work-needed criterion. — [TREC 2021 overview](https://arxiv.org/pdf/2302.10856)
- **Fair near-neighbour search (theory).** Aumüller, Har-Peled, Mahabadi, Pagh & Silvestri, "Sampling a Near Neighbor in High Dimensions — Who is the Fairest of Them All?", ACM TODS (2022), DOI 10.1145/3502867. LSH gives no fairness guarantee, so they study sampling uniformly among the points in a query's neighbourhood. Here fairness means every valid neighbour has equal chance of being reported. That is conceptually close to "every relevant document should be equally reachable", though it is not about arrival time or graph indexes. Peer-reviewed. Author list from my own knowledge. — [ACM DL](https://dl.acm.org/doi/10.1145/3502867)
- **Deterministic ANN and long-tail exposure (recommendation).** "Distributional Approximate Nearest Neighbour Search for Uncertainty-Aware Retrieval" (arXiv 2606.04603, 2026). Point-estimate ANN imposes hard geometric cutoffs on what reaches the top-k, and the baseline ANN gave the least popular items 0% catalogue coverage. Seen only through a Bytez mirror; unverified preprint, recommender setting. — [Bytez mirror](https://bytez.com/docs/arxiv/2606.04603/paper)
- **Candidate generation as exposure gatekeeper.** In multi-stage dense retrieval, ANN scores are often too inaccurate to rank with but adequate for choosing candidates. Documents the ANN misses never reach the re-ranker. Source: Macdonald & Tonellotto, "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval" (2021; I believe CIKM 2021 short paper, not confirmed). — [arXiv 2108.11480](https://arxiv.org/html/2108.11480v1); [Glasgow eprint](https://eprints.gla.ac.uk/249263/3/249263.pdf)

### Inferences
- The finding can be cast as an exposure gap caused by the retrieval infrastructure and not by the ranking policy. Fair-ranking metrics assume the candidate set is given, so a document the ANN drops gets zero exposure whatever the re-ranker or fairness intervention does. That argument is clean and new.
- "Equal expected exposure for equally relevant items" (Diaz et al.) gives a direct normative yardstick. The claim then reads: among documents of equal relevance grade, arrival cohort predicts exposure under HNSW but not under exact search.
- Calling this "fairness" may draw pushback, because fairness usually concerns protected groups or producers. "Temporal exposure bias" or "arrival-time retrievability bias" are safer terms. Producer fairness (new authors, new papers) is the hook if fairness is wanted.

### Gaps
- **NOT FOUND:** any paper on exposure, fairness or retrievability caused by ANN index approximation (HNSW, IVF or PQ) in document retrieval.
- Not checked: Biega, Gummadi & Weikum, "Equity of Attention" (SIGIR 2018), the other standard citation. Verify its DOI before use.

## Q3. Studies of how ANN approximation changes *which* documents are retrieved (not only average effectiveness)

### Takeaway
IR and RAG work on ANN reports aggregate effects. HNSW typically lowers effectiveness slightly and non-deterministically (Lin, ACL 2025 Industry). Lower ANN recall costs RAG little on average (Leto et al., NeurIPS 2024 workshop), though systems work reports end-to-end accuracy losses at low efSearch. The vector-DB literature shows ANN errors are not uniform: some points are unreachable in HNSW after updates, and per-query recall is uneven. Nobody connects these per-item effects to document retrievability or to arrival time in an IR evaluation.

### Cited Findings
- **Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?"** (arXiv 2409.06464; ACL 2025 Industry Track). Flat brute-force indexes are viable for smaller corpora. HNSW scores are usually lower because the search is approximate, and HNSW construction is non-deterministic, so scores vary between trials. The paper uses BEIR with Lucene. Peer-reviewed. — [ACL Anthology](https://aclanthology.org/2025.acl-industry.61); [arXiv](https://arxiv.org/abs/2409.06464)
- **Leto et al., "Toward Optimal Search and Retrieval for RAG"** (arXiv 2411.07396; NeurIPS 2024 workshop). Lowering search accuracy has minor RAG implications. At search recall@10 of 0.7 versus exact search (BGE-base), gold-document recall drops only about 2–3%. Workshop paper. — [arXiv](https://arxiv.org/abs/2411.07396); [NeurIPS listing](https://neurips.cc/virtual/2024/105295)
- **"Accelerating Retrieval-Augmented Generation"** (arXiv 2412.15246). HNSW could not match exact search in end-to-end generation accuracy while being more than 2× faster, and lower efSearch gives higher throughput but lower generation accuracy. This conflicts in emphasis with Leto et al. I have not confirmed the venue (believed ASPLOS 2025). — [arXiv 2412.15246](https://arxiv.org/pdf/2412.15246)
- **Per-query non-uniformity.** A 2026 paper on vector search in databases reports that at an average recall of 0.99, indexes still left 5–19% of in-distribution queries below that target. Only a search-result summary was seen; it was not fetched. The listing was "Abstract", arXiv 2611.07579. — [arXiv 2611.07579](https://arxiv.org/html/2611.07579)
- **Unreachable points in HNSW.** Xiao et al., "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation" (arXiv 2407.07871, 2024). After about 3,000 delete/reinsert cycles on SIFT, 3–4% of points become unreachable. The roughly 3% recall loss cannot be recovered by raising ef. Preprint. The mechanism is deletion and replacement, not insert-only growth. — [arXiv 2407.07871](https://arxiv.org/abs/2407.07871v1)
- **Similar update-driven degradation in product catalogues.** A CEUR workshop paper reports 1–3% recall degradation across update cycles. — [CEUR Vol-4250 paper776](https://ceur-ws.org/Vol-4250/paper776.pdf). The same theme appears in "Graph-based Nearest Neighbors with Dynamic Updates via Random Walks" (arXiv 2512.18060). — [arXiv 2512.18060](https://arxiv.org/html/2512.18060v1)
- **Filtered ANN.** Recall falls as filter selectivity rises, so predicate-filtered subsets of the corpus are systematically under-retrieved. — [arXiv 2602.11443](https://arxiv.org/pdf/2602.11443)
- **Background.** Most dense-retrieval papers use FAISS without discussing the ANN algorithm (Zhao et al., dense retrieval survey). — [arXiv 2211.14876](https://arxiv.org/pdf/2211.14876)

### Inferences
- The gap the paper fills is that every IR/RAG ANN study reports query-averaged recall or nDCG, while ANN errors are concentrated on particular items. The project's observation is this exactly: nDCG@10 gap changes below 0.01, yet a 2–3× disparity by arrival cohort. It is the IR counterpart of the "average recall hides per-query misses" point from the systems side.
- LongEval-Sci 2026 snapshots are insert-only (see Q5), so the Xiao et al. deletion mechanism does not explain the result directly. Reviewers will ask what does: insertion order in graph construction, entry-point and layer assignment, a local-density or hubness difference, or the distribution of new documents in embedding space. The paper needs a mechanism or at least ablations, for example rebuilding the index from scratch in a shuffled order versus incremental insertion.

### Gaps
- **NOT FOUND:** any IR, RAG or vector-search paper measuring whether ANN errors correlate with document age or insertion order in an append-only index.
- **NOT FOUND:** papers titled along the lines of "ANN search effect on dense retrieval fairness", "approximate search fairness" or "vector search bias" in the document-exposure sense. Searches for these phrasings returned only the loosely related items above.
- I did not search IVF/PQ-specific per-item studies separately. Hubness literature (e.g. Radovanović et al., JMLR 2010) was not verified in this session.

## Q4. Freshness and recency: why new documents matter (web search, scientific literature, RAG)

### Takeaway
Recency has been a first-class ranking problem in web search since at least WSDM 2010, and search engines reward fresh content. LLM and RAG work shows models fail on fast-changing knowledge unless retrieval surfaces current evidence. For scholarly search, LongEval-Sci 2026 itself finds that recency does not drive click relevance on CORE. That cuts against a pure "freshness matters" argument for this collection.

### Cited Findings
- **Dong et al., "Towards Recency Ranking in Web Search"** (WSDM 2010, DOI 10.1145/1718487.1718490). They detect recency-sensitive queries with a high-precision classifier and train a dedicated ranking model with recency features and recency-aware metrics. Gains: +1.1% NDCG5,demote and +5.2% ΔDCG5,demote on classified recency queries. Peer-reviewed. — [VideoLectures](https://videolectures.net/wsdm2010_dong_trr/); [Springer Professional DOI copy](https://www.springerprofessional.de/doi/pdf/10.1145/1718487.1718490)
- **Session click features for recency ranking** (AAAI). — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/7517)
- **Search engines reward freshness.** Pages that appear freshly updated are rewarded, on the assumption that newer content better satisfies users (citing Dong et al. among others). Preprint, 2025. — [arXiv 2509.11353](https://ar5iv.labs.arxiv.org/html/2509.11353)
- **Vu et al., "FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation"** (Findings of ACL 2024; arXiv 2310.03214). Introduces FreshQA. More than 50K human judgments show all models struggle with fast-changing knowledge and false premises. FreshPrompt injects Google Search evidence into the prompt, and the number and order of retrieved evidence strongly affect correctness. Peer-reviewed. — [ACL Anthology](https://aclanthology.org/2024.findings-acl.813); [arXiv](https://arxiv.org/abs/2310.03214)
- **Dense retrievers on evolving corpora.** "When Should Dense Retrievers Be Updated in Evolving Corpora? Detecting Out-of-Distribution Corpora Using GradNormIR" (Findings of ACL 2025; arXiv 2506.01877). Indexing newly emerging documents without updating the retriever can degrade retrieval for future queries, while unnecessary retraining can also hurt. — [ACL Findings PDF](https://aclanthology.org/2025.findings-acl.1334.pdf); [arXiv](https://arxiv.org/html/2506.01877)
- **Generative retrieval vs dual encoders on dynamic corpora.** Dual encoders need full re-indexing when the model is updated (arXiv 2305.18952; I believe EMNLP 2024, not confirmed). — [alphaXiv 2305.18952](https://www.alphaxiv.org/abs/2305.18952)
- **Temporal generalisation gap.** A 2026 survey frames temporal drift as a growing mismatch between a frozen retriever and shifting distributions, crediting Lazaridou et al. for showing degradation on post-training data. Preprint. — [arXiv 2601.20131](https://arxiv.org/pdf/2601.20131)
- **Benchmarks drift too.** "Evaluating Temporal Drift in Retrieval Benchmarks" (FreshStack, arXiv 2603.04532, 2026). Preprint; seen only in a search listing. — [arXiv 2603.04532](https://arxiv.org/pdf/2603.04532)
- **Static-index assumption in RAG.** Most RAG studies assume a static corpus whose index is built once. — [arXiv 2510.13590](https://arxiv.org/html/2510.13590v1)
- **LongEval-Sci 2026: recency is weak on this testbed.** The overview says "document recency does not appear to be a driving factor of relevance on this testbed". Temporal signals helped more under DCTR qrels than under LLM judgments. Peer-reviewed working notes. — [CEUR Vol-4283 paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)

### Inferences
- The motivation should not rest on "new documents are more relevant". On LongEval-Sci they are not clearly so, and a reviewer can quote the overview against that claim. The robust framing is that relevant new documents are systematically under-exposed relative to equally relevant old ones. That is a fairness and retrievability argument and does not depend on recency being relevant. Freshness-sensitive settings (news, fast-moving science, RAG over updating corpora; see FreshLLMs, GradNormIR) supply the cost argument.
- The GradNormIR line concerns model staleness: the encoder does not understand new documents. The project's claim concerns index staleness with the model held fixed. The paper should draw this distinction explicitly, because reviewers may conflate the two.

### Gaps
- StreamingQA (Liška et al., ICML 2022) and TemporalWiki (Jang et al., EMNLP 2022) were not verified in this session. Confirm them on DBLP before citing.
- I found no quantitative estimate of the cost of missing new scientific documents (for example, impact on literature reviews).

## Q5. Temporal IR and LongEval (2023–2026), especially LongEval-Sci, and dense retrieval or ANN in LongEval

### Takeaway
LongEval (CLEF 2023–2026) studies temporal persistence: effectiveness and system rankings drift as collections evolve. The 2026 LongEval-Sci collection is a cumulative, insert-only CORE sample in three 3-month 2025 snapshots, with click-derived (DCTR) qrels and GPT-OSS-120b synthetic labels. Almost all runs were lexical-first. No LongEval 2025–2026 paper studies ANN indexing, and the only ANN mention in the 2026 overview concerns building the RAG task data.

### Cited Findings
- **LongEval 2025 (third edition).**
  - Covers temporal persistence: how performance degrades as test data drift from training data. Adds SciRetrieval (CORE) alongside web retrieval. — [arXiv 2503.08541 (ECIR 2025 lab teaser)](https://arxiv.org/abs/2503.08541)
  - LNCS overview "LongEval at CLEF 2025: Longitudinal Evaluation of IR Systems on Web and Scientific Data". — [Springer DOI 10.1007/978-3-032-04354-2_20](https://dl.acm.org/doi/10.1007/978-3-032-04354-2_20)
  - Extended abstract in CEUR Vol-4038: 19 teams; measures are nDCG@10, nDCG@1000, Relative Improvement and Delta RI, plus rank stability via Pearson and Kendall's τ. Participation counts differ between sources (one summary: 45 web and 23 sci runs; arXiv: 14 teams and 73 web runs). — [CEUR Vol-4038 paper264](https://ceur-ws.org/Vol-4038/paper_264.pdf); [arXiv 2509.17469](https://arxiv.org/pdf/2509.17469)
- **LongEval 2026 overview.** "Overview of the CLEF 2026 LongEval Lab on Longitudinal Evaluation of Model Performance", LNCS chapter DOI 10.1007/978-3-032-39150-6_32. The Springer page redirected to a login and was not read. Extended overview by Bouaraba, Breuer, Cancellieri, El-Ebshihy, Fröbe, Galuščáková, Goeuriot, Iturra-Bocaz, Keller, Knoth, Kruff, Mulhem, Piroi, Pride, Schaer and Schwab: CEUR Vol-4283 paper374 (CLEF 2026, Jena, 21–24 Sep 2026). There are four tasks (Sci ad-hoc, TopEx, USim, RAG) and 104 runs in total. — [Springer](https://link.springer.com/chapter/10.1007/978-3-032-39150-6_32); [CEUR paper374](https://ceur-ws.org/Vol-4283/paper374.pdf); [CEUR Vol-4283 index](https://ceur-ws.org/Vol-4283/)
- **LongEval-Sci 2026 collection details** (from the extended overview text) — [CEUR paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
  - **Snapshots:** S1 Mar–May, S2 Jun–Aug, S3 Sep–Nov 2025.
  - **Insert-only:** "documents are only added to the snapshots but never removed", meant to simulate a scientific collection where publications are added but almost never removed. Snapshots are strictly cumulative supersets.
  - **How documents were chosen:** documents from each query's click logs and logged SERPs, plus the top 1,000 documents in CORE's January 2026 ranking. Each snapshot adds "documents newly appearing in the system rankings" plus a proportional number of random documents, "since these documents are likely to be relevant".
  - **Sizes:** stated as "approximately 87K, 1.3M, and 1.7M". The 87K is very likely a typo for about 870K, which matches the project's 869K. The project's description of S3 as 1.66M matches 1.7M.
  - **Queries:** 434 queries with at least 30 clicks, plus 47 RAG questions. DCTR-evaluable queries per snapshot: 149, 100 and 136. Only 12 queries appear in all three snapshots.
  - **Training data:** S1, 100 queries, 2,275 raw and 10,406 DCTR assessments.
  - **Synthetic labels:** GPT-OSS-120b labels from top-10 pooling over Task 1 runs.
- **Task 1 runs and results** — [CEUR paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
  - 20 runs from eight teams. All use a lexical ranker somewhere; three use a dense component; 15 are multi-stage.
  - Baselines: BM25 (PyTerrier) and Qwen3-4B embeddings over abstracts.
  - Best DCTR nDCG@10 in S1 was Dense-Fusion-Run (0.326), a BM25+LTR model with document-age, publication-count and embedding features.
  - Effectiveness on DCTR qrels falls across snapshots for almost all systems, "the collection size increases, which might explain this observation".
  - System-ranking correlation versus S1: Pearson 0.458 and 0.479; Kendall's τ 0.574 and 0.515. The rankings are more dynamic than in 2025. Effectiveness and robustness behave as competing goals, and robustness does not depend on approach class.
  - Measures are Result Delta (González Sáez), RC = (ARP1 − ARP2)/ARP1, and DRI against a BM25 pivot.
  - LLM labels barely vary across snapshots ("temporal asymmetry": a frozen LLM cannot capture temporal changes in relevance).
- **ANN in LongEval 2026.** The only ANN mention in the overview is that LongEval-RAG candidate pairs were found with SPECTER2 embeddings and ANN search (similarity above 0.95 plus a shared author). — [CEUR paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- **LongEval 2026 participant papers (CEUR Vol-4283, papers 374–385).** Titles include "Temporal Stability in Scientific Information Retrieval: A Three-System Analysis for CLEF LongEval 2026" (paper379) and "Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci" (paper384; arXiv 2607.04088). No title mentions ANN, HNSW, FAISS or vector indexes. Only titles were checked, not full texts. — [CEUR Vol-4283](https://ceur-ws.org/Vol-4283/); [arXiv 2607.04088](https://arxiv.org/pdf/2607.04088)
- **Task page and data access.** — [LongEval tasks](https://clef-longeval.github.io/tasks/); [ir_datasets-longeval](https://github.com/clef-longeval/ir-datasets-longeval/tree/main)

### Inferences
- LongEval's core finding is that effectiveness and system rankings drift as collections grow. One untested explanation for part of that drift with dense first stages is index-side. Shown that way, the paper sits directly inside LongEval's research question: the index is a temporal-persistence factor, not only the model or the queries.
- Insert-only cumulative snapshots match HNSW's native operations (insert and query), so the "unreachable points after deletion" literature does not apply directly. The finding would be a new failure mode for append-only growth. Mechanism evidence would strengthen this considerably.
- The 2026 lab was almost entirely lexical-first with dense re-ranking or fusion. A dense-first-stage, ANN-versus-exact longitudinal study is therefore new relative to what LongEval participants did.

### Gaps
- I did not read the LongEval 2023 and 2024 overviews (Alkhalifa et al.; Galuščáková et al.) in this session, so their key findings (for example, the 2023 result that neural systems degraded less than lexical ones) are not cited here. Verify them before use.
- The full texts of the 2026 participant papers were not checked for dense or ANN content.
- The LNCS 2026 overview page could not be read (login redirect).
- **NOT FOUND:** any LongEval 2023–2026 paper on ANN indexing, index maintenance, or retrievability of newly added documents.

## Q6. What CLEF/ECIR/SIGIR reviewers would expect as evidence for a per-document bias claim

### Takeaway
Reviewers will want three things. First, the bias must hold on relevance-based criteria, not only on neighbour recall against exact search. Second, judgment artefacts must be ruled out: LongEval qrels come from clicks on CORE's own rankings, and pooling is known to be biased on large collections. Third, there must be controls for content and topic confounds and for HNSW non-determinism, with significance testing across multiple index builds.

### Cited Findings
- **Pool bias grows with collection size.** Buckley, Dimmick, Soboroff & Voorhees, "Bias and the limits of pooling for large collections", *Information Retrieval* 10(6):491–508, 2007 (DOI 10.1007/s10791-007-9032-x; earlier SIGIR 2006 short paper). They show pooling bias on AQUAINT (TREC 2005) and suspect it on GOV2. Peer-reviewed. — [Springer](https://link.springer.com/doi/10.1007/s10791-007-9032-x); [NIST](https://www.nist.gov/publications/bias-and-limits-pooling-large-collections)
- **LongEval-Sci 2026 qrels are click-derived.** DCTR pseudo-relevance comes from CORE's logged SERPs; raw qrels mark every clicked document relevant. The overview notes that click-based pseudo-qrels carry position bias, and the synthetic labels come from top-10 pooling of participant runs. — [CEUR paper374](https://ceur-ws.org/Vol-4283/paper374.pdf)
- **HNSW non-determinism.** HNSW construction is non-deterministic, so effectiveness varies between builds. — [Lin, ACL 2025 Industry](https://aclanthology.org/2025.acl-industry.61)
- **Unjudged-document priors.** Annotation covers only part of the corpus, which induces document priors in supervised retrievers; niche or technical content is often unjudged. — [arXiv 2606.02814](https://arxiv.org/abs/2606.02814)
- **Topic confounds.** Differences in r(d) can reflect topic rather than bias, so comparisons should be made within topic groups. — [T-Retrievability, CIKM 2025](https://arxiv.org/abs/2508.21704)
- **The yardstick.** Equal expected exposure for items of equal relevance grade. — [Diaz et al., CIKM 2020](https://arxiv.org/pdf/2004.13157)

### Inferences
Evidence a reviewer would likely expect:
1. **Two complementary definitions, both reported.**
   - (a) *Neighbour-based*: the probability that a document in the exact top-10 is missing from the HNSW top-10, by arrival cohort. This is model-internal, needs no qrels, and isolates the index.
   - (b) *Relevance-based*: among qrel-relevant documents in the exact top-10, the HNSW drop rate by cohort, plus the nDCG and recall impact per cohort.
   - (a) shows the mechanism; (b) shows it matters to users. Reviewers at IR venues will discount (a) alone.
2. **Judgment-bias controls.**
   - DCTR qrels are built from CORE's historical SERPs and clicks, and new documents have had less time and exposure to collect clicks. Relevant new documents are therefore likely under-judged, which biases the relevance-based measure in an unknown direction.
   - Mitigations: report on raw-click qrels, DCTR qrels and LLM labels; report judged@10 by cohort; use condensed-list evaluation, which removes unjudged documents.
   - The neighbour-based measure avoids this problem altogether, which is a key argument for reporting it.
3. **Confound controls.**
   - Compare HNSW against exact search with the same embeddings, so model-side priors cancel.
   - Stratify by topic or query and by local embedding density or hubness.
   - Control for document length and whether full text is available.
   - Run a *shuffled-insertion rebuild*: build the same final corpus from scratch in random order. If the cohort gap disappears, insertion order or time is causal; if it persists, the cause is where new documents sit in embedding space.
4. **Statistics.**
   - Run multiple HNSW builds with different seeds and parameter settings (M, efConstruction, efSearch), and give confidence intervals on the cohort ratio.
   - Use per-query paired tests and a dose-response check (does the gap shrink as efSearch grows?).
   - Repeat with IVF and PQ to show generality, or state the limitation.
5. **Retrievability-style reporting.**
   - Gini or Lorenz of r(d) under exact versus HNSW, by cohort, using a large synthetic query set (docT5query-style, following the CIKM 2023 paper). This connects to the retrievability community, beyond the roughly 100–150 judged queries per snapshot.
   - The small evaluable query sets (149, 100 and 136 queries, only 12 shared across snapshots) will draw criticism on statistical power. Synthetic queries are the standard answer.
6. **Practical significance.** Average nDCG@10 changes below 0.01 are consistent with Leto et al.'s "minor average impact". The paper should argue explicitly that per-cohort disparity matters even when the mean does not, using the exposure-fairness framing.

### Gaps
- I did not verify the specific literature on condensed lists (Sakai, SIGIR 2007), judged@k, or click-model debiasing (e.g. Joachims et al.) in this session. Verify those citations before relying on them.
- **NOT FOUND:** published CLEF/ECIR reviewer guidelines specific to bias claims. The expectations above are inferences from the cited methodological literature.
