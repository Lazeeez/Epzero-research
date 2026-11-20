# Prior work on separating search/index error from model error in dense retrieval, RAG and recommendation evaluation

Search log: 16 web searches and 6 page fetches (arXiv abstract/HTML pages, Harrisburg Digital Commons) on 2026-11-12. ACM DL / Semantic Scholar were not queried directly. Several numbers below come from search-result snippets rather than full-text reads, and are marked as such. "Not found" below means absence of evidence within these searches, not proof of absence.

Proposal under test: split retrieval effectiveness (relevance-judged, e.g. nDCG) into (a) exact-search effectiveness, the model/data component, and (b) the index component, exact minus approximate, and then track (b) over time (LongEval-style snapshots, index refreshes).

## Q1. Has the effect of ANN on retrieval effectiveness and on system conclusions been measured?

### Takeaway
Yes, but only cross-sectionally. Several papers measure exact (flat) minus HNSW/IVF effectiveness on fixed collections and find it small: roughly ≤0.015 nDCG@10 on BEIR. Lin explicitly warns that HNSW nondeterminism can matter when comparing models of similar quality. No paper I found reports exact-vs-approximate as a named effectiveness decomposition, measures ranking flips (Kendall tau) between exact and approximate system rankings, or tracks the gap over time.

### Cited Findings
- **Macdonald & Tonellotto, "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval", CIKM 2021, pp. 3318–3322, doi:10.1145/3459637.3482156.** They use approximate FAISS scores to prune the ColBERT candidate set before exact MaxSim re-scoring. On MS MARCO passage, cutting the set to 200 candidates gave no statistically significant effectiveness difference with about a 2x speedup (from a search snippet). — [arXiv 2108.11480](https://arxiv.org/abs/2108.11480v1); [IR Anthology](https://ir.webis.de/anthology/2021.cikm_conference-2021.377/). **Verdict: different.** This is an efficiency study of ANN candidate selection in a multi-stage pipeline. It reports approximate versus full pipeline effectiveness on one static collection, with no decomposition framing and no temporal axis.
- **Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?", arXiv 2409.06464 (Sept 2024). An ACL Anthology preview lists it as 2025.acl-industry.61, i.e. probably ACL 2025 Industry Track (venue not confirmed from the paper itself).**
  - Flat indexes are used as the exact reference. HNSW minus flat nDCG@10 is negligible below 100K docs, usually under 0.01 for 100K–1M docs, and about 0.015 on BioASQ (15M docs). Negative deltas (HNSW better than flat) occur because of randomness.
  - "HNSW index construction is non-deterministic, and scores from each trial may differ slightly." Five trials were run, and the variability is flagged as practically relevant when comparing embedding models of similar quality.
  - int8 quantization costs about 0.001–0.01 nDCG@10.
  - The paper does not directly test whether model rankings change between flat and HNSW.
  - Sources: [arXiv HTML](https://arxiv.org/html/2409.06464v1); [ACL preview](https://preview.aclanthology.org/setup/2025.acl-industry.61).
  - **Verdict: partially overlaps (the closest static match).** It effectively measures the "exact minus approximate" quantity on a static benchmark (BEIR, 29 collections) and notes that nondeterminism can affect conclusions. It does not name or formalise a model/index decomposition, track the gap over time or across index refreshes, or test ranking flips.
- A search snippet says HNSW plus quantization degrade nDCG@10 by "4% to 5.8%" relative to exact search. The underlying source is unclear (it appeared in a result set alongside 2409.06464 and 2601.11557), so it is unverified. — [arXiv 2601.11557](https://arxiv.org/pdf/2601.11557)
- HNSW query-time nondeterminism in a RAG/agent setting: a dense+HNSW baseline was nondeterministic on 40 of 50 queries, and the worst query had only 21% average overlap across 20 repetitions (search snippet, ContextNest, arXiv 2607.02116). — [arXiv 2607.02116](https://arxiv.org/pdf/2607.02116). **Verdict: different.** It is about determinism and governance, not a decomposition of effectiveness.
- Reproducibility papers on DPR (Ma et al., ECIR 2022; Pyserini replication 2104.05740) compare reproduced dense-retrieval scores, but my searches did not surface any result where HNSW versus flat changed a published conclusion. — [Ma et al. ECIR 2022](https://cs.uwaterloo.ca/~jimmylin/publications/Ma_etal_ECIR2022.pdf); [arXiv 2104.05740](https://arxiv.org/pdf/2104.05740)

### Inferences
- The quantity "exact nDCG minus ANN nDCG" is well established as a static measurement. Lin 2024/2025 has effectively measured it on BEIR, and exact search being the upper bound is standard practice (e.g., EHI, [arXiv 2310.08891](https://arxiv.org/html/2310.08891)). The proposal cannot claim novelty for the static measurement itself. Novelty has to rest on formalising it as a decomposition, tracking it over time, and showing it affects conclusions.
- The prior static evidence suggests the index component is small, at ≤1.5 nDCG points. That is a risk for the proposal: the temporal signal may be small relative to noise. It is also an opportunity, since temporal deltas between LongEval snapshots are also often small (a few nDCG points), so an index component of ~0.01 could plausibly be comparable in magnitude. This comparison is my inference and is not measured anywhere I found.

### Gaps
- I found no study computing Kendall tau or ranking flips between exact and approximate evaluation of multiple systems (searched explicitly; nothing returned).
- I found no MTEB/BEIR documentation on whether leaderboards mandate exact search. MTEB retrieval is generally run with exact search, but I did not verify this in these searches.

## Q2. "Search error vs model error" analogues and retrieval-error decomposition in RAG and recommendation

### Takeaway
The search-error/model-error dichotomy comes from NMT (Stahlberg & Byrne 2019). RAG work decomposes errors into retriever versus generator, and some work varies ANN accuracy and reports downstream QA impact. Industry recommendation and search papers measure ANN recall against exact KNN. I found no paper that transfers the Stahlberg–Byrne framing to retrieval as an "index error versus model error" split of relevance effectiveness, and none that tracks it over time.

### Cited Findings
- **Stahlberg & Byrne, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?", EMNLP-IJCNLP 2019 (D19-1331).** They run exact inference (beam search plus DFS) to separate search errors from model errors. Beam search fails to find the global-best score for more than half of sentences even at beam 100. For more than half of sentences, the model's global best is the empty translation, so search errors mask model errors. — [ACL Anthology](https://aclanthology.org/D19-1331/). **Verdict: different domain, same methodological core.** This is the conceptual ancestor: exact search as an oracle for separating model error from search error. It has the direct parallel that search error can mask model error, which corresponds to the occasional HNSW > flat cases in Lin. The proposal should cite it as its lineage and not claim the idea is original. No retrieval transfer was found.
- **Leto et al., "Toward Optimal Search and Retrieval for RAG", arXiv 2411.07396 (Nov 2024; listed at NeurIPS 2024, apparently a workshop).**
  - Compares exact search (search recall 1.0) with ANN at 0.95, 0.9 and 0.7 recall@10, using BGE-base on ASQA and NQ.
  - Setting search recall to 0.7 costs only 2–3% gold-document recall, with minor effects on QA correctness and citation metrics.
  - Conclusion: "gold document recall … is a far bigger factor for QA performance than search recall."
  - No temporal aspect: static 2018–19 Wikipedia dumps.
  - Sources: [arXiv HTML](https://arxiv.org/html/2411.07396v1); [NeurIPS page](https://neurips.cc/virtual/2024/105295).
  - **Verdict: partially overlaps (the closest RAG match).** It holds the model fixed and varies ANN accuracy, effectively separating retriever-model quality (gold recall) from search approximation. It is static, has no formal decomposition, and has no index-refresh or time dimension.
- **Quinn et al. (incl. Zamani), "Accelerating Retrieval-Augmented Generation", arXiv 2412.15246.** Lower HNSW efSearch gives higher throughput but lower generation accuracy. They argue exact retrieval can reduce end-to-end inference time (their IKS system gives 13.4–27.9x faster exact NN over 512 GB). — [arXiv 2412.15246](https://arxiv.org/abs/2412.15246). **Verdict: different.** It is a hardware/efficiency paper that only touches the accuracy trade-off.
- **RAG-X (arXiv 2603.03541)** decouples retrieval from generation for error attribution (Accuracy 71% vs Context Hit Rate 57.6%). **RAG-E (arXiv 2601.21803)** finds generators ignore the retriever's top-ranked documents in 47.4–66.7% of queries. — [RAG-X](https://arxiv.org/html/2603.03541v1); [RAG-E](https://arxiv.org/abs/2601.21803). **Verdict: different.** Both decompose retriever versus generator and have no index (ANN) component.
- **Huang et al., "Embedding-based Retrieval in Facebook Search", KDD 2020.**
  - Measures ANN accuracy as 1-recall@10 against exact KNN.
  - Tunes IVF/PQ parameters offline, then picks the setting through online experiments.
  - Sources: [arXiv 2006.11632](https://arxiv.org/pdf/2006.11632).
  - **Verdict: partially overlaps (operational).** It separates ANN error from the model via an exact-KNN reference, but measures it in neighbour-overlap terms, not as a loss in relevance effectiveness, and does not track it over time.
- Pinterest Manas launched SQ/PQ quantization in online experiments with 20–30% serving cost savings (InfoQ, Sept 2026). — [InfoQ](https://infoq.com/news/2026/09/pinterest-search). **Verdict: different.** It is a cost/quality engineering report with no decomposition.
- Two-tower recommendation papers (e.g., Allegro RecSys 2025; the multifaceted learnable index paper arXiv 2602.16124) treat ANN as necessary at scale. The search did not surface any paper that relates ANN recall loss to engagement over time. — [Allegro](https://doi.org/10.1145/3705328.3748135); [arXiv 2602.16124](https://arxiv.org/pdf/2602.16124)

### Inferences
- The strongest framing for the proposal is "a Stahlberg–Byrne style search-error/model-error decomposition for dense retrieval, made longitudinal." The static half is anticipated in spirit by Leto et al. and Lin. The longitudinal half appears unanticipated.

### Gaps
- I did not find the ScaNN papers (Guo et al. ICML 2020) or YouTube or LinkedIn industry papers with ANN-recall-to-engagement curves. They were not specifically retrieved, so this is unverified either way.
- I found no paper studying how the index component behaves across model refreshes (re-embedding plus re-indexing) in recommendation.

## Q3. Efficiency–effectiveness studies across index versions or over time (ReNeuIR, drift)

### Takeaway
ReNeuIR (SIGIR 2022–2024 and later) focuses on benchmarking efficiency alongside effectiveness on static MS MARCO workloads. The ANN-systems literature does track index quality over time under content drift, but only as recall against exact neighbours, never as a relevance-judged effectiveness split. This is the nearest work to the temporal half of the proposal.

### Cited Findings
- **ReNeuIR 2024 (third workshop)** asks for index→retrieve→re-rank pipelines that TIREx runs on workloads sampled from ir_datasets (MS MARCO passage), measuring effectiveness and efficiency (including energy via Scaphandre). The stated goal is efficiency-evaluation standards. — [Microsoft Research page](https://www.microsoft.com/en-us/research/?p=1045908); [ReNeuIR motivation](https://reneuir.webis.de/motivation.html). **Verdict: different.** It has no exact-versus-approximate decomposition and no temporal snapshots. I did not survey the individual ReNeuIR papers or the 2025/2026 editions.
- **Baranchuk et al., "DeDrift: Robust Similarity Search under Content Drift", ICCV 2023.**
  - Shows that query and database content drift degrades the accuracy and efficiency of IVF-style indexes unless the index is frequently reconstructed.
  - Evaluates k-recall@k after each content-update step across the whole time range.
  - DeDrift almost eliminates the degradation while being up to 100x faster than a full rebuild.
  - Sources: [CVF PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Baranchuk_DEDRIFT_Robust_Similarity_Search_under_Content_Drift_ICCV_2023_paper.pdf); [arXiv 2308.02752](https://arxiv.org/pdf/2308.02752).
  - **Verdict: partially overlaps (the closest temporal match).** It tracks the index component over time, but the measure is neighbour recall rather than relevance effectiveness, there is no model/data component, and there is no IR test collection or evaluation-conclusion angle.
- **Quake (OSDI 2025, arXiv 2506.03437)** and **ACRONYM (arXiv 2606.03151)** show that ANN methods degrade over update cycles and under skewed or dynamic workloads, and need rebuilds or re-partitioning. Both use recall against exact ground truth. — [Quake](https://www.usenix.org/system/files/osdi25-mohoney.pdf); [ACRONYM](https://arxiv.org/html/2606.03151). **Verdict: different.** These are systems papers with recall-only, index-maintenance focus.
- **Nakacwa & Shaalan, "Dynamic-Query Robustness of ANN Indexes under Time-Indexed Drift", Harrisburg University, 24 Apr 2026.**
  - Uses MS MARCO passage embeddings with bge-base-en-v1.5, under HNSW and IVF-Flat.
  - Models query drift as stochastic corruption on the hypersphere tied to compute or network time.
  - Measures recall against exact NN.
  - Finds that time-coupled drift shifts the optimal ANN settings toward shallower search.
  - Source: [Digital Commons](https://digitalcommons.harrisburgu.edu/other-works/28/).
  - **Verdict: different but shares vocabulary.** Its "time" is intra-query latency, not corpus or query evolution across months. It uses no relevance judgements and no exact/model component. It is still worth citing to avoid an apparent name clash ("time-indexed drift").
- **"Embedding Surgery" (arXiv 2609.05110)** finds HNSW is robust to in-place embedding updates without changing graph connectivity, evaluated on HNSW and IVF. — [arXiv 2609.05110](https://arxiv.org/pdf/2609.05110). **Verdict: different.** It concerns index maintenance under embedding edits and is static in evaluation.

### Inferences
- Two literatures each cover one half of the proposal. ANN-systems papers track index error over time, but in recall-vs-exact units. IR papers measure exact-vs-ANN effectiveness, but statically. I found no paper joining them: none expresses the index component in relevance units, tracks it on a temporal IR test collection such as LongEval, and asks whether it changes temporal-robustness conclusions. That join appears to be the novel contribution, subject to the gaps listed here.
- A reviewer could argue that DeDrift plus Lin, taken together, make the result obvious. The proposal should pre-empt this by showing a case where the index component and the model component move differently over time, or where the index component changes a ranking or a temporal-degradation verdict.

### Gaps
- I did not examine any LongEval 2023–2026 participant papers for ANN versus exact settings. The searches did not surface any that discuss it, which is absence of evidence only.
- I did not check the ReNeuIR 2025/2026 accepted papers individually.

## Q4. How ANN recall loss translates into nDCG or answer-quality loss

### Takeaway
Recent work argues that Recall@k against exact neighbours overstates quality loss, because downstream quality stays flat as recall drops. Dimitropoulos & Mamoulis (2026) propose 1/Ratio@k as a better proxy. This supports the proposal's motivation, since neighbour recall is not effectiveness, but it is static and does not decompose effectiveness.

### Cited Findings
- **Dimitropoulos & Mamoulis, "ANN Search: Recall What Matters", arXiv 2606.04522 (3 June 2026).**
  - Proposes 1/Ratio@k (an inverse distance-approximation ratio) in place of Recall@k.
  - Downstream indicators (label precision, semantic similarity, BERTScore, LLM-graded quality) stay stable even when Recall@k drops a lot. Snippet: retrieval quality stays stable as Recall falls to 0.4.
  - 1/Ratio's prediction error stays under 3%, versus 25–30% for Recall (snippet).
  - Optimising 1/Ratio@k reaches operational quality thresholds at lower compute.
  - No temporal or index-refresh analysis.
  - Sources: [arXiv abs](https://arxiv.org/abs/2606.04522); [HTML](https://arxiv.org/html/2606.04522v1).
  - **Verdict: partially overlaps (motivation).** It decouples neighbour-recall from task quality, but evaluates image classification and RAG quality rather than relevance-judged nDCG on IR collections, and has no model/index decomposition over time.
- Leto et al.: ANN recall 0.7 costs only 2–3% gold-document recall (see Q2). — [arXiv 2411.07396](https://arxiv.org/html/2411.07396v1)
- Lin: HNSW versus flat costs ≤0.015 nDCG@10 at the 15M-doc scale (see Q1). — [arXiv 2409.06464](https://arxiv.org/html/2409.06464v1)
- "How important is Recall for Measuring Retrieval Quality?" (arXiv 2512.20854) appeared in results, but I did not read it. Its relevance (IR recall versus ANN recall) is unverified. — [arXiv 2512.20854](https://arxiv.org/pdf/2512.20854)

### Inferences
- There is convergent evidence (Lin, Leto, Dimitropoulos & Mamoulis) that moderate ANN recall loss causes small effectiveness loss on static data. The proposal's open question is whether this stays true as the corpus and queries drift, given that the graph is built on old data and DeDrift shows recall erodes. No work I found answers it in relevance units.

### Gaps
- I found no work giving an analytic mapping from ANN recall to nDCG loss as a function of relevance density. Rank-aware ANN metrics other than 1/Ratio were not surveyed in depth.
- Overall verdict on novelty: no work found subsumes the proposal. The closest partial overlaps are Lin 2024/25 (static exact-minus-HNSW nDCG, nondeterminism caveat), Leto et al. 2024 (static ANN accuracy versus RAG quality), DeDrift 2023 (temporal index degradation in recall units) and Dimitropoulos & Mamoulis 2026 (recall is not quality). The Stahlberg & Byrne 2019 framing is the methodological ancestor and must be cited. This verdict is limited by 16 searches without direct ACM DL or Semantic Scholar queries.
