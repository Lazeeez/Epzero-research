# ANN search error vs downstream retrieval effectiveness, and threats to a LongEval index-effect study

Scope: (1) what is known about ANN recall@k (vs exact) and end-task effectiveness (nDCG/MRR/recall, RAG answer quality); (2) methodological threats to decomposing LongEval effectiveness change into a model/data/query effect (exact search) and an index effect (exact minus approximate). Research date 2026-11-12. Sources marked "(not fetched this session)" are canonical references cited from their known identifiers; their specific content was not re-verified here.

## Q1. Exact vs approximate search on IR benchmarks: reported effectiveness differences

### Takeaway
Direct, controlled exact-vs-HNSW comparisons with numbers are rare. The best one (Lin 2024/ACL Industry 2025, Lucene, BGE-base on BEIR) finds negligible nDCG@10 loss on corpora under 100K docs, "mostly small" loss at 100K-1M, and a measurable loss above 1M (e.g. BioASQ, 14.9M docs: about -0.015 nDCG@10 averaged over 5 HNSW builds, range -0.011 to -0.020; with int8 quantization about -0.017, max -0.024, roughly 4-5.8% relative). The loss grows with corpus size, which matters because LongEval snapshots are millions of documents.

### Cited Findings
- Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?" (arXiv 2409.06464, Sep 2024; ACL 2025 Industry track): treats flat indexes as exact ground truth. Compared with flat, HNSW differences are negligible on small corpora (<100K), "mostly small" for 100K-1M, and more pronounced as size grows past 1M. — [arXiv HTML](https://arxiv.org/html/2409.06464v1); [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-industry.61)
- Same paper: on BioASQ (14.9M docs), HNSW scores about 0.015 nDCG@10 below flat on average over five indexing trials (min 0.011, max 0.020). int8-quantized HNSW is about 0.017 lower on average (max 0.024), about 4-5.8% relative. — [arXiv HTML](https://arxiv.org/html/2409.06464v1)
- Same paper, setup: Lucene 9.9.1, M1 Ultra Mac Studio (20 cores, 128 GB), 16 threads, HNSW M=16, efC=100, efSearch=1000. On FEVER (5.4M docs), flat search runs at 7.3 QPS vs 63 QPS for HNSW (cached); indexing takes 4.2 min for flat vs 35 min for HNSW. Quote: "both HNSW indexing and quantization are non-deterministic and typically degrade retrieval effectiveness." — [arXiv HTML](https://arxiv.org/html/2409.06464v1)
- Lin, Pradeep, Teofili, Xian, "Vector Search with OpenAI Embeddings: Lucene Is All You Need" (arXiv 2308.14963; WSDM 2024): OpenAI ada-002 with Lucene HNSW (M=16, efC=100, efSearch=1000) gets MRR@10 0.343 and R@1k 0.984 on MS MARCO dev, and nDCG@10 0.704 on DL19 and 0.676 on DL20, at 9.8 QPS for 1000 hits. The paper reports **no flat baseline**, so it cannot isolate the HNSW loss. It notes that "due to the non-deterministic nature of HNSW indexing, effectiveness figures may vary slightly from run to run." — [arXiv](https://arxiv.org/abs/2308.14963); [ar5iv](https://ar5iv.labs.arxiv.org/html/2308.14963)
- Anserini/Pyserini publish separate regression pages for flat and HNSW dense indexes (e.g. BGE-base and cosDPR-distil on MS MARCO v1). I could not retrieve a specific page in this session (the URL I tried returned 404), so this section gives no Anserini flat-vs-HNSW numbers. — [Anserini repo](https://github.com/castorini/anserini)

### Inferences
- An HNSW loss of 0.01-0.02 nDCG@10 at multi-million scale is the same size as many year-over-year LongEval changes and many reported system differences. So the index effect is not negligible a priori for 1-2M+ doc snapshots, and it is a real quantity to measure.
- The HNSW loss itself varies between builds (0.011-0.020 across 5 builds on one dataset). The index effect therefore needs repeated builds (several seeds or insertion orders) per snapshot, not a single build.

### Gaps
- No verified, citable flat-vs-HNSW MRR@10 on MS MARCO dev from the Anserini regression pages (not retrieved).
- I found no published flat-vs-IVF/PQ nDCG comparison on BEIR/TREC DL in this session. FAISS-style IVF-PQ effects on nDCG remain unverified here.
- I found no study that measures ANN loss on French web data or on LongEval specifically (the LongEval 2025 overview mentions no ANN indexes; see Q3).

## Q1b. RAG / end-task studies: ANN recall vs answer quality, and "which recall do you need"

### Takeaway
Two studies agree that lowering ANN recall (vs exact) costs RAG answer quality only slightly: recall@10 of 0.7 costs about 2-3% gold-document recall, and recall down to 0.4 moves BERTScore by at most about 1.2%. Recall-vs-exact is a poor predictor of downstream quality; a distance-ratio measure predicts it far better.

### Cited Findings
- Leto, Aguerrebere, Bhati, Willke, Tepper, Vo, "Toward Optimal Search and Retrieval for RAG" (arXiv 2411.07396; NeurIPS 2024 ATTRIB workshop): graph ANN (Intel SVS), search recall@10 targets of 0.95, 0.9 and 0.7 vs exhaustive search. Datasets: ASQA (948 queries) and QAMPARI (1,000) over 21M Wikipedia passages; NQ (2,837) over 112M. Retrievers BGE-base and ColBERTv2; readers LLaMA-2-7B-chat and Mistral-7B-Instruct-v0.3. Quote: "Setting the search recall@10 to 0.7 only results in a 2-3% drop in gold document recall with respect to using exhaustive search." Gold-document recall mattered far more than search recall. — [arXiv HTML](https://arxiv.org/html/2411.07396v1); [NeurIPS page](https://neurips.cc/virtual/2024/105295)
- Dimitropoulos & Mamoulis, "ANN Search: Recall What Matters" (arXiv 2606.04522, Jun 2026): as Recall@k falls from 1.0 to 0.4, BERTScore F1 changes by at most 1.2% on HotpotQA and under 1% on MS MARCO and PubMedQA; LLM-graded scores change by 0.07 on a 10-point scale (PubMedQA). Recall@k predicts true downstream quality with about 30% mean absolute deviation; inverse approximation ratio (1/Ratio@k) has 0.5-2.6% MAD, 11-57x better. Meeting a Recall target (T=0.95, k=100) costs 1.86-9.36x more distance computations on average than meeting a 1/Ratio target, up to 33.3x for HNSW on ImageNet. — [arXiv HTML](https://arxiv.org/html/2606.04522)
- Practitioner reports (not peer reviewed) say that with fixed HNSW settings, recall falls faster than exact search as the corpus grows. — [Towards Data Science](https://towardsdatascience.com/hnsw-at-scale-why-your-rag-system-gets-worse-as-the-vector-database-grows/)

### Inferences
- ANN "misses" are often near-ties in embedding distance, so they swap in documents nearly as good. This explains why answer and nDCG metrics move much less than recall-vs-exact. For the proposed study, recall@k vs exact will exaggerate the index effect relative to nDCG. Report both, plus a distance-ratio measure.
- The RAG findings are on Wikipedia QA with graded or LLM metrics. They may not transfer to web search with sparse click qrels, where a single swapped top document can flip nDCG@10 for a query.

### Gaps
- I found no Pinecone or Weaviate study with controlled numbers in this session; only vendor/FAQ pages turned up (e.g. [Zilliz FAQ](https://zilliz.com/ai-faq/in-comparing-two-vector-stores-or-ann-algorithms-for-use-in-rag-what-performance-and-accuracy-metrics-should-be-part-of-the-evaluation-to-make-an-informed-choice.md)), and they are not evidence.
- I could not locate a paper titled exactly "The Impact of Approximate Search on RAG". The Leto et al. paper is the closest match.

## Q2. Are ANN errors non-random (clustered on hard/OOD queries)? "Search error" in dense retrieval

### Takeaway
Yes, for graph indexes. The ANN literature documents a sharp accuracy drop on "hard" queries, especially queries far from the base data (out-of-distribution). Per-query hardness is predictable from graph structure (Steiner-hardness) better than from LID. Nobody has shown whether these hard queries line up with IR-hard queries (low nDCG). That alignment is the mechanism by which the index effect could bias evaluation rather than just add noise.

### Cited Findings
- Wang et al., "Steiner-Hardness: A Query Hardness Measure for Graph-Based ANN Indexes" (PVLDB 17, 2024): graph methods "often suffer from a significant drop in accuracy for certain queries"; queries farther from the base data (OOD) tend to have lower accuracy. Distance-based measures such as LID ignore graph structure. Steiner-hardness correlates much better with actual query effort. — [PVLDB PDF](https://vldb.org/pvldb/vol17/p4668-wang.pdf); [arXiv](https://arxiv.org/html/2408.13899v1)
- Dimitropoulos & Mamoulis say geometric closeness and identifier matching diverge in high dimensions. They do **not** claim that misses cluster on IR-hard queries. — [arXiv HTML](https://arxiv.org/html/2606.04522)

### Inferences
- In LongEval, queries and documents drift over time while the embedding model is fixed. Later-snapshot queries may become more OOD relative to the corpus or the model, so ANN recall may fall over time even with fixed index parameters. The index effect could then be **time-correlated**, which is exactly what the decomposition wants to detect, but it also means efSearch tuned on early snapshots is not "fixed quality" later.
- Text queries are asymmetric with passages (short query vs long doc embeddings), which is a mild form of OOD. Expect per-query recall to be heavy-tailed. Report the per-query distribution, not just the mean.

### Gaps
- I found no paper that formally defines and measures "search error" in dense retrieval, in the MT sense of the model scoring a missed item higher than what the search returned. Treat this as absence of evidence, not evidence of absence.
- I found no study that correlates per-query ANN recall with per-query nDCG or query difficulty on IR benchmarks.

## Q3. Methodological threats

### Takeaway
The main threats: (a) LongEval click-model qrels are sparse and noisy, and typical topic-set sizes give very low power for ~0.01 nDCG effects, though LongEval's large query counts help if all queries carry judgments; (b) exact search over 1-2M+ docs per snapshot is feasible, but slow on CPU; (c) re-encoding, overlap and deduplication confound "data" with "index" effects; (d) ANN retrieves documents outside the judgment pool, so the index effect partly measures how unjudged documents are treated.

### Cited Findings
**(a) Qrels noise and statistical power**
- LongEval 2025 WebRetrieval: training set of 19M documents and 119,341 queries (Jun 2022-Feb 2023); test set of 14M documents and 63,416 queries over 7 monthly snapshots (Mar-Aug 2023). Qrels come from a simplified Dynamic Bayesian Network (sDBN) click model over Qwant clicks. The overview gives no thresholds or relevant-per-query statistics. Overlap "is highest between consecutive months and decreases over time." Metrics: nDCG, nDCG@10, Relative Improvement, Delta Relative Improvement, Effect Ratio, standardized nDCG. Participants used dense retrievers (E5-large-v2, MiniLM-L6-v2); no ANN index is mentioned. — [LongEval 2025 overview, arXiv 2509.17469](https://arxiv.org/html/2509.17469v1)
- LongEval click-model qrels are released as a dataset. The click model is described as reducing position-bias noise from raw clicks and protecting privacy, serving as a "soft relevance estimate." One earlier LongEval held-out set had 1,420 judgments, of which 74 were highly relevant and 326 relevant. — [LINDAT dataset](https://lindat.mff.cuni.cz/repository/items/f0aaece7-0434-4d0d-aa20-9b14a51e0a46); [search snippet of LongEval 2023 material](https://ar5iv.labs.arxiv.org/html/2308.10549)
- Urbano, Lima, Hanjalic, "Statistical Significance Testing in IR: An Empirical Analysis of Type I, Type II and Type III Errors" (SIGIR 2019, DOI 10.1145/3331184.3331259): simulated n=25, 50 and 100 topics. Recommends the t-test (top choice) and the permutation test, both of which hold the nominal α. Recommends dropping Wilcoxon, sign and bootstrap-shift. According to a summarizer reading of the paper (exact figures to be checked in its tables), power at 50 topics and α=0.05 for δ=0.01 is only about 10-15% for nDCG@20. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.11096); [Webis anthology](https://ir.webis.de/anthology/2019.sigirconf_conference-2019.53)
- Sakai's topic-set-size design (paired t-test, ANOVA and CI-based) shows that typical IR topic sets are smaller than statistical requirements imply for small differences. — [Sakai, IRJ 2016](https://link.springer.com/content/pdf/10.1007/s10791-015-9273-z.pdf); [tutorial slides](https://www.slideshare.net/slideshow/ictir2016tutorial-65845256/65845256)

**(b) Feasibility of exact search**
- Lucene flat search on 5.4M docs (FEVER) runs at 7.3 QPS on a 20-core M1 Ultra. At that rate, about 60K test queries take roughly 2.3 h per snapshot per configuration (my arithmetic from the reported QPS). — [Lin 2024](https://arxiv.org/html/2409.06464v1)
- GPU brute-force k-NN is the standard route for exact ground truth at scale (Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs"). — [arXiv 1702.08734](https://arxiv.org/abs/1702.08734) (not fetched this session; no throughput figure verified)

**(c) Confounds (re-encoding, deduplication, overlap)**
- Snapshot document overlap decays with time distance (highest between consecutive months). Changes in the index effect therefore mix index aging (insertions/deletions) with corpus change. — [LongEval 2025 overview](https://arxiv.org/html/2509.17469v1)
- HNSW builds are non-deterministic, so the same corpus and model give different nDCG across builds (range 0.011-0.020 loss on BioASQ). — [Lin 2024](https://arxiv.org/html/2409.06464v1)

**(d) Unjudged documents**
- Fröbe, Gienapp, Potthast, Hagen, "Bootstrapped nDCG Estimation in the Presence of Unjudged Documents" (ECIR 2023): condensed lists "often overestimate the effectiveness of a system"; naive bounds are "very loose," especially for nDCG when top-ranked documents are unjudged; trec_eval's default (unjudged = non-relevant) "favors systems that retrieve many (relevant) judged documents." Condensed lists were preferred over bpref in earlier evidence (Sakai). The paper proposes bootstrapped nDCG with run/pool priors, which is more accurate than condensed lists. Dense retrievers (ANCE on TREC-COVID) retrieved many unjudged but relevant documents. — [Webis PDF](https://downloads.webis.de/publications/papers/froebe_2023a.pdf)
- BEIR: dense retrievers have high Hole@10 (ANCE 14.4%, TAS-B 31.8%), i.e. a large share of their top-10 hits were never judged. The lexical pool bias penalizes dense methods. — [BEIR, arXiv 2104.08663](https://arxiv.org/pdf/2104.08663) (figures from search snippet)

### Inferences
- **Power**: if the index effect is about 0.005-0.02 nDCG@10, detecting it per snapshot needs thousands of judged queries. LongEval's tens of thousands of queries may suffice, but only for queries with at least one click-relevant document. Use paired tests (t-test or permutation, per Urbano) on per-query differences between exact and ANN. The pairing removes most qrel noise because both runs are judged by the same qrels.
- **Qrel bias specific to the index effect**: click qrels come from Qwant's production ranker. Exact and ANN runs differ mostly on near-tie documents that are less likely to have been shown or clicked. They tend to be unjudged and so count as non-relevant, which pushes the measured index effect toward zero (or makes its sign arbitrary). Report judged@10 for both runs. Compute the effect under the non-relevant, condensed and bootstrapped treatments as a sensitivity analysis.
- **Recall-against-exact as an intrinsic metric**: it needs no qrels, it is immune to (a) and (d), and it is standard in ANN benchmarking ([ann-benchmarks, arXiv 1807.05614](https://arxiv.org/abs/1807.05614), not fetched). IR precedents include rank-safety in dynamic pruning and rank-biased overlap ([Webber, Moffat, Zobel, TOIS 2010](https://doi.org/10.1145/1852102.1852106), not fetched). But Dimitropoulos & Mamoulis show it over-predicts quality loss by about 30%, so pair it with a distance ratio or score-gap measure and with qrel-based deltas.
- **Re-encoding confound**: if the embedding model changes across snapshots, "exact" effectiveness folds in the model change. Fix the model for the decomposition and treat model updates as a separate arm. For deduplication, near-duplicates create ties that inflate recall-vs-exact misses with no effectiveness change. Dedupe, or count a hit as correct if it is a near-duplicate of an exact neighbour.
- **Non-additivity**: ANN loss depends on the model and corpus (scale, OOD). The "index effect" is therefore an interaction term, not a separable additive component. State this explicitly, and possibly report it as conditional on the snapshot.

### Gaps
- Reliability or agreement of LongEval sDBN qrels vs human or LLM judgments: no quantitative figures found this session. LongEval LLM-judgment use is also unverified.
- No verified GPU flat-search throughput or cost figures for the 1-20M x 768-1024-dim scale.
- No study applies bootstrapped or condensed nDCG specifically to exact-vs-ANN comparisons. This would be novel.
