# ANN-minus-exact effectiveness gap in dense retrieval: static, recall-to-effectiveness link, and across time / collection growth

Scope: prior work (2018–2026) most likely to make a reviewer say "this is already known" about the paper's claims (1) and (4). About 15 searches and fetches. Quotes marked "verbatim" were checked against the primary page (arXiv abstract, HTML or PDF text). Items marked UNVERIFIED were seen only in search snippets or secondary summaries.

## (a) Which papers measure ANN vs exact (flat) effectiveness on judged IR collections, and what numbers do they report?

### Takeaway
The static gap is well documented and small. HNSW against flat costs roughly 0.000–0.015 nDCG@10 on BEIR and about 0.001 MRR@10 on MS MARCO at efSearch=1000. Lin (ACL 2025 Industry) already states that "retrieval quality degrades more as corpus size increases", but that is across *different* corpora, not snapshots of one growing collection. A reviewer will see the static gap as known. The temporal/snapshot framing is the novel part.

### Cited Findings
- **Lin, "Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes?"**, arXiv 2409.06464 (Sep 2024), published at ACL 2025 Industry Track. Setup: BGE on BEIR, Lucene HNSW M=16, efC=100, efSearch=1000, five trials.
  - Verbatim: "Since HNSW indexes enable fast approximate nearest-neighbor search, there is typically some effectiveness degradation, i.e., scores from HNSW indexes are usually lower."
  - Verbatim: "There are also some effectiveness differences, but they are mostly small."
  - Verbatim: "Also, we observe that retrieval quality degrades more as corpus size increases." **This is the closest prior statement to claim (4)'s "steepens as the collection grows".**
  - Table 2 (avg Δ nDCG@10, HNSW minus flat): BioASQ 0.015 (min 0.011, max 0.020); Signal-1M 0.010; HotpotQA 0.010; TREC-COVID, Touché, Quora, Climate-FEVER 0.000; TREC-NEWS min −0.004 (HNSW sometimes better). By my own arithmetic the mean of the 12 avg Δ values is about 0.004 (not stated in the paper).
  - With HNSW plus int8 quantization on BioASQ (verbatim): "nDCG@10 scores are 0.017 lower (averaged across five trials), with a max difference of 0.024."
  - Verbatim: "for corpora with fewer than 1M documents, we do not see a compelling advantage to using HNSW indexes" (this is about speed).
  - The paper also notes that HNSW construction is non-deterministic. This is relevant to the paper's ±0.01 claim, since run-to-run variance (max−min up to about 0.009 on BioASQ) is the same order as the claimed temporal change.
  - [arXiv abs](https://arxiv.org/abs/2409.06464) · [HTML](https://arxiv.org/html/2409.06464) · [ACL 2025 Industry PDF](https://aclanthology.org/2025.acl-industry.61.pdf)
- **Xian, Teofili, Pradeep, Lin, "Anserini Gets Dense Retrieval: Integration of Lucene's HNSW Indexes"**, arXiv 2304.12139 (2023; I did not verify the peer-reviewed venue). Model cosDPR-distil, M=16, efC=100.
  - MS MARCO passage dev (8.8M passages): Flat MRR@10 0.390 / R@1k 0.980, HNSW 0.389 / 0.974 (Table 1). Faiss FlatIP is 0.3896. Lucene HNSW at efSearch 10/100/1000 gives MRR@10 0.2551/0.3612/0.3842. Faiss HNSW at the same settings gives 0.1906/0.3539/0.3806 (Table 4). **This is an efSearch sweep against effectiveness: the gap is large at low efSearch and shrinks to about 0.005–0.009 MRR@10 at efSearch=1000.**
  - BEIR average nDCG@10: Flat 0.417, HNSW 0.415 (Table 2).
  - Verbatim: "compared to Faiss FlatIP (brute-force nearest neighbor search), we see that the HNSW implementation successfully retains retrieval effectiveness while improving retrieval efficiency."
  - Verbatim: "as the efSearch parameter increases, the MRR@10 metric also increases (slightly), reaching a peak of 0.3842 at efSearch=1000, which is very close to exact nearest neighbor search."
  - [arXiv PDF](https://arxiv.org/pdf/2304.12139)
- **Macdonald and Tonellotto, "On Approximate Nearest Neighbour Selection for Multi-Stage Dense Retrieval"**, CIKM 2021, pp. 3318–3322, DOI 10.1145/3459637.3482156. ColBERT on MS MARCO. Using approximate scores to trim candidates to 200 caused no statistically significant effectiveness loss and gave about a 2x speedup. Approximate scores alone are too inaccurate for high-precision metrics (from the summary of the abstract and full text; I did not re-quote it verbatim). [arXiv](https://arxiv.org/abs/2108.11480v1) · [Glasgow eprint](https://eprints.gla.ac.uk/249263)
- **Tu, Yang, Fu, Xie, Tan, Xiong, Li, Lin, "Approximate Nearest Neighbor Search and Lightweight Dense Vector Reranking in Multi-Stage Retrieval Architectures"**, ICTIR 2020, pp. 97–100, DOI 10.1145/3409256.3409818. ANN for candidate generation plus dense reranking. I could not verify its numbers. [PDF](https://cs.uwaterloo.ca/~jimmylin/publications/Tu_etal_ICTIR2020.pdf)
- **Hofstätter, Craswell, Mitra, Zamani, Hanbury, "Are We There Yet? A Decision Framework for Replacing Term Based Retrieval with Dense Retrieval Systems"**, arXiv 2206.12993 (2022). UNVERIFIED (search snippet only, the abstract does not say this): it compares exhaustive search with HNSW (128 neighbours) and IVF on TREC-DL using nDCG@10, finds approximate dense retrieval still beats term-based retrieval in most cases, and finds HNSW costs much more than IVF in indexing time and storage. [arXiv](https://arxiv.org/abs/2206.12993)

### Inferences
- The paper's numbers (gap < 0.012 at about 95% recall, about 0.002–0.003 at ≥98%) fall in exactly the range Lin 2024/2025 and Anserini 2023 report. Reviewers will call the static magnitude known. Framed as confirming them on LongEval-Sci, it is a strength rather than a weakness.
- Lin's corpus-size remark is a cross-collection observation (BEIR datasets differ in size, domain and model fit). It is not a controlled growth study. The paper should cite it and present snapshot growth as the controlled version of that observation.
- Lin's five-trial min/max spread (up to about 0.009) means "negligible change within ±0.01" needs repeated builds or confidence intervals. Otherwise a reviewer can say the temporal change sits inside build noise.

### Gaps
- I could not extract Tu et al. (ICTIR 2020) or Hofstätter et al. numbers from the primary sources.
- I did not check Pyserini/Anserini regression pages (MS MARCO v1/v2.1 HNSW vs flat, int8). They likely hold more numbers of the same kind.
- I did not verify where the Anserini HNSW paper was published.

## (b) Is there work relating neighbour recall@k to downstream effectiveness (nDCG/MRR, RAG), including 95% vs 98% operating points?

### Takeaway
Yes, and it is recent and active. Several 2024–2026 papers show that ANN neighbour recall is a weak proxy for downstream quality, with *flat* curves: recall can fall a long way with little downstream loss. None reports nDCG@10 on TREC-style judged qrels as a function of neighbour recall at the 95%/98% operating points. This is the gap claim (4) can fill, but "the gap is a function of recall" is not new in itself.

### Cited Findings
- **Leto, Aguerrebere, Bhati, Willke, Tepper, Vo, "Toward Optimal Search and Retrieval for RAG"**, NeurIPS 2024 Workshop ATTRIB, arXiv 2411.07396. ANN search recall@10 targets of 1.0 (exact), 0.95, 0.9 and 0.7, using Intel SVS and BGE-base.
  - Abstract (verbatim): "lowering search accuracy has minor implications for RAG performance."
  - Gold-document recall@10 (Table 3): ASQA exact 0.387 / 0.95→0.377 / 0.9→0.363 / 0.7→0.361; NQ exact 0.278 / 0.95→0.274 / 0.9→0.264 / 0.7→0.245. **This is effectively the "relevant-document recall lost at 95% neighbour recall" measurement, done on QA data: about 0.004–0.010 absolute at 0.95.**
  - Verbatim: "Setting the search recall@10 to 0.7 only results in a 2-3% drop in gold document recall with respect to using exhaustive search."
  - Verbatim: "Manipulating ANN search recall only results in minor decrements in QA performance."
  - Verbatim: "gold document recall ... is a far bigger factor for QA performance than search recall."
  - Citation recall falls more steeply: NQ/Mistral exact 75.0, then 55.3 at 0.95, 52.4 at 0.9, 49.9 at 0.7 (Table 4).
  - [arXiv abs](https://arxiv.org/abs/2411.07396) · [HTML](https://arxiv.org/html/2411.07396)
- **Dimitropoulos and Mamoulis, "ANN Search: Recall What Matters"**, arXiv 2606.04522 (June 2026, preprint).
  - Verbatim: "Low Recall values have little effect on downstream task quality." Also: "Recall overstates the cost of approximation."
  - RAG on HotpotQA, MS MARCO and PubMedQA: Recall drops 60% while BERTScore F1 varies ≤1.2% and the LLM grade varies <1%. On BEIR (SciFact, NFCorpus) at Recall 0.4: semantic similarity −3.5%/−5.7%, LLM grade −≤1.5%. Proposes 1/Ratio@k as a better proxy (MAD 0.5–2.6% vs about 28–30% for Recall).
  - **It does not evaluate nDCG with qrels and does not study dataset size.** Its RAG corpora are 3k–79k items.
  - [arXiv HTML](https://arxiv.org/html/2606.04522v1)
- **Kuffo, Tsakalidou, De Viti, Angel, Iša, Lenhardt, "Semantic Recall for Vector Search"**, SIGIR 2026 (per arXiv comments), arXiv 2604.20417. Proposes semantic recall (verbatim: it "does not penalize algorithms for failing to retrieve objects that are semantically irrelevant to the query") and Tolerant Recall as a proxy. It does not use nDCG or TREC qrels according to the abstract. [arXiv](https://arxiv.org/abs/2604.20417)
- UNVERIFIED (search snippet): a RAG-acceleration paper (arXiv 2412.15246) reports that lower efSearch gives higher throughput but lower generation accuracy, and that high-quality ANNS gives only a modest speedup over exact search for RAG. [arXiv](https://arxiv.org/pdf/2412.15246)
- UNVERIFIED (search snippet): arXiv 2609.36359 ("Better Nearest Neighbor Graph Indices via (Efficient) LLM-Guided Pruning") sweeps HNSW ef_search 10–50 and reports both NDCG@10 and Recall@10 on BRIGHT. This is a template for a recall-vs-nDCG curve on judged data, though static. [arXiv](https://arxiv.org/pdf/2609.36359)
- The Anserini efSearch sweep (see (a)) maps search effort to MRR@10 but does not report neighbour recall for each setting. [arXiv PDF](https://arxiv.org/pdf/2304.12139)

### Inferences
- A reviewer may cite Leto et al. 2024 to say "we know 95% search recall costs almost nothing in relevant-document recall". The paper's contribution has to be the *judged nDCG@10* curve, its *steepness near the high-recall end*, and the *change of that curve over snapshots*. Leto et al. and Dimitropoulos & Mamoulis report gentle, non-steep curves on small QA corpora. The paper's "steep function of neighbour recall" finding is a mild contrast worth stating explicitly: their recall axis spans 0.4–1.0, while the paper's spans the 0.9–1.0 region.
- Lin's ±0.009 build variance and Leto's 0.95→0.004–0.010 gold-recall drop give an external sanity range for the paper's "< 0.012 at about 95%".

### Gaps
- I found no paper reporting nDCG@10 on TREC/BEIR qrels as a continuous function of measured neighbour recall@k with explicit 95% vs 98% operating points. Absence is not proof, since the search was limited.
- I did not verify the 2412.15246 and 2609.36359 details from full text.

## (c) Has anyone measured how the ANN-minus-exact gap changes as a collection grows, or over temporal snapshots of a judged collection?

### Takeaway
I found no prior work tracking the ANN-minus-exact *effectiveness* (nDCG with qrels) gap across temporal snapshots of a judged collection, LongEval or otherwise. Three nearby literatures exist that reviewers may raise:
1. Cross-corpus "degrades more as corpus size increases" (Lin, ACL 2025).
2. ANN *neighbour recall* decaying with dataset growth or streaming updates (FreshDiskANN, Xiao et al. 2024, DCO benchmark 2026), measured without relevance judgments.
3. Exact dense retrieval effectiveness falling with index size (Reimers & Gurevych, ACL 2021), with no ANN component.

LongEval participant papers use dense runs but, as far as I could check, do not compare ANN against exact search.

### Cited Findings
- **Lin (ACL 2025 Industry)**, verbatim: "Also, we observe that retrieval quality degrades more as corpus size increases." This is across BEIR corpora, not controlled growth. [HTML](https://arxiv.org/html/2409.06464)
- **Reimers and Gurevych, "The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes"**, ACL-IJCNLP 2021 (short), pp. 605–611, arXiv 2012.14210. They show theoretically and empirically on MS MARCO that dense retrieval worsens relative to BM25 as index size grows (a "tipping point"), and that the lower the dimension, the higher the risk of retrieving irrelevant documents. **This concerns exact dense search versus sparse search, not the ANN gap**, but it is the classic "collection growth hurts dense retrieval" citation. [ACL Anthology](https://aclanthology.org/2021.acl-short.77) · [arXiv](https://arxiv.org/abs/2012.14210v2)
- **Singh et al., "FreshDiskANN"**, arXiv 2105.09613 (2021). From the search snippet: simple graph update rules degrade index quality over a stream of insertions and deletions for HNSW and NSG. Measured as neighbour recall, with no judgments. [arXiv](https://arxiv.org/pdf/2105.09613)
- **Xiao et al., "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation"**, arXiv 2407.07871 (2024). From the search snippet: after about 3000 delete/reinsert iterations, 3–4% of points become unreachable and recall drops about 3%. Neighbour recall only. [arXiv](https://arxiv.org/html/2407.07871v2)
- **DCO benchmark, arXiv 2604.02801 (2026)**, UNVERIFIED snippet: inserting GIST vectors in batches, "as the dataset grows, the recall of all methods declines due to the increased search space." This is the closest analogue of "neighbour recall vs N at fixed parameters", with no judgments. [arXiv](https://arxiv.org/pdf/2604.02801)
- UNVERIFIED snippets: DIGRA (CUHK) notes that a static index built on the original data loses significant recall when many new elements are added. Wolverine (PVLDB vol. 18) finds that HNSW with low construction parameters loses recall as updates progress. [DIGRA](https://www1.se.cuhk.edu.hk/~swang/V3mod148-DIGRA.pdf) · [Wolverine](https://www.vldb.org/pvldb/vol18/p2268-zheng.pdf)
- **"As We May Search"**, arXiv 2606.29652 (2026 preprint), UNVERIFIED snippet: on personal-scale corpora, exact dense nDCG@10 drops from 0.990 (1K docs) to 0.910 (100K) and 0.789 (500K), and HNSW at 1M loses about 2% nDCG@10 against flat. This may be the closest work to "ANN gap vs corpus size with judgments" (likely synthetic or personal collections). **It needs a full-text check before submission.** [arXiv](https://arxiv.org/pdf/2606.29652)
- **LongEval-Sci 2026 participants:** Yang and Wu, "Submitted and Diagnostic Analysis of Full-Text Temporal Retrieval for LongEval-Sci", arXiv 2607.04088 (July 2026 preprint), uses Qwen3 dense baselines over three snapshots. The abstract and landing page do not mention HNSW, IVF, FAISS, or ANN against exact. [arXiv](https://arxiv.org/abs/2607.04088)
- The LongEval CLEF 2025 overview (arXiv 2509.17469) frames the lab as measuring effectiveness change across snapshots. I saw no ANN-versus-exact analysis in what the search returned. [arXiv](https://arxiv.org/pdf/2509.17469)
- Lin's Anserini/Faiss work and the big-ANN-style benchmarks measure static corpora. The MS MARCO Web Search benchmark (arXiv 2405.07526) reports ANN recall@k against QPS (for example SPANN recall@10 80.55%, DiskANN 87.07%; snippet), not effectiveness over time. [arXiv](https://arxiv.org/pdf/2405.07526)

### Inferences
- Claim (1), "first measurement of the ANN-minus-exact nDCG@10 gap tracked across snapshots of a real relevance-judged longitudinal collection", survived this search. Nothing found does this. The phrasing should be narrow ("real", "relevance-judged", "longitudinal snapshots"), because collection-size effects on ANN recall (unjudged) and on exact dense effectiveness (Reimers & Gurevych) are both known.
- Claim (4), "steepens as the collection grows", is anticipated *qualitatively* by Lin's corpus-size remark and the ANN streaming/growth literature. The novelty is the controlled, within-collection, judged version. Snapshot growth is confounded with query and qrels drift, so reviewers may ask that size be separated from drift (for example by subsampling a later snapshot down to the earlier size).
- Highest-risk prior work to cite and differentiate: Lin 2024/ACL 2025 (static gap plus corpus-size remark), Leto et al. 2024 (search recall → gold-document recall at 0.95/0.9/0.7), Dimitropoulos & Mamoulis 2026 (recall is a poor proxy), Kuffo et al. SIGIR 2026 (semantic recall), and Reimers & Gurevych 2021 (growth hurts dense retrieval).

### Gaps
- I did not search the CEUR LongEval 2024/2025/2026 working-notes volumes exhaustively for participants who ran HNSW/IVF and reported exact baselines. Individual working notes might contain an incidental ANN-vs-flat table, so a targeted CEUR full-text check is recommended.
- I did not verify "As We May Search" (2606.29652) beyond the snippet. It is the most likely hidden overlap on "ANN nDCG gap vs corpus size".
- I did not check TREC temporal or evolving collections (TREC Fair, TREC RAG 2024/2025 with MS MARCO v2.1 HNSW vs flat in Anserini regressions) in primary sources.
