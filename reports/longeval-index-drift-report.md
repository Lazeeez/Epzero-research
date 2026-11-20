# Approximate search in longitudinal IR evaluation: problem, gap, findings and next steps

**Literature review and research plan** · Lajat · November 2026

## Summary

Dense retrieval in practice runs on an approximate nearest-neighbour (ANN) index, built once and then updated as documents arrive. Longitudinal IR evaluation, CLEF LongEval in particular, measures how retrieval quality changes over time, but every dense system it has evaluated searches exactly. So nobody knows whether the lab's conclusions about temporal robustness hold for the approximate search that deployed systems actually run, or how much quality the index itself loses as a collection evolves.

Two literatures each cover half of this question and never meet:

- **Systems research** tracks index decay over time (FreshDiskANN, SPFresh, Quake, DeDrift), but only as nearest-neighbour recall, on synthetic or unjudged workloads.
- **IR research** tracks retrieval quality over time (LongEval, FreshStack), but with exact search, or with the index held fixed.

An adversarial review found no work that joins the two: measuring, on a real time-ordered collection with relevance judgements for each snapshot, how the gap between exact and approximate search changes, and what maintaining the index recovers. Every individual piece of the proposal has close precedent, so the novelty lies in the combination and must be framed that way.

The review also found one wrong premise in the original framing, several claims to soften, and real methodological risks. A small pilot on LongEval's released data would settle the central question faster than further reading.

## 1. The problem

### 1.1 How dense retrieval runs in practice

Searching millions of document embeddings exactly means comparing each query with every document. Deployed systems avoid this with ANN indexes:

- **graph indexes** such as HNSW;
- **partition indexes** such as IVF;
- **compressed vectors** (product or scalar quantization) to cut memory and time.

These return most, not all, of the true nearest neighbours. On a fixed collection, the quality lost is well measured: negligible below about 100,000 documents, and around 0.011–0.020 nDCG@10 at 15 million documents (Lin 2024).

### 1.2 Why a changing collection is different

A deployed index is built once and then updated. Several parts of it can drift away from the data they were fitted to:

- **The graph or partitions** were shaped by early documents. Deletions are known to leave graph nodes unreachable (Xiao et al. 2024), and partitions become unbalanced under real drift (DeDrift 2023).
- **Compressed vectors** are fitted to the early data's value ranges and distributions.
- **The search effort,** tuned once to reach a recall target, was measured on the original collection and queries. As the collection grows, the same setting reaches less.
- **Queries** can move away from the indexed documents, which lowers recall without anyone noticing, because recall is rarely measured in production.

### 1.3 Why it matters for longitudinal evaluation

LongEval's dense runs are exact by construction. The official 2026 baseline searches by brute force and is rebuilt for each snapshot, and earlier years used exact (flat) indexes. The lab's findings therefore describe an idealised system. Two consequences follow:

- **For the evaluation community:** it is unknown whether conclusions about which systems are temporally robust transfer to approximate search. If the index's error grows over time, a system judged robust under exact search may not be robust as deployed.
- **For practitioners:** teams watching quality drift in production cannot tell index error from model or data drift, and they face a recurring, poorly evidenced choice between rebuilding the index and maintaining it in place.

## 2. The research question and proposed contribution

**Question.** How much does an approximate index add to retrieval-quality change over time on a real evolving collection, and how much of that does each way of keeping the index current recover?

**Proposed measurement.** At each snapshot, run the same embeddings with exact search and with the approximate index. Write E for exact-search nDCG, A for approximate nDCG, and I = E − A for the index term. The change in observed quality between snapshots then splits as:

> A₀ − Aₜ = (E₀ − Eₜ) + (Iₜ − I₀)

The first term is what exact search also shows (the model, documents and queries). The second is how the index term changed. This is an accounting identity, not a causal split; section 5 explains why that distinction matters.

**Proposed comparisons.**
- Rebuilding the index at each snapshot.
- Updating it incrementally, without maintenance.
- Updating it incrementally, with targeted maintenance: recalibrating the search effort, refitting compressed vectors, repairing the graph.
- Freezing the first snapshot's settings.

All are judged by nDCG on relevance judgements, alongside recall against exact search, and by compute and memory cost.

**Proposed signal.** A query-drift indicator that needs no relevance judgements: how far new queries sit from their nearest documents, compared with how far documents sit from each other.

## 3. Findings: what already exists

Two review rounds were run: a general review, then an adversarial one aimed at disproving the gap. They covered LongEval 2023–2026, streaming ANN systems, quantization under drift, out-of-distribution queries, the effect of ANN on retrieval quality, evolving-corpus retrieval, the search-error-vs-model-error literature, and industry practice.

### 3.1 LongEval has never examined the index

- **A full-text search of every 2025 and 2026 LongEval working-notes paper** (CEUR Vol-4038 and Vol-4283) found no ANN first stage, no reported index configuration, no comparison of approximate with exact search, and no attribution of drift to the index.
- **The 2026 overview** attributes declining effectiveness mostly to collection growth, and finds that robustness does not depend on the approach class.
- **Two 2026 teams came close but measured nothing:** Lexora listed online dense index updates as something it could not assess; GLHF proposed a metadata-based refresh schedule and explicitly disclaimed any claim about effectiveness.
- **DS@GT's 2025 paper** suggested analysing "the semantic k-NN graph of a dense embedding model … in the context of the temporal evolution of systems". This proposal follows directly from that suggestion.

### 3.2 Closest prior work

No work was found that subsumes the proposal. These are the closest, and each must be cited with the line that separates it:

| Work | What it does | How the proposal differs |
|---|---|---|
| **PLAID SHIRTTT** (Lawrie et al., SIGIR 2024) | Streams date-ordered documents into a ColBERT-PLAID index. Index clusters fitted on the first 5% of documents drop nDCG@20 from 0.440 to 0.047 | Reference is another approximate index, not exact search; one fixed set of judgements, not per-snapshot; no maintenance options. **The strongest threat** |
| **MURR** (Yang et al., 2025) | Keeps an updated encoder compatible with already-indexed vectors over simulated document and query streams; PQ index; Success@5 | Simulated streams; studies model updates, not index error; no exact reference |
| **Embedding Surgery** (Amendola et al., 2026) | Overwrites document embeddings in place in HNSW/IVF; reports no consistent loss in effectiveness on TREC DL | Targeted edits on a static collection, not corpus evolution. Read from abstract only |
| **AURORA** (Saravanan et al., Findings of ACL 2026) | Refits a quantizer in place under drift, keeping the HNSW graph; +26.9% Recall@10 on new topics | Synthetic drift only; no exact reference; names time-stamped corpora as future work |
| **Quake** (Mohoney et al., OSDI 2025) | Maintains an IVF index over 103 real monthly Wikipedia updates (1.6M→12M vectors) | Measures recall against exact search; no relevance judgements; no split of quality change |
| **DeDrift** (Baranchuk et al., ICCV 2023) | Real temporal drift: partition indexes lose 5–11 recall points, compressed codes under 1%; a cheap refresh recovers it | Recall only; no IR collection |
| **When to Repair** (Mandarapu and Kunkunuru, 2026) | At equal repair budget, repair triggered by a drift signal does not beat a fixed schedule | Synthetic churn, recall only. **Sets the bar** for any maintenance rule |
| **Lin 2024**; Tu et al. 2020; Leto et al. 2024 | Measure exact-vs-HNSW quality loss | Static collections, no time axis |
| **Still Fresh?** (Kuissi et al., 2026) | Compares two real FreshStack snapshots across dense models | No index term, no maintenance |
| **Keller et al., ICTIR 2024** | Splits temporal change by test-collection component (documents, topics, judgements) | Holds the system fixed. The proposal **extends** it with a system-internal index term |
| **Stahlberg and Byrne, EMNLP 2019** | Uses exact search to separate search error from model error in machine translation | The method's lineage; never transferred to retrieval over time |

### 3.3 Verdict on each part

| Part | Verdict | What remains open |
|---|---|---|
| Exact-minus-approximate decomposition | **Real but narrower.** Measured statically, and over time only in recall | Tracking it in nDCG across real snapshots with their own judgements |
| Rebuild vs incremental vs in-place maintenance | **Real but narrower.** Heavily studied in recall; four papers use effectiveness, none on real snapshots against exact search | The comparison on a real evolving IR collection, judged by nDCG |
| Query-drift signal | **Not new as a statistic.** Nearest-neighbour distance is a standard out-of-distribution score (Sun et al. 2022; RoarGraph 2024; DARTH 2025) | Whether it predicts the index's share of quality loss over time, controlled for corpus size |
| Rebuild-vs-maintain guidance | **Real but narrow;** depends on the maintenance comparison | A rule that beats a fixed schedule at equal cost, judged by retrieval quality |
| **The combination** | **Real.** Nothing found does it | The whole study |

**Narrowest defensible claim:** the first measurement, on a real time-ordered IR test collection with relevance judgements for each snapshot, of how the exact-minus-approximate quality gap changes across snapshots, and of how much of it a rebuild, incremental insertion and specific maintenance actions each recover, reported in nDCG alongside recall against exact search.

### 3.4 Claims the literature contradicts

| Original claim | What the evidence shows | Revised framing |
|---|---|---|
| Evaluations treat the approximation as exact, so index error is booked as model drift | LongEval's dense runs *are* exact, so nothing has been misattributed | LongEval measures an idealised exact system; what approximate search adds over time is unmeasured |
| Little evidence on when to rebuild | Well studied in recall terms (FreshDiskANN, SPFresh, Quake); production systems compact at 20% deletions | Studied on synthetic workloads in recall terms, not on real evolving IR collections with judgements |
| Compressed vectors lose precision as data drifts | Product quantization loses under 1% under real drift (DeDrift); retraining showed no gain in a 2026 preprint | Narrow to scalar quantization's fitted value ranges, which nobody has measured |
| A well-configured index loses very little | True below 1M documents; 0.011–0.020 nDCG@10 at 15M | Losses are measurable at LongEval's scale |
| The drift signal is a contribution | The statistic is established | Test it as a validated, size-controlled predictor |
| Recall against exact search isolates the index's effect cleanly | It over-predicts downstream loss, because most misses are near-ties (Dimitropoulos and Mamoulis 2026) | Report recall, a distance-ratio measure and nDCG together |

### 3.5 Does drift over time even happen?

- **Retrieval quality drifting:** yes, documented across LongEval editions.
- **Data distributions drifting:** yes, measured on real time-stamped data (DeDrift).
- **Index error growing under heavy deletion churn:** yes. HNSW recall fell from about 98% to about 85% over 20 rounds of replacing 5% of SIFT1M (FreshDiskANN, a figure reading), and unreachable nodes that higher search effort cannot recover appear (Xiao et al.).
- **Index error growing on a collection that only grows:** **not established.** Recall at fixed effort falls with corpus size, but that is a size effect, and no study isolates drift beyond it for HNSW. **This is the crux of the proposal and its biggest risk.**

## 4. Shortcomings of this review

The verdict is "no counter-example found", not proof of novelty.

- **Structured search was largely blocked.** Of 81 planned database queries, about 16 ran: Semantic Scholar was rate-limited on 21 of 25, and the arXiv and DBLP APIs were fully blocked. Google Scholar was unreachable. About 100 general web searches and 70 page reads made up for part of this.
- **No forward-citation crawl.** Papers citing DeDrift, PLAID SHIRTTT, Keller et al. 2024, Lin 2024, Quake or MURR were never listed. A citing paper that measures nDCG over time under ANN maintenance is the most likely way the gap is already filled.
- **Some close work was read only in abstract:** BANCO (WWW 2026, blocked), Embedding Surgery, GradNormIR, L²R.
- **Not covered:** ReNeuIR workshop papers, SIGIR 2025–2026 short papers, the Big-ANN 2023 streaming metrics, thesis repositories, and several vendors' documentation.
- **Some figures are approximate,** read from plots or summaries: FreshDiskANN's recall curves, a statistical-power figure from Urbano et al., and insertion-order effects from Elliott and Clark. They must be checked against the PDFs before being quoted.
- **One automated summary invented content:** it reported FAISS and HNSW in the 2024 LongEval overview, which a full-text search disproved. Every "LongEval never did X" claim here rests on direct text search for that reason.
- **Unverified data facts:** the 2026 per-snapshot query counts, and whether the CORE collection ever removes documents.

## 5. Methodological risks in the proposed study

1. **The decomposition is accounting, not causation.** The index term interacts with corpus size and query distribution, so it should be presented as a component of observed change, not as "the index's fault".
2. **Corpus growth confounds the index term.** A rebuilt index loses recall at fixed effort simply because the collection is bigger. **Mitigation:** a size-matched control, rebuilding on a sample of a later snapshot cut to the first snapshot's size.
3. **HNSW is always built by incremental insertion.** For a collection that only grows, "rebuild vs incremental" reduces to insertion order. The documented damage comes from deletions. **Mitigation:** separate insertion order, deletions and distribution shift as factors, and check whether CORE removes documents. If it doesn't, add LongEval 2025 Web, which does.
4. **Unjudged documents bias the effect toward zero.** Exact and approximate runs differ mostly on near-tie documents, which click-based judgements rarely cover. **Mitigation:** report the share of judged results, compute nDCG three ways (unjudged as non-relevant, condensed lists, bootstrapped), and use the 2026 LLM judgements as a second source.
5. **Statistical power.** Small differences need many queries. **Mitigation:** exact and approximate runs share queries and judgements, so pair them per query with t-tests or permutation tests.
6. **Build-to-build noise.** HNSW builds are not deterministic; Lin's repeated builds varied by up to about 0.009 nDCG@10. **Mitigation:** three to five builds per setup.
7. **A null result.** The index term may be negligible. **Mitigation:** report it as a curve at recall 0.90, 0.95 and 0.99 rather than one number. A flat curve is still useful: a measured licence to ignore the index above a given recall.
8. **The drift signal has confounds.** A denser corpus raises the ratio on its own, and near-duplicates distort it. Three snapshots also can't show it rising early. **Mitigation:** compare old and new queries against a fixed corpus, use more snapshots (LongEval 2025 Web has seven monthly ones), and benchmark against established out-of-distribution scores.
9. **Maintenance rules face a known null.** A drift-triggered policy must beat a fixed schedule at equal cost, which the 2026 "When to Repair" study found it doesn't. "Rebuild on a schedule, recalibrate every snapshot" has to be an acceptable outcome.

## 6. What to do next

**Step 1. Run a pilot, which decides the project.**
- Use the organisers' released exact Qwen3-Embedding-4B indexes for LongEval 2026 scientific search, which avoids re-embedding the corpus. Pair them with pyterrier-dr's HNSW retriever as the approximate counterpart.
- At each snapshot, compute the index term for a per-snapshot rebuild and for an index grown in time order.
- Set search effort for recall of 0.90, 0.95 and 0.99 against exact search.
- Add a size-matched rebuild, and three to five builds per setup.

The pilot decides it either way. If the index term changes no more than build-to-build noise, the "growing index error" angle is dead and the project becomes a short, useful note for the lab. If it grows beyond the size control, that is the strongest evidence available.

**Step 2. Close the review's coverage gaps,** in a browser, in this order:
- the Google Scholar "Cited by" pages for DeDrift, PLAID SHIRTTT and Keller et al. 2024, then Lin 2024, Quake and MURR, filtered for nDCG with "temporal", "stream", "snapshot" or "update";
- the full texts of BANCO and Embedding Surgery;
- recent work from the PLAID SHIRTTT and MURR group (Lawrie, Yang, Mayfield, Oard), the team most likely to close this gap first.

Any paper reporting relevance effectiveness under ANN maintenance on time-ordered snapshots against exact search would reduce the contribution to a replication on LongEval.

**Step 3. Verify the data:** 2026 per-snapshot sizes and query counts, and whether CORE ever removes documents. That decides whether LongEval 2025 Web should be added.

**Step 4. Revise the research statement:**
- reframe the premise as in section 3.4;
- add the closest prior work from section 3.2;
- open with DS@GT's 2025 suggestion;
- state the narrowest defensible claim from section 3.3;
- treat the drift signal and the maintenance rule as tests against established baselines, not as contributions in themselves.

**Step 5. Then the full study,** if the pilot supports it:
- a factorial design separating growth, insertion order, deletions and query shift;
- each maintenance action alone and combined;
- a partition-index (IVF) arm, where drift effects are documented, to guard against a null HNSW result;
- results expressed in LongEval's own measures, with exact search as the pivot;
- targeting LongEval 2027 working notes if the lab runs, or the public 2026 data as an analysis paper otherwise.

## Selected references

- Lawrie, Kayi, Yang, Mayfield, Oard. PLAID SHIRTTT for Large-Scale Streaming Dense Retrieval. SIGIR 2024. <https://arxiv.org/abs/2405.00975>
- Yang, Tonellotto, Lawrie, MacAvaney, Mayfield, Oard, Miller. MURR: Model Updating with Regularized Replay for Searching a Document Stream. 2025. <https://arxiv.org/abs/2504.10250>
- Amendola, Mallia, Perego. Embedding Surgery. 2026. <https://arxiv.org/abs/2609.05110>
- Saravanan, Salla, Amancherla. AURORA: Neuro-Symbolic Continual Indexing for Evolving RAG Systems. Findings of ACL 2026. <https://aclanthology.org/2026.findings-acl.495/>
- Mohoney et al. Quake: Adaptive Indexing for Vector Search. OSDI 2025. <https://arxiv.org/abs/2506.03437>
- Baranchuk, Douze, Upadhyay, Yalniz. DeDrift: Robust Similarity Search under Content Drift. ICCV 2023. <https://arxiv.org/abs/2308.02752>
- Mandarapu, Kunkunuru. When to Repair a Graph ANN Index: A Matched-Budget Negative Result. 2026. <https://arxiv.org/abs/2607.00728>
- Lin. Operational Advice for Dense and Sparse Retrievers: HNSW, Flat, or Inverted Indexes? ACL 2025 Industry. <https://arxiv.org/abs/2409.06464>
- Kuissi, Subrahmanyan, Thakur, Lin. Still Fresh? Evaluating Temporal Drift in Retrieval Benchmarks. 2026. <https://arxiv.org/abs/2603.04532>
- Keller, Breuer, Schaer. Evaluation of Temporal Change in IR Test Collections. ICTIR 2024. <https://arxiv.org/abs/2407.01373>
- Stahlberg, Byrne. On NMT Search Errors and Model Errors: Cat Got Your Tongue? EMNLP 2019. <https://aclanthology.org/D19-1331/>
- Singh et al. FreshDiskANN. 2021. <https://arxiv.org/abs/2105.09613>
- Xu et al. SPFresh: Incremental In-Place Update for Billion-Scale Vector Search. SOSP 2023. <https://arxiv.org/abs/2410.14452>
- Xiao et al. Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation. 2024. <https://arxiv.org/abs/2407.07871>
- Leto et al. Toward Optimal Search and Retrieval for RAG. 2024. <https://arxiv.org/abs/2411.07396>
- Dimitropoulos, Mamoulis. ANN Search: Recall What Matters. 2026. <https://arxiv.org/abs/2606.04522>
- Sun, Ming, Zhu, Li. Out-of-Distribution Detection with Deep Nearest Neighbors. ICML 2022. <https://arxiv.org/abs/2204.06507>
- Chen et al. RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal ANN Search. VLDB 2024. <https://arxiv.org/abs/2408.08933>
- Fröbe et al. Bootstrapped nDCG Estimation in the Presence of Unjudged Documents. ECIR 2023. <https://downloads.webis.de/publications/papers/froebe_2023a.pdf>
- Extended Overview of the CLEF 2026 LongEval Lab. CEUR-WS Vol-4283, paper 374. <https://ceur-ws.org/Vol-4283/paper374.pdf>
- Kumaran D et al. (Lexora). CEUR-WS Vol-4283, paper 379. <https://ceur-ws.org/Vol-4283/paper379.pdf>
- Wu, Yang (GLHF). CEUR-WS Vol-4283, paper 384. <https://ceur-ws.org/Vol-4283/paper384.pdf>
- Miyaguchi, Afrulbasha, Pramov. DS@GT at LongEval: Evaluating Temporal Performance in Web Search Systems and Topics with Two-Stage Retrieval. 2025. <https://arxiv.org/abs/2507.08360>
- Cancellieri et al. LongEval at CLEF 2025. <https://arxiv.org/abs/2509.17469>
- LongEval organisers' dense baseline. <https://github.com/clef-longeval/longeval-code/tree/main/clef26/scientific-retrieval/baseline-pyterrier-dense>

The two full literature reviews behind this report, with every citation and verdict, are in `reports/ANN index drift in LongEval.md` and `reports/ANN index drift novelty check.md`.
