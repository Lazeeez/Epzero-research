# Insertion-order, arrival-time and recency bias in graph-based ANN indexes: prior art

Scope note: searched arXiv, ACM/Crossref metadata, Semantic-Scholar-style aggregators, and the GitHub issue trackers of hnswlib, FAISS, Lucene, DiskANN (via the public GitHub search API), plus downstream HNSW users (Chroma, seekdb/VSAG). Today is 2026-11-23. "Preprint" means arXiv with no journal-ref and no acceptance note found. The library sources were read directly from GitHub `main`/`master` on 2026-11-23.

## Q1. Has anyone measured in-degree, reachability or recall as a function of insertion position / item age in graph indexes?

### Takeaway
No source found stratifies in-degree, reachability or recall by insertion position or item age for HNSW, NSG, Vamana/DiskANN, FreshDiskANN, IP-DiskANN or CAGRA. The closest items are: (a) GRAB-ANNS, which names the mechanism ("Isolation of Newcomers": older neighbours form a "geometric shield" so new nodes fail the pruning test) but never measures it by age; (b) the FlatNav "hub highway" paper, which regresses node *access count* (not in-degree) on insertion order and finds it explains at most 0.5% of variance on real datasets; (c) a Lucene issue (Nov 2025) that proposes plotting neighbour quality against insertion time but has no data; (d) a handful of 2026 GitHub bug reports showing that specific new or early vectors become unreachable under particular conditions (duplicates, cluster-ordered arrival, distribution shift).

### Cited Findings
- **GRAB-ANNS** (GPU-native bucketing, hybrid search; arXiv 2604.16402v1, listed as submitted 31 Mar 2026, cs.DB). It is a preprint: it uses the PVLDB template but no acceptance is stated. The paper explicitly describes "Isolation of Newcomers": existing neighbours "often form a geometric 'shield' around v", so newly inserted nodes fail the diversity test even when geometrically close, and "entrenched" edges leave new nodes "under-connected (isolated) and thus unreachable during search". It gives **no per-node or per-age quantification** of this isolation. Recall is reported against α and filter range ratio. Insertion stability is reported as QPS versus dataset size (Sec 5.3.2). Connectivity is reported as whole-graph strongly-connected-component (SCC) counts (Sec 5.4). — [arXiv 2604.16402](https://arxiv.org/html/2604.16402)
- **"Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'"** (arXiv 2412.01940, v1 2 Dec 2024, v3 3 Jul 2025; preprint, no venue listed). Appendix F.3 tests "metric-space hubness vs preferential attachment". It fits a log-log linear model of node access count against insertion order (M=32, efC=100, efS=200, k=100). R² was 3.0–24.6% for IID-normal angular data (d=16 to 1536) and <0.1–7.6% for IID-normal L2. On real data it was 0.2% (GloVe), 0.3% (NYTimes), <0.1% (GIST), 0.3% (Yandex-DEEP) and 0.5% (SpaceV). The authors call the effect "modest" and attribute hubs mainly to dimensionality. They **did not measure in-degree directly**, did not report the sign or coefficient of the insertion-order effect, and did not analyse node age against hub status. They could not compute k-occurrence because of its O(n²) cost. — [arXiv 2412.01940v3](https://arxiv.org/html/2412.01940v3)
- **Lucene issue #15427** "Should we rewrite/optimize the HNSW graph in 2nd pass?" was opened by a Lucene committer on 2025-11-13 and is still open. It asks whether "vectors added early can have poor-ish neighbors since they didn't see all the future vectors yet", and suggests plotting "average node neighbor distance as a function of when the node was added" to see whether bidirectional linking offsets this bias. No measurement was posted. A maintainer replied that "it is possible there are weird cases where a node A would be near new node B, but never gets the back-link because the new B node doesn't consider node A one of its nearest neighbors." Follow-up: issue #15504 (use old segment graphs during merge so candidate retrieval "see[s] everything"). A 2026-11-17 comment calls the problem "future-blindness" of early-inserted vectors. — [apache/lucene#15427](https://github.com/apache/lucene/issues/15427)
- **IP-DiskANN** (arXiv 2502.13826, 19 Feb 2025; preprint, its text says it is under anonymous review). It reports only run-averaged Recall@10 per runbook, for example SlidingWindow MSTuring-10M: FreshDiskANN 94.4, IP-DiskANN 94.8, HNSW 91.8; Clustered MSTuring-30M: 91.8 / 92.5 / 89.5. It has **no breakdown by recency, no in-degree distribution, and no per-age statistics**. — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- **FreshDiskANN** (arXiv 2105.09613, 20 May 2021; arXiv lists no venue). It tracks recall and *average* degree over delete/re-insert cycles, and recall while ramping an index up from 100K to 1M points and from 1 point to SIFT100M (Figs 9 and 10). It never splits recall by when a point was inserted. — [arXiv 2105.09613 PDF](https://arxiv.org/pdf/2105.09613)
- **CAGRA** (arXiv 2308.15136, ICDE 2024). It targets reachability through SCC count and average 2-hop node count. In-degree is not a stated target. It gives no in-degree distribution and does not address incremental insertion; it is a batch NN-Descent plus optimisation pipeline. — [arXiv 2308.15136v2](https://arxiv.org/html/2308.15136v2)
- **Slipstream** (arXiv 2606.02992, 2 Jun 2026; preprint) speeds up streaming HNSW/FAISS insertion by seeding each insert from the previous insert's candidates. Recall is aggregate only (≥0.95 recall@10), with no new-vs-old breakdown in the abstract. — [arXiv 2606.02992](https://arxiv.org/abs/2606.02992)
- **"Recall-Bounded Staleness for Filtered Multi-Tenant ANN Graph Indexes under Concurrent Updates"** (arXiv 2610.15130, Oct 2026; preprint). It defines staleness as "updates not yet integrated into the graph", which is a batch-merge notion on a FreshDiskANN substrate. It reports terminal recall@10 (e.g. 99.44%→99.01%) as merge size varies and says larger merges degrade recall super-linearly. The abstract mentions no per-age recall and no in-degree of new nodes. — [arXiv 2610.15130](https://arxiv.org/abs/2610.15130)
- **DiskANN RFC PR #1546** (2026-11-01, open) concerns online insertion up to 63.94M vectors. Baseline mean distance-threshold Recall@20 was 93.980% over the first 32 snapshots and 92.349% over the last 32, so recall declines as the index grows by insertion. Its metric is top-20 churn, not per-age recall. — [microsoft/DiskANN#1546](https://github.com/microsoft/DiskANN/pull/1546)

### Inferences
- The specific measurements in our finding have not been published for HNSW or any other graph index, as far as these searches reach. Those measurements are: base-layer in-degree by insertion decile (28.6→13.3); the same gradient under a shuffled order, which separates position from corpus order; and recall of relevant documents split into new and old (2–3× pooled, 1.3–1.7× within-query).
- GRAB-ANNS is the nearest work on mechanism, but it is (i) GPU-only, (ii) batch-insertion with a fixed degree, (iii) evaluated on SIFT-128 filtered search, and (iv) never quantifies the bias by age. It supports the mechanism qualitatively and leaves the measurement and the CPU/HNSW case open.
- FlatNav's low R² for insertion order is about **traffic (access counts) under search**, not in-degree. A halving of mean in-degree across deciles can coexist with low node-level R² for access counts, because node-level variance is large. It should still be cited as the only published quantitative test of insertion order against a node-centrality measure, and the difference in what was measured should be explained.

### Gaps
- Could not read Figure 12 of GRAB-ANNS (values exist only in the figure). The text gives no numbers.
- Could not access the full text of the SIGMOD 2026 hubness paper (ACM returned 403) to check whether it plots hub status against insertion order.
- No source found for NSG, Vamana, CAGRA or SPANN/SPFresh reporting in-degree or recall by insertion time.

## Q2. Elliott and Clark 2024 (arXiv 2405.17813) on insertion order and LID: what exactly do they show; is it peer-reviewed?

### Takeaway
It is peer-reviewed: ICTIR '24 (ACM SIGIR International Conference on the Theory of Information Retrieval), pp. 25–33, DOI 10.1145/3664190.3672512. It shows that *whole-index* recall depends on insertion order sorted by local intrinsic dimensionality (LID) or by category, by up to about 12 points. It does **not** analyse early versus late items, in-degree or reachability.

### Cited Findings
- Title: "The Impacts of Data, Ordering, and Intrinsic Dimensionality on Recall in Hierarchical Navigable Small Worlds". arXiv v1 is dated 28 May 2024. Crossref records it in *Proceedings of the 2024 ACM SIGIR International Conference on Theory of Information Retrieval* (ICTIR '24, Washington DC), pp. 25–33, published 2024-08-02. — [arXiv 2405.17813](https://arxiv.org/abs/2405.17813); [Crossref record](https://api.crossref.org/works/10.1145/3664190.3672512)
- Setup: hnswlib and FAISS, M=16, efC=128 (the median of vector-DB defaults), efSearch 10 and 40, 16 embedding models × 7 benchmark datasets, plus proprietary e-commerce CLIP data. — [arXiv html](https://arxiv.org/html/2405.17813)
- Inserting in **descending LID** order gave higher recall than random or ascending for all models except gte-base on FAISS. Random order fell between the two. At efSearch=10 the mean gain was +2.6 points (hnswlib) and +6.2 points (FAISS). The largest gaps were 5.6 (hnswlib, e5-small-v2) and 12.8 (FAISS, all-MiniLM-L6-v2) at ef=10, and 7.1 and 11.5 at ef=40. Category-sorted orders differed by about 7.7 points (fashion, ViT-B-32, ef=10) and about 2–2.5 points (homewares). — [arXiv html](https://arxiv.org/html/2405.17813)
- Proposed mechanism: on four small datasets, recall correlated with mean base-layer path length (Pearson r=0.61). The authors hypothesise that inserting high-LID points first "delays the integration of tight low-LID clusters". There is no early-versus-late, in-degree or reachability analysis. — [arXiv html](https://arxiv.org/html/2405.17813)

### Inferences
- Elliott and Clark establish order sensitivity of *global* recall when the order is *correlated with data properties*. Our finding is different in kind: a *position* effect that persists under random order, and a *per-item* recall disparity by age. The two are complementary. Elliott and Clark do not anticipate or measure the per-item disparity.

### Gaps
- Not verified whether the ICTIR camera-ready differs from arXiv v1. There is only one arXiv version.

## Q3. GRAB-ANNS (arXiv 2604.16402): freshness-bias pruning rule (Sec 4.4.3), ablation (Sec 5.6.3, Fig 12); does it measure bias by age; is it CPU-applicable?

### Takeaway
GRAB-ANNS adds a multiplicative discount α∈(0,1) to the left-hand side of the RNG/"small-world" pruning test whenever the candidate is a fresh node from the current batch. This directly targets the "isolation of newcomers" mechanism. The ablation reports only aggregate filtered recall against α on SIFT-128. Bias is never measured by age, and the method is evaluated only on GPU.

### Cited Findings
- Base rule (Eq. 2): a candidate c may take a neighbour slot of v only if dist(v,c) < dist(c,n) for every existing neighbour n. Freshness bias (Eq. 3): d_eff(v,x) = α·dist(v,x) if x is from the new batch, otherwise dist(v,x), with α∈(0,1). The paper does not say whether the right-hand term is also discounted when n is fresh. — [arXiv 2604.16402](https://arxiv.org/html/2604.16402)
- Motivation is GPU-specific: a fixed maximum degree K_max is needed for coalesced memory access. The paper notes that heuristic pruning comes from CPU indexes such as HNSW. The formula itself contains nothing GPU-specific, but there is **no CPU evaluation**. — [arXiv 2604.16402](https://arxiv.org/html/2604.16402)
- Sec 5.6.3 / Fig 12: SIFT-128 with dynamic insertion, itop_K=128, search_width=4. At α=1.0 (unbiased) recall "drops sharply" for selective filters (1% and 20% range ratio), and α≈0.4–0.6 is best across ratios. The text gives no numbers. There is an internal inconsistency: α is defined on (0,1) but the default is given as "0.6 (or 0.4)". — [arXiv 2604.16402](https://arxiv.org/html/2604.16402)
- There is no stratification of recall or reachability by vector age. — [arXiv 2604.16402](https://arxiv.org/html/2604.16402)

### Inferences
- On CPU HNSW, the analogue would be applying α only inside the *reverse-link* re-prune of full neighbours (hnswlib `mutuallyConnectNewElement`, FAISS `shrink_neighbor_list`) when the candidate is the new node. That is a one-line, testable intervention, and it is untested in the literature. A fixed α trades away diversity, so it needs an equal-memory and equal-ef comparison.
- Because GRAB-ANNS shows the problem only through aggregate filtered recall, it cannot separate "new nodes are under-linked" from "selective filters are hard". Measuring by age would be new evidence even for their own setting.

### Gaps
- The final 16.5K characters of the HTML (mostly references and appendix) were not read. Figure 12 values are not recoverable from the text.

## Q4. MN-RU (arXiv 2407.07871) and other "unreachable points" work; any insertion-only analysis?

### Takeaway
The "unreachable points" literature is entirely about **delete/update churn**. MN-RU explicitly says unreachable points are "exceedingly small" in likelihood during plain construction, and does not measure the insertion-only case. Our finding, which is in-degree erosion and recall loss for new items with no deletes, falls outside this literature's framing.

### Cited Findings
- MN-RU: "Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation" (arXiv 2407.07871, v1 10 Jul 2024, v2 15 Jul 2024; preprint). It defines an unreachable point as one with out-edges but **no in-edges in any layer**. On SIFT, with 5% delete+reinsert per iteration over 3,000 iterations, unreachable points reached about 3–4% and recall fell about 3%, and raising ef does not fix it. After 200 iterations HNSW-RU reached about 3–4% unreachable on GIST and 2–3% on ImageNet. The cause is the `replaced_update` repair in hnswlib. Remedies are MN-RU (2–4× faster updates), a backup index, and dualSearch. It says the likelihood during construction is "exceedingly small" and gives **no static-build count**. — [arXiv 2407.07871](https://arxiv.org/html/2407.07871)
- LR-RN: "Robust HNSW Vector Retrieval under Dynamic Product Updates via Local Rewiring" (CEUR-WS Vol-4250, paper 776; workshop paper, independent researcher; the year is not printed in the extract, and the CEUR volume number suggests 2026). It says existing repair strategies "silently accumulate unreachable nodes" under heavy update workloads, and claims LR-RN cuts unreachable-node growth by two thirds on ANN benchmarks with recall degradation within 1–3% over cumulative cycles. It covers updates only. — [CEUR-WS Vol-4250 paper776](https://ceur-ws.org/Vol-4250/paper776.pdf)
- FreshDiskANN Fig 1: over 20 cycles of deleting and re-inserting 5% of SIFT1M, statically built HNSW, Vamana and NSG all show "a consistently deteriorating trend" in recall. The authors attribute this to the graph becoming sparser, "due to the very aggressive pruning policies of existing algorithms such as HNSW and NSG". — [arXiv 2105.09613 PDF](https://arxiv.org/pdf/2105.09613)
- Lucene #16552 (2026-08-24, closed) and PR #16555: when merges reuse a graph, out-degree decays silently across merge generations, because the repair threshold is relative to the node's *current* degree. Over 100 generations of 2% deletes (GloVe-100, M=32), `main` loses 7.0 / 5.6 pp recall@100 at generations 50 / 100, with L0 out-degree 15.5 against 34.7 for a rebuild. A cumulative degree floor of 0.5 restores recall at about 6× the merge time. — [apache/lucene#16552](https://github.com/apache/lucene/issues/16552); [apache/lucene#16555](https://github.com/apache/lucene/pull/16555)

### Inferences
- The literature treats in-degree loss as a by-product of deletes or merges. No source found frames *insertion alone* as producing a systematic in-degree gradient by position.

### Gaps
- No insertion-only unreachable-point count was found for any library.

## Q5. Hubness and in-degree skew in kNN/ANN graphs; hub-based HNSW variants; relation to item retrievability

### Takeaway
Hubness is well established for kNN graphs (Radovanović et al. 2010) and is now established as a structural cause of skew in ANN graphs (SIGMOD 2026, FlatNav). Both attribute skew to the **geometry** of high-dimensional data, not to insertion position. No source connects ANN-graph in-degree to IR **retrievability**, meaning how often each document can be retrieved.

### Cited Findings
- Radovanović, Nanopoulos, Ivanović, "Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data", JMLR 11:2487–2531 (2010); preliminary version at ICML 2009. The k-occurrence distribution becomes strongly skewed as dimensionality grows, and this is an inherent property of high-dimensional data. — [JMLR](https://jmlr.org/papers/v11/radovanovic10a.html)
- "Through the Lens of Hubness: A Revisit on Graph-Based Approximate Nearest Neighbor Search [Experiments & Analysis]", Proc. ACM Manag. Data (SIGMOD) vol 4, no 3, published 2026-05-18, DOI 10.1145/3802120. It says hubness "induces topologically skewed proximity graphs" with central hubs and "isolated anti-hubs", which breaks greedy traversal for queries that target anti-hubs. It gives a taxonomy of ANNS designs as hubness-mitigation strategies, with experiments up to 100M vectors, and finds a "trade-off between suppressing hubs and ensuring the reachability of anti-hubs". It proposes a low-overhead hubness-aware pruning. Peer-reviewed. — [Crossref abstract](https://api.crossref.org/works/10.1145/3802120); [ACM DL](https://dl.acm.org/doi/10.1145/3802120)
- FlatNav / Hub Highway Hypothesis (arXiv 2412.01940; preprint): a flat NSW matches HNSW on high-dimensional data because a well-connected set of hub nodes acts as the "highway". Access-count skew grows with dimension for L2 and shows little skew for cosine. Insertion order explains ≤0.5% of access-count variance on real data (see Q1). — [arXiv 2412.01940v3](https://arxiv.org/html/2412.01940v3)
- Hubness reduction for ANN: scikit-hubness (arXiv 1912.00706) does ANN candidate retrieval followed by hubness-reduction re-ranking. — [arXiv 1912.00706](https://arxiv.org/pdf/1912.00706). The rnndescent vignette notes that adding reverse edges ensures antihubs appear in their neighbours' lists. — [rnndescent hubness vignette](https://search.r-project.org/CRAN/packages/rnndescent/vignettes/hubness.html). See also "The Influence of Hubness on NN-Descent". — [NJIT record](https://digitalcommons.njit.edu/fac_pubs/7368)
- Practitioner evidence of per-query systematic misses: an OpenSource Connections post (2025-02-27) on Lucene HNSW reports that some queries with an "unfortunate" embedding miss multiple true top documents even at high k. — [OSC blog](https://opensourceconnections.com/blog/2025/02/27/vector-search-navigating-recall-and-performance/)

### Inferences
- Anti-hub status in the hubness literature is a function of **position in space**. Our effect is a function of **position in the insertion sequence**, and it survives shuffling. The two are distinct, could interact, and should be controlled for each other. One way is to regress per-item in-degree on insertion decile with exact k-occurrence (or local density) as a covariate. FlatNav could not do this because of the O(n²) cost; at 1.66M it is feasible with GPU brute force.
- Linking ANN in-degree to IR retrievability for documents, and to *temporal* retrievability for new documents, appears unexplored in the sources found.

### Gaps
- No paper found that measures retrievability (in the Azzopardi–Vinay sense) of documents under an ANN graph index against exact search. The concept's origin was not re-fetched in this session.
- The full SIGMOD 2026 hubness text was inaccessible (403), so it is unverified whether its pruning is in-degree-aware or whether it reports in-degree by insertion order.

## Q6. Theory: navigability, RNG pruning, in-degree distributions, incremental vs batch construction

### Takeaway
Theory covers sparsity and navigability (out-degree, edge count, α-navigability) and worst-case guarantees. No theoretical work found models **in-degree as a function of insertion time** under conditional reverse links. The one explicit incremental-versus-batch statement (PVLDB 2025) concerns forward-candidate quality, which is the opposite side of the same asymmetry.

### Cited Findings
- "Revisiting the Index Construction of Proximity Graph-Based ANN Search" (arXiv 2410.01231, Oct 2024, revised Feb 2025; accepted to PVLDB 2025). Because NSWG and HNSW insert "node-by-node" and each insert searches only the partial graph, the "expected value of k-CNA quality in NSWG is only 0.5 even if all the k-ANN queries are answered correctly", and HNSW's k-CNA quality is "limited to 0.5 due to its incremental node-by-node insertion strategy" (Fig 2b). The fix is layer-by-layer *global* construction. The paper does **not** analyse in-degree or insertion order. — [arXiv 2410.01231v2](https://arxiv.org/html/2410.01231v2)
- Khanna, Padaki, Waingarten, "Sparse Navigable Graphs for Nearest Neighbor Search: Algorithms and Hardness" (arXiv 2507.14060, 18 Jul 2025; preprint, cs.DS). Finding the sparsest α-navigable graph is approximation-equivalent to Set Cover: there is an O(ln n) approximation, and o(ln n) is NP-hard. Slow-preprocessing DiskANN can be Ω̃(n) times denser than optimal. Any o(n)-approximation must examine Ω(n²) distances. The paper does not address incremental construction or in-degree. — [arXiv 2507.14060](https://arxiv.org/abs/2507.14060)
- "Graph-Based Nearest-Neighbor Search with Recall Guarantees Using Graph Spanners" (arXiv 2607.02338, v1 2 Jul 2026, v3 11 Nov 2026; preprint). Certify-then-Rectify accepts an HNSW result under a calibrated rule. Otherwise it expands using graph-stretch bounds and falls back to an exact scan, giving an "end-to-end population recall guarantee". It does not address incremental insertion or degree distribution. — [arXiv 2607.02338](https://arxiv.org/abs/2607.02338)
- The hnswlib PR #674 text states the asymmetry informally: an element "can acquire a later element only if that element subsequently selects it, and `getNeighborsByHeuristic2` is not symmetric, so often it does not". — [nmslib/hnswlib#674](https://github.com/nmslib/hnswlib/pull/674)

### Inferences
- Combining the PVLDB result (early nodes have weak forward candidates) with our measurement (late nodes have few in-links) gives a two-sided picture of incremental construction. Early nodes are out-degree-poor in quality but in-degree-rich. Late nodes have good forward links but few in-links. Neither side has a published analytic model of in-degree versus insertion time. A simple model (expected in-links ≈ Σ over later arrivals of P[selected as forward neighbour] × P[survives pruning in the other direction]) would be new.
- Guarantees in the CTR style are a query-side backstop that would also catch missed new items, but they do not remove the structural bias.

### Gaps
- Did not re-fetch Indyk and Xu, "Worst-case performance of popular ANN implementations" (NeurIPS 2023), or the original HNSW paper's discussion of NSW long-range links coming from early elements. The arXiv abstract of 1603.09320 does not mention insertion order. Both should be checked before citing specifics.

## Q7. Known remedies (backlink repair, refinement, α-relaxed pruning, in-degree-aware pruning, re-insertion, consolidation): which would remove an age bias, and which were evaluated for it?

### Takeaway
None of the remedies found has been evaluated for an age or recency bias. Several target the same structural symptom (low in-degree or unreachable nodes), but each was evaluated on churn, duplicates, ordered corpora, or aggregate recall.

### Cited Findings
- **Shuffle insertion order** (hnswlib PR #674, opened 2026-08-29, open). With `add_items(shuffle=True)` by default, a cluster-ordered synthetic 1M×128 set gains +2.5 to +8.0 points accuracy@10 across the ef ladder. MS MARCO v1 passage / Dragon (8.84M×768, inner product) gains +3.71 at ef=25 and +0.83 at ef=520, a mean of +1.98, and needs 1.36–2.64× fewer distance computations for equal accuracy. SIFT1M, whose file order is unstructured, gains only +0.11. It targets **corpus-order** effects only. — [nmslib/hnswlib#674](https://github.com/nmslib/hnswlib/pull/674)
- **α-relaxed pruning (Vamana/FreshVamana).** With α>1, average degree rises over update cycles and recall stays stable, while α=1 degrades. α=1.2 was chosen (R=64, L=75). This was evaluated on delete+reinsert cycles. — [arXiv 2105.09613 PDF](https://arxiv.org/pdf/2105.09613). IP-DiskANN adds a back-edge on insertion and re-prunes an over-full neighbour with RobustPrune at α≈1.2. — [arXiv 2502.13826](https://arxiv.org/html/2502.13826)
- **Freshness-biased pruning** (GRAB-ANNS α·dist for fresh candidates). Evaluated on aggregate filtered recall on GPU only (Q3). — [arXiv 2604.16402](https://arxiv.org/html/2604.16402)
- **Re-insertion / re-upsert.** In Chroma #7963, re-upserting the early small-collection vectors unchanged restored self-recall@6 to 120/120 and 40/40, from 80–109/120. — [chroma-core/chroma#7963](https://github.com/chroma-core/chroma/issues/7963)
- **Second pass / hindsight refinement.** Proposed for Lucene (#15427, #15504) and untested there. — [apache/lucene#15427](https://github.com/apache/lucene/issues/15427). FreshDiskANN notes that static Vamana, built with two refinement passes, "might have slightly better quality than the streaming FreshVamana algorithm" (a small initial recall drop). — [arXiv 2105.09613 PDF](https://arxiv.org/pdf/2105.09613). The PVLDB 2025 global layer-by-layer construction (Q6). — [arXiv 2410.01231v2](https://arxiv.org/html/2410.01231v2)
- **Connect-components pass (Lucene).** `HnswGraphBuilder.connectComponents()` links smaller components to the largest one. It is currently commented out in `finish()` ("Connect components can be exceptionally expensive, disabling, see #14214"). — [Lucene HnswGraphBuilder.java (main)](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/hnsw/HnswGraphBuilder.java); [apache/lucene#14214](https://github.com/apache/lucene/issues/14214)
- **Sink repair (FAISS PR #5705, closed 2026-10-05).** `IndexHNSW::add_level_0_links_from_sample()` finds "sinks", meaning nodes where greedy search stops although the nearest node is elsewhere, and adds level-0 links to the true nearest nodes of their wrongly routed vectors. Runs with a sink fell from 5/20 to 0/20 on 65,536-centroid quantizers (d=2048). Build-side changes did not remove sinks: smaller batches, `keep_max_size_level0`, a minimum level-0 degree, and sequential build all left sinks in 2–6 of 15 runs. — [facebookresearch/faiss#5705](https://github.com/facebookresearch/faiss/pull/5705)
- **Duplicate handling.** FAISS PR #5969 "Keep exact-duplicate vectors reachable in HNSW" (2026-11-21, open). — [faiss#5969](https://github.com/facebookresearch/faiss/pull/5969). VSAG `support_duplicate=true` fixed the seekdb reproducer, tracked upstream as antgroup/vsag#3447. — [oceanbase/seekdb#2653](https://github.com/oceanbase/seekdb/issues/2653)
- **Degree-floor repair on merge** (Lucene PR #16555): see Q4 numbers. — [apache/lucene#16552](https://github.com/apache/lucene/issues/16552)
- **Churn repairs.** MN-RU, backup index and dualSearch. — [arXiv 2407.07871](https://arxiv.org/html/2407.07871). LR-RN. — [CEUR-WS](https://ceur-ws.org/Vol-4250/paper776.pdf)
- **Reverse-edge addition at build (CAGRA).** It merges about d/2 pruned and d/2 reversed edges per node, mainly to reduce SCCs. It is batch-only. — [arXiv 2308.15136v2](https://arxiv.org/html/2308.15136v2)
- **Even-regular undirected graphs (DEG).** The graph is built incrementally as a single undirected even-regular graph with continuous edge refinement, so every vertex has the same degree, and connectivity is guaranteed. — [arXiv 2307.10479](https://arxiv.org/pdf/2307.10479); follow-up [arXiv 2607.27640](https://arxiv.org/pdf/2607.27640)
- **Hubness-aware pruning** (SIGMOD 2026; Q5). — [Crossref](https://api.crossref.org/works/10.1145/3802120)
- **Query-side fallback.** Certify-then-Rectify with exact-scan fallback (Q6). — [arXiv 2607.02338](https://arxiv.org/abs/2607.02338). Chroma has an exact-search fallback for fragmented indexes (PR #7845, referenced in #7963). — [chroma-core/chroma#7963](https://github.com/chroma-core/chroma/issues/7963)

### Inferences
Which remedies should remove an insertion-position bias (expected from the mechanism; none verified for age):
- **Should remove it structurally:**
  - Constructions where in-degree cannot depend on arrival order. These are DEG's regular undirected graph, and global/batch refinement over the full set: Vamana's second pass, NSG and CAGRA reverse-edge merging, and PVLDB-2025 global layers. All of them re-prune every node over a candidate set that includes later arrivals.
  - Periodic re-insertion of recent arrivals. This gives new nodes a second chance to be selected by later neighbours.
- **Should reduce it:**
  - Freshness-biased (GRAB-style) α applied in the reverse-link re-prune.
  - Vamana α>1, which relaxes the test that rejects the newcomer.
  - Lucene-style "add, then evict one worst-non-diverse neighbour" back-linking (see Q8). Its pruning test is milder than hnswlib/FAISS's full re-prune, but it can still evict the new node when that node is the farthest and no neighbour is non-diverse.
  - A query-side exact-scan buffer for recent items. This removes the *symptom* for the newest items only.
- **Should not remove it:**
  - Shuffling. It fixes corpus-order effects, not position effects. Our shuffled-order in-degree gradient is exactly the residual that shuffling leaves.
  - Raising ef. Chroma #7963 shows ef 2000 did not help, and MN-RU says the same for unreachable points.
  - Delete-oriented repairs (MN-RU, LR-RN, the Lucene degree floor). They trigger only on deletes or merges.
- **Sink repair:** it should help only where sinks coincide with old hubs.

### Gaps
- No paper found that evaluates any of these remedies with recall stratified by insertion time.
- FAISS's `keep_max_size_level0` semantics were not re-verified in source. It is mentioned only in PR #5705.

## Q8. Library-level evidence (hnswlib / FAISS / Lucene / DiskANN and downstream users)

### Takeaway
None of hnswlib, FAISS, Lucene or DiskANN has an issue reporting "recall is lower for recently inserted vectors" as a general, age-dependent effect. Several recent (Aug–Nov 2026) reports describe special cases of the same mechanism, where a newly inserted vector loses reverse links to full, pruning neighbours (duplicates, distribution shift), or cases of order dependence. The two main libraries differ in the reverse-link rule.

### Cited Findings
- **hnswlib reverse-link rule** (`mutuallyConnectNewElement`, master). If the neighbour's list is below `Mcurmax`, the new node is appended. Otherwise the new node plus all existing neighbours are re-run through `getNeighborsByHeuristic2(candidates, Mcurmax)`, which can drop the new node, and can drop several old neighbours. Read from source on 2026-11-23. — [hnswlib hnswalg.h](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h)
- **FAISS**: reverse links from later points "lose to the tied copies in `merge_reverse_links_impl`". Ties do not prune one another in `shrink_neighbor_list` (strict comparison), and level-0 forward lists keep 2M targets. With 200 identical copies among 1,000 vectors and M=32, only 135/200 copies are findable at efSearch=1000. With M=16, 28/200 copies have in-degree 0, against 2/200 at M=32. On 423,137 real code chunks (17.6% exact duplicates), self-searches miss 1,045 duplicate chunks. (Issue #5968, opened 2026-11-21, open; the fix is PR #5969.) — [facebookresearch/faiss#5968](https://github.com/facebookresearch/faiss/issues/5968)
- **FAISS PR #5524** "Don't raise HNSW entry_point to a point that isn't linked yet" (closed 2026-08-17) is a construction-order corner case. — [faiss#5524](https://github.com/facebookresearch/faiss/pull/5524)
- **Lucene reverse-link rule** (`NeighborArray.addAndEnsureDiversity`, main). The new node is always added. If the list is then oversize, *one* neighbour is removed: the first "worst non-diverse" one scanning from the most distant, or else the most distant one (`size-1`). Read from source on 2026-11-23. — [Lucene NeighborArray.java](https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/hnsw/NeighborArray.java)
- **hnswlib #80** "Accuracy depends on the order of the insertion" (2018-12-11, closed, no data in the body). — [nmslib/hnswlib#80](https://github.com/nmslib/hnswlib/issues/80). hnswlib PR #674 (2026-08-29) has the ordered-corpus numbers in Q7. — [nmslib/hnswlib#674](https://github.com/nmslib/hnswlib/pull/674)
- **Chroma #7998** (2026-11-06, open). Chroma's HNSW (default ef_search=100, 384-d bge-small) misses a freshly inserted vector when queried with that same vector, at miss rates of 15% (N=1,000), 36% (N=2,645) and about 63% (N=10,000). ef_search=1000 still misses 13% at N=10,000. M=48 made it worse (50/120). A read between two writes raised the miss rate from 4% to 35–45%. No maintainer diagnosis. The title links it to "freshly-inserted/out-of-distribution vectors". — [chroma-core/chroma#7998](https://github.com/chroma-core/chroma/issues/7998)
- **Chroma #7963** (2026-10-27, open) shows the *opposite* direction. Vectors added while the collection was small become hard to find after a much larger batch. Self-recall@6 for a 7-vector document was 0/7 at ef 100. In a synthetic test, small-first gave 80–109/120 against 115–118/120 large-first. A commenter found no in-degree-0 nodes; only 38 of 1,717 links into the small set came from the large cluster. Part of the problem was a cached-ef bug where `modify` does not reach a loaded index. — [chroma-core/chroma#7963](https://github.com/chroma-core/chroma/issues/7963)
- **seekdb / VSAG #2653** (2026-11-15, open). 4,000 identical vectors followed by one distinct sentinel: the sentinel is never found at ef 64, 1000 and 4001. In a legacy-HNSW incident, the reporter proved vectors had "no incoming level-0 edges" because zero-distance ties filled neighbour slots and reverse edges were dropped. A commenter proposes that reverse-link pruning keeps the duplicates, which makes the new vector a zero-in-degree sink. — [oceanbase/seekdb#2653](https://github.com/oceanbase/seekdb/issues/2653)
- Other downstream reports surfaced by search: laurus #1241 (recall loss on cluster-by-cluster insertion), hyperspace-engine #538 (self-recall failures for sequentially inserted vectors "starting near the middle of the sequence"), and hadron-server #438 (small tenants' recall collapses in a shared index). These are search snippets only and were not read. — [mosuka/laurus#1241](https://github.com/mosuka/laurus/issues/1241); [skaiy/wild_agentos#538](https://github.com/skaiy/wild_agentos/issues/538); [hadron-memory/hadron-server#438](https://github.com/hadron-memory/hadron-server/issues/438)
- The Lucene merge degree-decay issue (#16552) and the DiskANN online-insertion RFC (#1546) are covered in Q4 and Q1.

### Inferences
- The FAISS #5968 and VSAG #2653 duplicate cases are *degenerate instances of our mechanism*. A new node's reverse link is rejected by full older neighbours whose pruning favours existing members, which leaves the new node with in-degree near zero. Our finding generalises this from exact ties to ordinary geometry, as a smooth gradient across insertion position.
- hnswlib and FAISS (full re-prune, which can reject the newcomer) differ from Lucene (always add, evict one). That predicts a smaller age gradient in Lucene. It is a cheap, decisive cross-implementation test that no one has published.
- Reliability: these GitHub reports are recent, unreviewed and mostly unconfirmed by maintainers. At least one (FAISS #5968) links scripts on a branch whose name suggests AI-assisted authoring. They should be cited as practitioner signals, not established results. Chroma #7998 is confounded by out-of-distribution queries and by Chroma's ef-caching bug.

### Gaps
- GitHub search was keyword-based and unauthenticated, with 5–8 results per query. Discussions, closed PR review threads and JIRA-era Lucene (LUCENE-9004 etc.) were not exhaustively searched.
- No blog post from a vector-database vendor was found that discusses recency-dependent recall for graph indexes.

## Q9. Novelty summary: how close does the nearest work come?

### Takeaway
The following appear to be **unpublished**:
- a per-item, age-stratified measurement of recall disparity in HNSW (FAISS and a second implementation) on a real growing corpus;
- a base-layer in-degree gradient by insertion decile that persists under shuffled order;
- attribution of that gradient to conditional reverse links with RNG re-pruning.

The nearest neighbours are GRAB-ANNS (same mechanism, stated qualitatively, GPU-only, no age measurement), FlatNav (insertion order against access count, small effect, no in-degree), Elliott and Clark at ICTIR 2024 (global recall against data-correlated order), hnswlib PR #674 (corpus-order effect and an informal asymmetry argument), and the duplicate-induced zero-in-degree reports in FAISS #5968 and VSAG #2653.

### Cited Findings
- Mechanism stated but not measured: [GRAB-ANNS](https://arxiv.org/html/2604.16402)
- Insertion order against node access, R² ≤ 0.5% on real data: [FlatNav arXiv 2412.01940v3](https://arxiv.org/html/2412.01940v3)
- Order-dependent global recall, up to about 12 pp, peer-reviewed at ICTIR '24: [arXiv 2405.17813](https://arxiv.org/abs/2405.17813)
- Lucene committers hypothesise an age-dependent neighbour-quality effect but have no data: [apache/lucene#15427](https://github.com/apache/lucene/issues/15427)
- Incremental insertion caps candidate quality at about 0.5 (forward side): [arXiv 2410.01231v2](https://arxiv.org/html/2410.01231v2)

### Inferences
- A defensible novelty claim: "the first measurement of an insertion-position (arrival-time) bias in per-document recall and base-layer in-degree for HNSW under insertion-only growth, shown to persist under random insertion order, with a mechanism tied to conditional reverse-link pruning." Cite GRAB-ANNS as prior qualitative identification of newcomer isolation in a GPU batch setting, and the hubness literature as an orthogonal, geometry-driven source of in-degree skew that must be controlled for.
- Risks to the claim:
  - The full text of the SIGMOD 2026 hubness paper was not readable. It may contain in-degree-by-order plots.
  - Lucene and others may have internal benchmarks that were never published.
  - Fast-moving 2026 GitHub threads (FAISS #5968/#5969, Chroma #7998) could produce an age-stratified analysis soon.

### Gaps
- No exhaustive Semantic Scholar citation sweep of GRAB-ANNS, FlatNav or Elliott and Clark was done to find later work that cites them and measures age effects.
