# Paper A: claims and the evidence behind each

Working map for the draft (`plan-paper-a.md`). Started on 28 November 2026; each row names the record that holds
the numbers. Epzero is named as one HNSW implementation among others. Rows marked *pending* wait for runs.

## C1. Incremental graph insertion serves its newest vectors worst

| Evidence | Numbers | Record |
| --- | --- | --- |
| LongEval-Sci, real arrival order, two HNSW implementations (frozen order rerun) | newest cohort's within-query excess +2-3 points as arrived | `research/longeval/HELDOUT_EXPLORATORY.md` §13-14, §19 |
| Big-ANN stream, `msturing-10M-clustered` runbook | within-query position effect, both implementations | `scripts/pace/BIGANN_STREAM.md` |
| Public data with random arrival cohorts (SIFT, GloVe) | position alone produces the gradient | `HELDOUT_EXPLORATORY.md` §14 |
| A third HNSW implementation (hnswlib) | as the other two | `docs/screen-x1-hnswlib.md` |
| Independent FreshDiskANN-style Vamana | as arrived 1.32 (LongEval), 1.99 (SIFT) | `docs/screen-m3-vamana.md` |
| DiskANN's library | 1.43 on SIFT, 0.99 on LongEval | `docs/screen-m3-vamana.md` |
| arXiv in submission order, ~2.8M abstracts | *pending* | `docs/screen-a1-arxiv-stream.md` |

## C2. The bias follows insertion position, not content

| Evidence | Numbers | Record |
| --- | --- | --- |
| Reversing the same cohorts reverses who loses | change −3.4 to −4.3 points, two engines | `HELDOUT_EXPLORATORY.md` §19 |
| Shuffled builds: last insertion tenth against first | positive at every scale tested | `HELDOUT_EXPLORATORY.md` §14, `BIGANN_STREAM.md` |
| Fixed-effects logistic model (per-query) | odds ratios as arrived / shuffled / reversed 1.77 / 1.02 / 0.77 | `docs/analysis-r1-reanalyses.md` |

## C3. The mechanism: exposure minus pruning

| Evidence | Numbers | Record |
| --- | --- | --- |
| Choices by later arrivals equal exposure Σ k_j/(j−1), no preferential term | within 1% per tenth, M 16 and M 32 | `docs/screen-m1-accumulation-model.md`, `docs/a3-mechanism-derivation.md` §5 |
| Pruning is bulk and mainly by length | lists cut from 33 to about 21; farthest fifth removed 2.6× the closest | `docs/a3-mechanism-derivation.md` §7 |
| Measured-rate model, nothing fitted to the scored build | M 16: S within 0.3 links, R within 1.4 | `docs/a3-mechanism-derivation.md` §9 |
| List-process model | R within 0.7 (M 16) and 1.8 (M 32) links | `docs/a3-mechanism-derivation.md` §11 |

## C4. The repair removes it at small cost

| Evidence | Numbers | Record |
| --- | --- | --- |
| Extra back-links plus batch refine, LongEval full scale | gap removed, about 9% more insertion work | `research/longeval/paper/manuscript-long.md` §13 |
| Big-ANN stream | gap removed | `BIGANN_STREAM.md` |
| Sparse deletions (LongEval, 10% per step) | −1.93 [−2.36, −1.52] points, all of it | `docs/screen-m2-deletions.md` Part B |
| CleanANN (authors' code) at the paper's bridge window | bridges barely fire; as arrived 1.47 / 1.50 (SIFT, off / on), 1.01 / 1.01 (LongEval): DiskANN's behaviour | `docs/screen-a2-cleanann.md` |
| CleanANN at a shallow window that fires | SIFT: as arrived 1.53, recall −0.06 points, 0.74× build; LongEval *pending* | same, amendment 2 |
| NGFix by arrival | needs new builds on PACE (AVX-512, query-fitted) | `plan-paper-a.md` A2 |

## C5. Deletion and other graph designs

| Evidence | Numbers | Record |
| --- | --- | --- |
| Heavy deletion with tombstones doubles the gap | +8.37 grown against +4.22 rebuilt | `docs/screen-m2-deletions.md` Part A |
| The repair alone closes a quarter of it | +6.20 | same |
| Compaction with the repair | +1.23 [+0.87, +1.60] points (85% of the grown excess removed), recall 0.797 | `docs/screen-a5-deletion-repair.md` |
| DiskANN's list slack protects late arrivals | slack: as arrived 1.29 SIFT, 1.04 LongEval; newest tenth's in-degree 25.0 and 27.4 (driver 1.99 / 1.32, 18.6 / 17.4; DiskANN 1.43 / 0.99, 26.4 / 28.1) | `docs/screen-a4-diskann-choices.md` |
| DiskANN's frozen start does not | a copy of the first vector: SIFT 1.98, in-degree 18.6 (as the driver); LongEval *pending*. A frozen node at the mean broke the graph on unit vectors | same, amendment |

## C6. Why benchmarks miss it

Streaming benchmarks (Big-ANN, CANDOR-Bench) average recall over the collection; Ada-ef and MERIT report overall
recall per batch. Prior work and its limits: `docs/arrival-bias-research.md` §7.
