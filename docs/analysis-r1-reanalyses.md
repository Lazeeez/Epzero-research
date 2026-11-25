# Analysis R1: three reanalyses of existing held-out and order-rerun outputs

Recorded on 26 November 2026, before any of them was computed. Exploratory; no new runs. Each uses
outputs already in the local evidence folders.

## 1. A calibration margin, as its own arm

The paper recommends a one-point safety margin on the calibration target. Here the protocol's rule is
applied with targets 0.96 and 0.97 instead of 0.95: the smallest ef in {100, 200, 400, 800} whose mean
calibration recall@100 reaches the target, at each snapshot, from the frozen held-out reports (seeds 11,
29, 47). The test queries' rankings at that ef come from the held-out extension's sweep (incremental arm,
ef 100–3,200), which reproduced the held-out rankings. Reported per HNSW arm: the chosen widths, D (374
queries, GPT-OSS-120B labels, seeds averaged within query, 10,000-draw query bootstrap), test recall per
snapshot, and the mean search time from the cost job at that width. As a check, the 0.95 target must
reproduce the frozen calibrated D exactly.

Known before computing, and stated so that the result is not taken as a test: every Epzero calibration
recall at ef 200 lies between 0.951 and 0.956, so a 0.96 target chooses ef 400 at every snapshot, and D
will be close to the ef-400 value already reported (−0.0026). The analysis records the arm and its cost
as an operator would run it; it is not a prediction.

## 2. The calibration-size study, resampling the calibration queries

`HELDOUT_EXPLORATORY.md` section 8 estimated how often 50 calibration queries keep a too-narrow width by
resampling the *test* queries. Section 18 left undone the same study done by resampling the 50
*calibration* queries, whose per-query recall the seven further seeds recorded at every grid width. For
each of those seeds, both HNSW arms and snapshot 3: draw 25, 50, 100, 200 and 400 queries with replacement
from the 50 (5,000 draws), apply the 0.95 rule, and report how often it keeps ef 200 or less. *Prediction:*
it keeps ef 200 more often than in section 8 at the same size (21–22% at 50), because the calibration
queries reach higher recall than the test queries (section 18). Sizes above 50 resample a 50-query pool
and are reported as such.

## 3. A model-based check of the within-query ratio

The within-query Mantel–Haenszel ratio was chosen after the pooled result was seen. As a check of the
measure, not of the finding: on the full-collection order rerun (Qwen, M 16, ef 200, three seeds per
order, both implementations), fit a logistic regression of miss (0/1) on arrival group (snapshot 2,
snapshot 3 against snapshot 1) and exact-rank band, with a fixed effect per query (queries with no miss
or no hit dropped, as a conditional model would), by Newton's method. Report the odds ratio of snapshot 3
against snapshot 1 per order and implementation, with a 5,000-draw query bootstrap.
*Prediction:* the odds ratios order as the Mantel–Haenszel ratios do (as arrived above 1, shuffled near 1,
reversed below 1), and the as-arrived interval excludes 1 for both implementations.

## Results (26 November 2026)

Module `research/longeval/reanalyses.py`; aggregates `r1-calibration-margin.json`, `r1-calibration-resample.json`,
`r1-fixed-effects.json` in `research/longeval/`.

**1. Margin.** The 0.95 target reproduces the frozen D exactly for both arms (the check passed). A 0.96 target
chooses ef 400 at every snapshot for every seed and both implementations (as known beforehand); a 0.97 target
chooses the same widths.

| Arm, target 0.96 | Gap s1 / s2 / s3 | D [95% interval] | Test recall@100, s1 / s2 / s3 |
| --- | --- | --- | --- |
| Epzero HNSW | −0.0027 / −0.0046 / −0.0053 | −0.0026 [−0.0059, +0.0002] | 96.8 / 96.5 / 96.3% |
| FAISS HNSW | −0.0028 / −0.0041 / −0.0050 | −0.0022 [−0.0052, +0.0005] | 96.8 / 96.5 / 96.3% |

Both intervals lie within ±0.01; the loss is about half of the protocol arms' at every snapshot (Epzero
−0.0057 / −0.0102 / −0.0117, FAISS −0.0048 / −0.0049 / −0.0055).

**2. Resampling the calibration queries.** How often the 0.95 rule keeps ef 200 or less at snapshot 3 (seven
seeds, 5,000 draws each):

| Calibration queries | 25 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- |
| Epzero | 58% | 58% | 60% | 63% | 66% |
| FAISS | 52% | 49% | 47% | 45% | 42% |

**Prediction held**: more often than the 21–22% of section 8 at the same size. Every seed's mean calibration
recall at ef 200 lies between 0.944 and 0.956; for Epzero the pool's mean is above 0.95 on six of seven seeds,
so a larger draw from it keeps the narrow width more often.

**3. Fixed-effects check.** Odds ratio of a snapshot-3 target being missed against a snapshot-1 target, with a
fixed effect per query and per exact-rank band (full-collection order rerun, ef 200, three seeds summed within
query, 5,000-draw query bootstrap):

| Implementation | As arrived | Shuffled | Reversed |
| --- | --- | --- | --- |
| Epzero | 1.77 [1.46, 2.11] | 1.02 [0.87, 1.18] | 0.77 [0.64, 0.92] |
| FAISS | 1.53 [1.27, 1.83] | 1.06 [0.91, 1.22] | 0.78 [0.65, 0.92] |

**Both predictions held**: the odds ratios order as the Mantel–Haenszel ratios do (1.42 / 1.00 / 0.82 and 1.28
/ 1.03 / 0.82), and the as-arrived interval excludes 1 for both. Odds ratios are larger than risk ratios, as
they must be when misses are not rare within a query. The finding does not depend on the post-hoc estimator.
