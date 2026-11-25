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
