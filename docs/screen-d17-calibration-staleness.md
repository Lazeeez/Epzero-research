# Development screen D17: a search width calibrated once, as the collection grows

Recorded on 24 November 2026, before any code for it was written or run. Exploratory, development
data only. It is the calibration leg of the general claim (section 11 of
`docs/arrival-bias-research.md`: "ef calibrated on snapshot 1 and held fixed, recall by cohort at
snapshots 2–3").

## Why

The frozen study recalibrated the search width at every snapshot. Section 4.7b found that
recalibration absorbs the average drift but not the arrival bias, and that calibration grows more
optimistic as the collection grows. Many deployments calibrate once (or tune once by hand) and keep
the width. With a fixed width, two things can happen as documents arrive: overall recall drifts below
its target, and the shortfall falls more on the newest documents. Neither has been measured by arrival
cohort.

## Design

The 300K development sample (snapshots 1, 2 and 3: 156,741 / 81,467 / 61,792 documents), Epzero's
product path as in screens D13 and D15 (8-bit codes, `fast_build`, M 16, efConstruction 100). Snapshot 1
is bulk-built and refined at width 50. Snapshots 2 and 3 are inserted one at a time in arrival order.
Two arms:

- **grown:** no repair;
- **repaired8w:** `extra_backlinks` 8, each later snapshot's insertions refined at width 100 (the
  third engine arm's candidate).

Seeds 11, 29 and 47. The 381 test queries are split once, at random (seed 20261124), into 100
calibration queries and 281 evaluation queries. The calibration queries come from the same distribution
as the evaluation queries, so this measures staleness alone, not the training-against-test gap of the
frozen study.

At the end of each snapshot, exact top-100 neighbours among the documents present are computed
(float32 candidates, re-scored in float64, ties by sample position; at snapshot 3 they must match the
sample's `exact-top100.npy`). The search width is the smallest of 50, 60, 80, 100, 120, 150, 200, 250,
300, 400, 500, 600 and 800 at which the calibration queries reach recall@100 0.95 (k 100, re-rank of the
whole beam).

- **fixed:** the width calibrated after snapshot 1, kept for snapshots 2 and 3.
- **recalibrated:** recalibrated at each snapshot (for comparison).

Measures, on the evaluation queries at each snapshot: recall@100 at the fixed and recalibrated widths;
the miss rate by arrival cohort and the newest cohort's within-query excess (snapshot 3 against 1, MH
risk difference) at the fixed width; and the recalibrated width itself.

## Predictions

Averaged over the three seeds:

1. **grown, fixed width:** recall@100 at snapshot 3 is below 0.95 by at least 0.5 points.
2. **grown, fixed width:** at snapshot 3 the newest cohort's excess is positive, at least 1.0 point.
3. **grown:** the recalibrated width at snapshot 3 is larger than at snapshot 1.
4. **repaired8w, fixed width:** at snapshot 3 the recall shortfall and the newest cohort's excess are
   both smaller than grown's.

If prediction 1 fails, a width calibrated once holds up over this growth, and the calibration leg is
reported as not shown on this collection at this scale.
