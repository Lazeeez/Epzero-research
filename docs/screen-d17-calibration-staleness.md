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
default build path as in screens D13 and D15 (8-bit codes, `fast_build`, M 16, efConstruction 100). Snapshot 1
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

## Result

Run on 24 November 2026, after the design above was committed (Epzero `min-backlinks` at 2d279c0e).
Evaluation queries (281), means over seeds 11, 29 and 47. At snapshot 3 the computed exact neighbours
matched the sample's on every seed (overlap 1.0). Raw results are in `Oryx/build/screen-d17/*.json`.

Calibrated width after each snapshot (the same on every seed): grown 200 / 250 / 250; repaired8w
200 / 200 / 250. The fixed width is therefore 200 throughout.

| Arm | Snapshot | Recall@100, fixed width | Recall@100, recalibrated | Newest excess, fixed | Newest excess, recalibrated | Miss by cohort, fixed |
| --- | --- | --- | --- | --- | --- | --- |
| grown | 1 | 0.9611 | 0.9611 | | | 3.89% |
| | 2 | 0.9517 | 0.9611 | 1.97 pp | 1.52 pp | 4.12 / 6.51% |
| | 3 | 0.9502 | 0.9594 | **2.05 pp** | **1.68 pp** | 3.82 / 6.53 / 6.64% |
| repaired8w | 1 | 0.9621 | 0.9621 | | | 3.79% |
| | 2 | 0.9549 | 0.9549 | −0.19 pp | −0.19 pp | 4.44 / 4.69% |
| | 3 | 0.9535 | 0.9619 | **0.20 pp** | **−0.08 pp** | 4.26 / 5.05 / 5.36% |

At snapshot 2 the newest cohort is snapshot 2, against snapshot 1. At snapshot 3 it is snapshot 3 against 1.

**Predictions:**

1. "grown, fixed width: recall@100 at snapshot 3 below 0.95 by at least 0.5 points": **failed**. It
   fell 1.1 points, from 0.9611 to 0.9502, and ends at the target. The coarse grid made the snapshot-1
   width overshoot (0.961 at 200, against 0.95 sought), and that margin absorbed most of the drift.
2. "grown, fixed width: newest excess at snapshot 3 at least 1.0 point": **held** (2.05; 1.97–2.09 on
   each seed).
3. "grown: the recalibrated width grows": **held** (200 → 250 by snapshot 2, on every seed).
4. "repaired8w, fixed width: smaller recall shortfall and smaller excess than grown": **held** (recall
   0.9535 against 0.9502; excess 0.20 against 2.05).

**Reading:** under the rule fixed beforehand, prediction 1 failed, so the calibration leg as defined
(recall falling below its target) is **not shown** on this sample at this scale. The width
calibrated once still meets the target at snapshot 3, though only because of its starting margin.

What the screen does show is who bears the drift. Keeping the snapshot-1 width costs 1.1 points of
recall by snapshot 3, and the loss is in the documents that arrived later: the snapshot-1 cohort's
miss rate does not rise (3.89 → 3.82%), while the later cohorts' is 6.5–6.6%. Recalibrating restores
the average (0.9594 at width 250) but leaves 1.68 of the 2.05 points of excess. This is the same
conclusion as section 4.7b with the frozen study's own recalibration, now with a fixed width beside it:
calibration on mean recall sees the drift and not who bears it. With the repair, the snapshot-1 width
still meets the target at snapshot 2, and the excess at the fixed width is 0.20 points. One sample,
three seeds, development data. The full-scale version needs per-snapshot exact neighbours and goes
with the next engine-arm package if the third run passes.
