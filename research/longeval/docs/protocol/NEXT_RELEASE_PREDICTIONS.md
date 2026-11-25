# Predictions for the next LongEval-Sci release, fixed before it exists

Recorded on 26 November 2026. No 2027 LongEval release, schedule or call had been published when this was
written (searched the same day). Nothing here changes `FINAL_PROTOCOL_V1` or `EXTENSION_PROTOCOL_V1`;
it is a forward test of their findings on data that did not exist when they were made, and the only
answer to the objection that every exploratory analysis reused the 2026 test queries.

## Scope

The predictions apply to the next LongEval scientific-search release that (a) adds at least one
snapshot that only appends to the 2026 collection, or a new collection built the same way, (b) comes with
organiser document vectors or a released encoder configuration, and (c) has new test queries. If the
release breaks (a), only predictions 1 and 2 are tested, on whatever growth it has. Every analysis
uses the code paths of the 2026 study unchanged (M 16, efConstruction 100, ef grid {100, 200, 400, 800},
calibration target 0.95 on the 2026 calibration queries, three seeds 11, 29, 47), the newest snapshot
in place of snapshot 3, the first in place of snapshot 1, and the new test queries.

## Predictions

1. **Index history.** An incremental index and its identical-history rebuild are identical in every
   comparison (state fingerprints and every ranking).
2. **Arrival.** At ef 200, in both HNSW implementations, documents of the newest arrival group are
   missed at least 1.10 times as often as first-snapshot documents by the same query (within-query
   Mantel–Haenszel ratio, strata query × exact-rank band), with the 95% query-bootstrap interval above 1;
   rebuilt with the arrival blocks reversed, the ratio is below 1.
3. **Inverse for IVF.** Frozen IVF's newest-group excess miss rate is below 1 percentage point.
4. **The gap's change.** For calibrated FAISS HNSW and frozen IVF, D (newest minus first snapshot) has
   its 95% interval within ±0.01. For Epzero, the point estimate is within ±0.006. A calibration margin
   (target 0.96) keeps all three intervals within ±0.01.
5. **Concentration.** At the newest snapshot, at least 75% of queries lose nothing to approximate search
   at nDCG@10, and the worst 5% of queries carry at least half of the loss.
6. **The repair.** Epzero's default build path with `extra_backlinks` 8 and each new batch refined at width
   100 removes at least half of the newest group's within-query excess, with the oldest group's miss rate
   rising by at most 0.75 points and recall@100 falling by at most 0.5 (upper 95% bounds).
7. **Calibration optimism.** At the calibrated width, the calibration queries' mean recall exceeds the new
   test queries' by more than at the first snapshot.

Each prediction is reported as held or failed; a failed one is not re-run with other settings.
