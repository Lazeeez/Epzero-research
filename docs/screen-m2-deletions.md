# Screen M2: arrival bias when documents are also deleted

Recorded on 26 November 2026, before any code for it was written or run. Exploratory, public data and
development data only. Every result of the paper is for append-only growth; this screen asks whether
deletions change who bears the misses.

## Why it could matter

Epzero and hnswlib delete by tombstone: a deleted node keeps routing searches but is never returned, and
in Epzero it is never linked to again. So deletions remove targets without removing routes, while new
arrivals can choose only live nodes. Old survivors next to deleted regions may then receive more of the
new arrivals' choices, and the newest documents may be worse off or better off than in a graph that
never held the deleted ones.

## Part A: Big-ANN's deletion runbook

`msturing-10M-clustered` in `neurips23/runbooks/delete_runbook.yaml` of `big-ann-benchmarks` (read as data,
SHA-256 c146fb5c…b48e, fetched 26 November 2026): the same 10,000,000 clustered MSTuring vectors as the
stream study (`BIGANN_STREAM.md`), inserted in 32 contiguous steps, with 10 range deletions of
5,400,000 vectors in all between them; 4,600,000 vectors are live at the end. Queries: the track's
10,000. Local, WSL (`~/bigann-data`), single-threaded builds, one runbook step at a time in file order.

**Arms** (M 16, efConstruction 100, full-precision vectors):

| Arm | Epzero | hnswlib |
| --- | --- | --- |
| grown: the runbook, deletions as tombstones | `remove` | `mark_deleted` |
| rebuilt: only the final live set, inserted in the runbook's order, no deletions | yes | yes |
| repaired: grown with `extra_backlinks` 8 and each insert step refined at width 100 | yes | — |

Seeds 11 (both implementations), 29 and 47 (Epzero). Searched at the end at ef 100, 200 and 400, k 100,
against the exact top 100 of the live set (float32 scan, top 200 re-scored in float64, ties by
position). Arrival cohorts: the third of the 10,000,000 insertions in which a vector's insert step
starts (as in the stream study); only live vectors are targets.

**Measures (ef 200, seeds averaged within query, query bootstrap 5,000 draws):** the newest cohort's
within-query excess miss rate over the oldest's (strata query × exact-rank band) in each arm; the paired
difference grown minus rebuilt; repaired minus grown; recall@100 per arm.

**Predictions.** (1) In the grown arm the newest cohort's excess is positive with its interval excluding
0, for both implementations: the bias survives deletions. (2) The direction of grown minus rebuilt is
the open question; the author's expectation, with low confidence, is that it is positive (deletions make
the newest worse off), because new arrivals cannot link to tombstoned nodes that still route searches.
(3) The repair removes at least half of the grown arm's excess, and the oldest cohort's miss rate rises
by at most 0.75 points (upper 95% bound).

## Part B: LongEval with synthetic deletions

The 300K development sample (`~/epzero-longeval/pilot/sample`), Epzero, seeds 11, 29, 47. Cohort 1 is
inserted in lexical order; before cohort 2 and again before cohort 3 are inserted, a random 10% of the
live documents (seeded) are tombstoned. Arms grown, rebuilt (the final live set in the same order) and
repaired (as above). The 381 test queries, exact top 100 of the live set. Same measures; prediction (1)
and (3) as in Part A, and |grown minus rebuilt| below 1 point (deletions this sparse change little).

Aggregates only; a new module `research/longeval/deletion_screen.py` builds and scores.

## Result, Part B: LongEval with synthetic deletions (26 November 2026)

Nine builds (Epzero 0.5.0 from `~/epzero-main-venv`, seeds 11, 29, 47), each schedule tombstoning 37,927
documents (262,073 live at the end); aggregates `research/longeval/m2-deletions-longeval.json`. ef 200, 381
test queries, seeds averaged within query, 5,000-draw query bootstrap.

| Arm | Newest cohort's within-query excess | Oldest cohort's miss rate | Recall@100 | Build (one thread) |
| --- | --- | --- | --- | --- |
| grown (tombstones) | +1.88 [+1.30, +2.49] points | 4.31% | 0.947 | 455 s |
| rebuilt (final live set, same order) | +1.64 [+1.08, +2.24] | 4.53% | 0.946 | 382 s |
| repaired (8 links, refine at 100) | −0.05 [−0.54, +0.43] | 4.24% | 0.955 | 524 s, of which refine 76 s on 4 threads |

**Predictions: all held.** (1) The grown arm's excess is positive with its interval excluding 0. Rebuilt
minus grown is −0.24 [−0.52, +0.06] points, within the predicted ±1: sparse deletions neither worsen nor relieve
the bias. (3) The repair changes the excess by −1.93 [−2.36, −1.52] points (all of it), with the oldest cohort's
miss rate changing by −0.08 points (upper 95% bound +0.08, against the 0.75 allowed) and recall@100 rising by
0.8 points.

## Result, Part A: Big-ANN's deletion runbook (26 November 2026)

Eleven builds: Epzero 0.5.0 (`~/epzero-main-venv`) grown, rebuilt and repaired with seeds 11, 29, 47; hnswlib
0.8.0 (`~/hnswlib-venv`, the screen X1 build; the package exposes no version attribute) grown and rebuilt with
seed 11. 10,000,000 vectors inserted, 5,400,000 deleted, 4,600,000 live at the end (1.5 / 1.8 / 1.3 million
by cohort); 10,000 queries; ef 200; aggregates `research/longeval/m2-deletions-bigann.json`. Intervals resample
queries (seeds averaged within query); the three Epzero seeds agree to within 0.2 points in every arm.

| Arm | Newest cohort's within-query excess | Miss rate, oldest / middle / newest third | Recall@100 | Build (one thread) |
| --- | --- | --- | --- | --- |
| Epzero grown (tombstones) | +8.37 [+7.99, +8.74] points | 13.5 / 30.0 / 18.6% | 0.772 | 37–44 min |
| Epzero rebuilt (live set only) | +4.22 [+3.81, +4.62] | 20.1 / 33.2 / 21.6% | 0.731 | 11 min |
| Epzero repaired | +6.20 [+5.86, +6.53] | 11.4 / 21.8 / 14.5% | 0.827 | 52 min, of which refine 9 min on 6 threads |
| hnswlib grown (tombstones) | +8.31 [+7.92, +8.69] | 13.7 / 29.7 / 18.5% | 0.773 | 71 min |
| hnswlib rebuilt (live set only) | +4.25 [+3.81, +4.69] | 19.9 / 32.8 / 21.4% | 0.733 | 24 min |

**Predictions.** (1) **Held**, for both implementations: the grown arm's excess is positive, its interval
far from 0. (2) The open question, with the author's low-confidence expectation that grown minus rebuilt is
positive: **positive** — heavy deletion doubles the arrival gap, by +4.15 [+3.83, +4.47] points (Epzero) and
+4.06 [+3.68, +4.44] (hnswlib). Tombstoned nodes keep routing searches, so the grown graph reaches 4 points more
recall than a rebuild of its live set, but the gain goes mostly to the oldest third (20.1% to 13.5% missed) and
little to the newest (21.6% to 18.6%), which cannot link to the deleted nodes around which the old survivors
were built. (3) **Failed**: the repair removes about a quarter of the grown arm's excess (−2.17 [−2.37, −1.97]
points), not half. It is no harm to anyone: the oldest third's miss rate falls by 2.17 points (upper bound
−2.05) and recall@100 rises by 5.5 points, above both the grown and the rebuilt arms. The middle third is the
worst served in every arm (22–33%), rebuilt included, so that comes from where its live vectors lie, not from
the deletion history.

**Reading.** Under append-only growth (Parts A and B of the earlier screens) and sparse deletions (Part B
here), the arrival bias is set by insertion position and the repair removes it. Under heavy deletion with
tombstones, deleted nodes add a second advantage for old documents that the repair, which only adds incoming
links to new ones, closes only in part. Compaction (rebuilding without the tombstones) would remove that
advantage and the recall it brings together.
