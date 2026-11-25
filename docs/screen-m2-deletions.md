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
