# Screen A5: removing the arrival bias under heavy deletion

Recorded on 28 November 2026, before the arm was added or run. For paper A (`plan-paper-a.md`). Exploratory, the
Big-ANN deletion runbook of screen M2, Part A.

## Why

Under `msturing-10M-clustered`'s deletion runbook (10,000,000 inserted, 5,400,000 deleted), screen M2 measured the
newest cohort's within-query excess at ef 200 as +8.37 points with tombstones (`grown`), +4.22 for the final live
set inserted in the runbook's order without tombstones (`rebuilt`, what compaction gives), and +6.20 with the
repair applied to the tombstoned graph (`repaired`). The two remedies address different causes: compaction removes
the tombstones that keep routing searches towards the old survivors, and the repair gives each batch the
incoming links that only later arrivals would give it. Neither was applied with the other.

## Design

A new arm of `research/longeval/deletion_screen.py`, **`rebuilt-repaired`**: the final live set inserted in the
runbook's order (as `rebuilt`), with `extra_backlinks` 8 and `refine_links(ef=100, first=…)` after each insert
step (as `repaired`). Epzero 0.5.0 from `~/epzero-main-venv` (M2's), seeds 11, 29 and 47, M 16, efConstruction
100, single-threaded inserts, the refine on 6 threads as in M2; M2's prepared runbook, queries and exact live
targets, unchanged. Measured as M2 (`deletion_screen.py analyze`): the newest cohort's within-query excess over the
oldest's at ef 200, seeds averaged within query, 5,000-draw query bootstrap; the miss rate by cohort; recall@100.

## Predictions

1. The newest cohort's excess is below +1.5 points, with the upper end of its interval below +2.5.
2. Recall@100 at ef 200 is at least 3 points above `rebuilt`'s (0.731).
3. The oldest cohort's miss rate is no higher than `rebuilt`'s (20.1%).

A result against a prediction is reported as such. Aggregates only, in `research/longeval/`.

## Result (28 November 2026)

Three builds of `rebuilt-repaired` (Epzero 0.5.0, seeds 11, 29 and 47; 16 min each single-threaded, of which the
refine 3 min on 6 threads), scored with M2's builds by `deletion_screen.py analyze`; aggregates
`research/longeval/a5-deletion-repair.json`. ef 200, 10,000 queries, seeds averaged within query.

| Arm | Newest cohort's excess | Miss rate, oldest / middle / newest third | Recall@100 |
| --- | --- | --- | --- |
| grown (tombstones) | +8.37 [+7.99, +8.74] points | 13.5 / 30.0 / 18.6% | 0.772 |
| rebuilt (live set) | +4.22 [+3.81, +4.62] | 20.1 / 33.2 / 21.6% | 0.731 |
| repaired (tombstones kept) | +6.20 [+5.86, +6.53] | 11.4 / 21.8 / 14.5% | 0.827 |
| **rebuilt-repaired** | **+1.23 [+0.87, +1.60]** | **17.0 / 24.0 / 16.2%** | **0.797** |

**Predictions.** (1) The newest cohort's excess below +1.5 points with the upper end below +2.5: **held** (+1.23,
upper +1.60). (2) Recall@100 at least 3 points above `rebuilt`'s: **held** (0.797, +6.6 points). (3) The oldest
cohort's miss rate no higher than `rebuilt`'s: **held** (−3.12 points, upper bound −2.93). The three seeds agree to
0.1 points in every cohort.

**Reading.** Under heavy deletion the bias has two causes, and each remedy addresses one: compaction removes the
tombstones that route searches towards the old survivors (+8.37 → +4.22), the repair the arrival order of the live
set's links (+4.22 → +1.23). Together they remove 85% of the grown index's excess and raise recall by 2.5 points
over it. Keeping the tombstones and repairing gives the highest recall (0.827, tombstones still route searches)
but leaves three quarters of the bias: the trade-off between recall over the collection and fairness to new data
is a choice an operator can now see.
