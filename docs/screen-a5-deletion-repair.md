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
