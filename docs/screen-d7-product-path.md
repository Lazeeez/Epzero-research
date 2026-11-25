# Development screen D7: arrival bias and `extra_backlinks` in Epzero's default build path

Recorded on 23 November 2026, before it was run. Development data only (300K LongEval sample, 381
test queries).

## Why

Screens D3–D6 used a plain configuration (full-precision vectors, no refine pass, every document
inserted one at a time). Epzero's automatic build instead uses 8-bit codes (SQ8), builds the graph
on the codes, and runs a global refine pass after the bulk build. Documents added later are inserted
one at a time with no refine pass. That is where arrival bias would reach users.

## Design

Epzero development build, SQ8 with full vectors kept, `fast_build` on, M 16, efConstruction 100,
seed 11:

- **Grown** (`extra_backlinks` 0 and 4): snapshot-1 documents bulk-built (training the 8-bit ranges on
  them, all cores), refine pass, then snapshot-2 and snapshot-3 documents inserted one at a time in
  arrival order (one thread).
- **Rebuilt** (reference, `extra_backlinks` 0): all documents trained on and bulk-built at once (all
  cores, so no meaningful insertion order), refine pass.

Search: ef 100, 200 and 400, re-ranking the whole beam with full vectors (rerank = ef). Measures: the
newest cohort's within-query excess (snapshot 3 against 1), miss rate per cohort, recall@100 and
recall@10, distance computations, insertion time for snapshots 2 and 3.

## Prediction

Grown without the option shows a newest-cohort excess clearly above the rebuilt reference (by at
least 1 point at ef 200); `extra_backlinks` 4 cuts the grown excess by at least half, with the
oldest cohort's miss rate rising by at most 0.3 points and recall not falling.

## Result (run 23 November 2026, after the design above was committed)

| Mode | extra_backlinks | Newest-cohort excess, ef 100 / 200 / 400 (pp) | Miss rate by cohort, ef 200 | Recall@100 / @10, ef 200 | Distance computations, ef 200 | Time |
| --- | --- | --- | --- | --- | --- | --- |
| rebuilt (reference) | 0 | 0.11 / 0.22 / 0.20 | 4.42 / 5.21 / 5.39% | 0.9522 / 0.9777 | 3,144 | bulk 49 s |
| grown | 0 | 3.29 / 2.27 / 1.07 | 4.06 / 6.80 / 6.96% | 0.9476 / 0.9793 | 3,032 | bulk 24 s + inserts 66 s |
| grown | 4 | 1.96 / 1.44 / 0.57 | 4.22 / 6.34 / 6.41% | 0.9487 / 0.9769 | 3,047 | bulk 24 s + inserts 69 s |

**Prediction 1 held:** documents added after the initial build are missed clearly more often than
after a rebuild (excess 2.27 against 0.22 points at ef 200). This is the arrival bias users of the
default build path would see.

**Prediction 2 failed:** `extra_backlinks` 4 cut the grown excess by 37% at ef 200 (47% at ef 400,
40% at ef 100), not by half; the oldest cohort's miss rate rose 0.16 points (within 0.3); recall@100
rose 0.11 points but **recall@10 fell 0.24 points** (0.16 at ef 400).

**Possible reasons, to test (screen D7b):** in this configuration the graph is built on 8-bit codes
(`fast_build`), so the extra links are chosen, and the farthest link evicted, by approximate
distances; and the bulk-built part was refined, which already equalised it. Tests: `extra_backlinks`
8, and `extra_backlinks` 4 with `fast_build` off.
