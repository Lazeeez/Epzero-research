# Screen A2: CleanANN's bridges against the arrival bias

Recorded on 28 November 2026, after CleanANN's code was read and before any CleanANN build. For paper A
(`plan-paper-a.md`). Exploratory, the samples of screens M3 and A4.

## Why

CleanANN (Zhang, Wei, Engels and Shun, SPAA 2026) reports that FreshVamana's insertion is not robust to
insertion order and repairs it with GuidedBridgeBuild: during each insertion's search, nodes of the search tree
at the same depth within a depth window are linked to each other. It measures recall per round, not which
vectors bear the misses. The question here is whether its bridges remove the arrival bias, against the repair
used in this study and against the list slack that screen A4 found protects late arrivals.

## The implementation

The authors' code, `github.com/SylviaZiyuZhang/CleANN` at commit `6bee643` (16 September 2025; MIT), a fork of
DiskANN, built in its own environment (`~/cleann`, Python 3.11, numpy 1.25.0 as its build pins) with
`-include cstdint` added to the compiler flags (its `ann_exception.h` uses `uint32_t` without including it; GCC
13 rejects that). Nothing in its source is changed. GuidedBridgeBuild is compiled in (`LAYER_BASED_PATH_COMPRESSION`,
on by default) and governed at run time by depth windows: a start node's depth in [start_lb, start_hb), an end
node's in [end_lb, end_hb), and the two depths equal. Read from the code: the defaults in
`concurrent_experiments/utils.py` (start [3, 5), end [9, 64)) cannot satisfy the equal-depth condition and add no
bridge; the experiments' settings B9 ([19, 22), [20, 64)) and B10 ([20, 64), [20, 64)) do.

## Arms

`backend cleann` in `research/longeval/vamana_screen.py`, M3's DiskANN arm otherwise unchanged (R 32, L 100, α
1.2, one insertion thread, the frozen point, DiskANN's 1.3 slack, which CleanANN inherits):

- **bridges on:** start and end depths [20, 64) (B10), cleaning threshold 7 (no deletions here, so unused);
- **bridges off:** start [3, 5), end [9, 64), the same binary adding no bridge.

Orders as arrived and shuffled, seed 11, on M3's LongEval sample and SIFT sample: 8 builds. Measures as M3: the
as-arrived within-query miss ratio of cohort 3 against cohort 1 at L 200, k 100; the in-degree of the first and
last insertion tenths in shuffled builds; recall@100 at L 200; build seconds.

## Predictions

1. With bridges off, CleanANN matches DiskANN's library within 0.15 in the as-arrived ratio on both samples
   (DiskANN: 0.99 on LongEval, 1.43 on SIFT).
2. Bridges change the as-arrived ratio by less than 0.15 on both samples: they are not aimed at late arrivals,
   and the slack already protects them.
3. Bridges raise build time by at least 20%.

A result against a prediction is reported as such. Aggregates only, in `research/longeval/`.

## Amendments (28 November 2026, from checks on synthetic data, before any A2 build)

- **Start.** CleanANN's index refuses inserts until it has been built (`assert(_has_built)` in `insert_point`);
  its experiments bulk-build a first batch (`_index.build`). Here the first 1% of each insertion order (3,000
  vectors) is bulk-built and the rest inserted one at a time, identically for both arms. M3's DiskANN arm
  inserted everything from empty; prediction 1 compares across that difference.
- **The bridges-on window.** The paper's own rule, both depth windows [⌊log₂|D|⌋ + 2, ⌊log₂|D|⌋ + 5), which is
  [20, 23) for 300,000 vectors, replaces B10 as the primary arm. Checked on 30,000 random 64-dimensional vectors
  (inserted after a 300-vector bulk start, L 100, R 32): with the window at the same rule ([16, 19)), recall@10
  rose from 0.150 / 0.243 / 0.361 to 0.185 / 0.302 / 0.426 at L 10 / 20 / 40, and the build took 1.21× as long;
  B10's window also changed results but less. A shallow window ([3, 12)) doubled recall at 7.5× the build time,
  so the bridge code changes the graph whenever its window is reached.
- **No in-degree.** `save()` fails an assertion in `reposition_points` after inserts, so the graph cannot be read
  back; the in-degree measure is dropped for A2. The miss ratios and recall come from searches, as for every arm.

## Amendment 2 (28 November 2026, after the paper-window builds and a depth probe, before any shallow-window build)

**The paper's window hardly fires here.** With both windows at [⌊log₂|D|⌋ + 2, + 5) = [20, 23), CleanANN's builds
on the 300,000-vector samples gave the same build time as without bridges and nearly the same recall (SIFT as
arrived 1.50 against 1.47; LongEval 1.011 against 1.011, recall@100 0.9630 against 0.9629): the insertions' search
trees rarely reach depth 20.

**Depth probe** (first 50,000 SIFT vectors, the sample's first 500 queries, 1% bulk start, R 32, L 100, α 1.2;
recall@10 at L 10 / 20; `~/a2/probe`):

| Window | Build vs off | Recall@10 | Results identical to off (L 10) |
| --- | --- | --- | --- |
| off | 1.00× | 0.888 / 0.962 | |
| [16, 19) (the paper's rule at 50,000) | 0.99× | 0.887 / 0.963 | 96% |
| [12, 15) | 1.08× | 0.879 / 0.959 | 74% |
| [8, 11) | 2.82× | 0.881 / 0.958 | 55% |
| [4, 7) | 3.60× | 0.879 / 0.958 | 58% |

Bridges that act lower SIFT's recall by about a point here (on random 64-dimensional vectors they raised it).

**Added arm `shallow`:** both windows [⌊log₂|D|⌋ − 4, ⌊log₂|D|⌋ − 1), the probe's [12, 15) scaled to the samples'
size, which is [14, 17) for 300,000; as arrived and shuffled, seed 11, both samples. **Predictions:** (a) the
as-arrived ratio changes by less than 0.15 from bridges off on both samples; (b) recall@100 at L 200 is within 0.5
points of bridges off; (c) the build takes at most 1.5× as long.
