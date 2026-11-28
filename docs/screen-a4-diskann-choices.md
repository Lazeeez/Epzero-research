# Screen A4: which DiskANN choice protects late arrivals

Recorded on 28 November 2026, before the driver was changed or any build run. For paper A (`plan-paper-a.md`).
Exploratory, the samples of screen M3.

## Why

In screen M3 the independent FreshDiskANN-style driver showed the arrival bias as strongly as HNSW (as-arrived
within-query miss ratio 1.32 on LongEval, 1.99 on SIFT), while DiskANN's own library hardly showed it on LongEval
(0.99) and less on SIFT (1.43). DiskANN's newest tenth kept 26-28 incoming links against the driver's 17-19. Two
differences from the driver are known: DiskANN lets a list grow past R before pruning it back to R (its saved
graphs reach a degree of 41 = 1.3 × 32), and it starts every search from a frozen point that is not a data vector.

## Design

The driver (`research/longeval/vamana_driver.cpp`) gains two switches, each off by default:

- **slack s:** a reverse link is appended to a neighbour's list without pruning until the list would exceed
  ⌊s·R⌋, and is then pruned back to R with RobustPrune (DiskANN's `GRAPH_SLACK_FACTOR`, 1.3);
- **frozen start:** the graph holds one extra node at the data's mean vector, present before the first insertion,
  from which every search starts; it can be linked to like any node and is never returned or counted.

Variants: `slack` (s 1.3), `frozen`, `both`. R 32, L_build 100, α 1.2, as in M3. Orders as arrived and shuffled,
seed 11, on M3's LongEval sample (300,000) and SIFT sample (300,000): 12 builds. Before them, the changed driver
with both switches off rebuilds M3's SIFT as-arrived seed-11 build, whose returned labels must be identical.

**Measures** (as M3, `vamana_screen.py analyze`): the as-arrived within-query miss ratio of cohort 3 against
cohort 1 at L 200, k 100, with its query-bootstrap interval; the mean in-degree of the first and last insertion
tenths in the shuffled builds; recall@100 at L 200.

## Predictions

1. Slack raises the newest tenth's mean in-degree in the shuffled builds by at least 4 links on both samples
   (from 17.4 and 18.6).
2. Slack lowers the as-arrived ratio below 1.20 on LongEval (from 1.32) and below 1.80 on SIFT (from 1.99).
3. The frozen start alone changes the as-arrived ratio by less than 0.10 and the newest tenth's in-degree by less
   than 1 link, on both samples.
4. With both, LongEval's as-arrived ratio is within 0.15 of DiskANN's 0.99.

A result against a prediction is reported as such. Aggregates only, in `research/longeval/`.

**Check (28 November 2026).** With both switches off, the changed driver rebuilt M3's SIFT as-arrived seed-11
build in 224 s; its returned labels at every width and its in-degrees are byte-identical to M3's.

## Amendment (28 November 2026, after the frozen-start builds, before any frozen-copy build)

**The frozen arms tested a start DiskANN does not use.** DiskANN fills its frozen slot with a copy of the medoid
of the points present when it first needs one (`generate_frozen_point`, which copies `calculate_entry_point()`'s
vector); diskannpy's dynamic index never sets the start point otherwise, so when inserting into an empty index the
frozen node is a copy of the first inserted vector, the driver's own entry, held as a separate node that is never
returned. The arms above put it at the data's mean. On the LongEval sample (unit-normalised vectors) that point is
close to every vector, and α-pruning let it occlude most other candidates: 277,421 of 300,000 nodes linked to it,
29% kept at most two out-links, 27.5% had no incoming link, and recall@100 fell to 0.685 (`frozen`) and 0.699
(`both`); on SIFT the degrees stayed normal. Those arms are reported but do not test DiskANN's frozen start.

**Added arm `frozen-copy`:** the frozen node is a copy of the first inserted vector (FROZEN 2 in the driver),
otherwise as `frozen`; as arrived and shuffled, seed 11, both samples. **Prediction:** its as-arrived ratio is
within 0.10 of the driver's (1.32 LongEval, 1.99 SIFT) and its newest tenth's in-degree within 1 link of the
driver's (17.4, 18.6), on both samples.
