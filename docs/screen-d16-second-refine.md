# Development screen D16: a second refine pass in a one-off build

Recorded on 24 November 2026, before it was run. Product question, development data only.

## Why

In screen D14, refining every node again after each snapshot raised recall@100 at ef 200 from
0.9506 to 0.9552 on the LongEval 300K sample, above a single-pass rebuild's 0.9520. It also cost 4%
more distance computations. At a fixed width, more work alone raises recall, so that is not yet a
gain. The question is whether a second pass gives a better graph: higher throughput at equal recall.
Screen D15 found that the refine's width mattered for new batches, so a single wider pass is tested
too. The automatic build's own note (`engine/src/auto.cpp`) says half the beam gives the same graph
on GloVe.

## Design

A one-off build of each 300K sample: SIFT (L2), GloVe (inner product) and LongEval (inner product,
2,560-d), in sample order. Product settings without the principal-direction walk: 8-bit codes, vectors
kept, `fast_build`, M 16, efConstruction 100. All vectors are added on all cores, then refined (all
cores). Epzero branch `min-backlinks`; no engine change. Arms:

| Arm | Refine after the build |
| --- | --- |
| one (today's, without the walk) | one pass at width 50 |
| two | two passes at width 50 |
| wide | one pass at width 100 |

Measures: build time, split into adding and refining. Then a sweep of 17 search widths at k = 10 and
k = 100 (re-rank of the whole beam with the full vectors, one thread): recall, queries per second and
distance computations. Compared with `one` at recall@10 0.90, 0.95, 0.98 and 0.99 by interpolation,
in queries per second and in distance computations. One build per arm and dataset (seed 11). Each
arm's sweep is timed in one process, one process at a time on the laptop, nothing else running.

## Predictions

- **two against one:** at least 5% fewer distance computations at recall@10 0.98 on at least two of
  the three datasets, and no worse than 2% more at any target; queries per second in the same
  direction. Refining takes at most 2.2× as long.
- **wide against one:** within 3% in distance computations at every target on all three (the
  automatic build's note).

## Decision

If `two` meets its prediction, a `refine_passes` option for `epzero.build` is designed next. As a
default it goes through a PACE round against Glass on the VIBE datasets first, as round 11 does for
the re-rank depth. If it does not, the D14 recall rise was the extra work, and nothing changes.

## Result

Run on 24 November 2026, after the design above was committed (Epzero `min-backlinks` at 2d279c0e).
Seed 11, one build per arm and dataset, searched with one thread. Raw results are in
`Oryx/build/screen-d16/*.json`. Change against `one`, by interpolation at recall@10 0.90 / 0.95 / 0.98 /
0.99 ("—" where `one` does not reach the target within the 17 widths):

| Dataset | Arm | Distance computations | Queries per second | Build (add + refine) |
| --- | --- | --- | --- | --- |
| SIFT | two | +0.9 / −1.4 / −1.2 / −1.5% | −1.1 / −5.5 / +7.5 / +3.3% | 5.2 → 7.5 s |
| | **wide** | −0.2 / −1.9 / −3.8 / −1.2% | **+6.0 / +5.3 / +18.1 / +11.0%** | 5.2 → 7.4 s |
| GloVe | two | −1.9 / −6.1% / — / — | +5.3 / +8.0% / — / — | 7.0 → 8.7 s |
| | **wide** | −4.1 / −8.0% / — / — | **+7.7 / +13.5% / — / —** | 7.0 → 8.9 s |
| LongEval | two | −3.6 / −7.2 / −13.3 / +6.1% | +1.1 / +9.3 / +9.5 / −8.7% | 47 → 68 s |
| | **wide** | +3.3 / +1.2 / −19.4 / −7.2% | −3.9 / +8.3 / **+26.1 / +11.7%** | 47 → 68 s |

On GloVe, `one` peaks at recall@10 0.9759 within the sweep, and `two` and `wide` reach 0.98 (0.9807 and
0.9805). Recall@100 at ef 200: SIFT 0.9880 / 0.9889 / 0.9906, GloVe 0.8123 / 0.8184 / 0.8250, LongEval
0.9519 / 0.9537 / 0.9583 (one / two / wide).

**Predictions:**

- **two:** "at least 5% fewer distance computations at 0.98 on two datasets, and never more than 2%
  more": **failed**. LongEval −13.3% at 0.98, SIFT only −1.2%, GloVe not comparable at 0.98 (`one`
  does not reach it), and LongEval +6.1% at 0.99. "Refining at most 2.2× as long": held (1.8–2.1×).
- **wide:** "within 3% everywhere" (the automatic build's note): **failed, in its favour**. At 0.98 it
  needs 3.8% fewer distance computations on SIFT and 19.4% fewer on LongEval, and it is 18% and 26%
  faster there. On GloVe it is 8% cheaper and 13.5% faster at 0.95, and it reaches 0.98. At 0.90 on
  LongEval it is 3–4% worse.

**Reading:** one pass at the full construction width builds a better graph than one or two passes at
half of it, as D15 found for new batches. The gain is in the high-recall range, where the leaders are
compared. The automatic build already uses the full width when it picks the principal-direction walk.
Without the walk (SIFT- and GloVe-like data), it halves the width; its note says that gives the same
graph (GloVe-100 at 200K, recall within 0.001 at every ef). This screen, on GloVe at 300K with 8-bit
codes, finds that the wider pass reaches higher recall at the top widths. The cost is build time: the
refine takes about twice as long, the whole build 35–45% longer here.

The QPS figures come from one build each, timed on a laptop. A replication comes before any change.

## Replication (designed 24 November 2026, after the result above, before it was run)

Seeds 29 and 47, arms `one` and `wide`, all three datasets, as above.

**Prediction:** on every seed and dataset, `wide` needs fewer distance computations than `one` at
recall@10 0.95, and at 0.98 where `one` reaches it, and is faster at 0.98 on SIFT and LongEval.

**Decision:** if the prediction holds, the engine change is for the automatic build to refine at the full
construction width without the walk as well (`engine/src/auto.cpp`). It goes to a PACE round against
Glass on the VIBE datasets before it becomes the default, as round 11 does for the re-rank depth,
with the build-time cost reported beside the speed gain. If it does not hold, nothing changes.

## Correction (24 November 2026, while the replication was running, before its results were seen)

The reading and the replication's decision above assumed that D16's `one` arm is the automatic
build's setting. It is not. `epzero.build` defaults to efConstruction **200** (`Plan.ef_construction`,
`AutoOptions::ef_construction`). Without the walk it refines at max(200 / 2, 2M) = **100**, which is
D16's `wide` width. D16, like the engine arms, builds at efConstruction 100, where half the width is
50. So:

- What D16 shows: at efConstruction 100, a refine at width 100 beats one at 50 by up to 18–26% at
  recall@10 0.98. A refine width of about 100 matters, whatever the construction width. It does not
  show that the automatic build's halving is wrong. At efConstruction 200 the halving gives 100.
- The automatic build's note (half of 200 gives the same graph as 200, GloVe-100 at 200K) is not
  contradicted. D16 tested 50 against 100, not 100 against 200.
- The replication's engine change is withdrawn. The replication still runs, as a check of the width
  effect at efConstruction 100. That effect matters for the research product path, the engine arms,
  and screen D15's choice of width 100 for new batches.
- A product-default check comes next: at efConstruction 200, refine at 100 (today's) against 200,
  on the three samples, as screen D16b (designed in its own section before it runs).

## D16b: refine width at the automatic build's defaults (designed 24 November 2026, before it was run)

**Design.** As D16 (8-bit codes, vectors kept, `fast_build`, M 16, no walk, all cores), at
efConstruction **200**, `epzero.build`'s default. Arms: `half`, refine at width 100 (today's), and
`full`, refine at width 200. Seed 11, the three 300K samples. Same sweep and measures as D16.

**Prediction (the automatic build's note):** `full` is within 3% of `half` in distance computations at
every recall@10 target on all three, and its refine takes about twice as long.

**Decision:** if `full` needs at least 5% fewer distance computations at recall@10 0.98 on at least
two datasets, with no target worse by more than 2%, it goes to a PACE round against Glass on the
VIBE datasets as a candidate default, with its build-time cost. Otherwise nothing changes, and the
note stands with this measurement added.
