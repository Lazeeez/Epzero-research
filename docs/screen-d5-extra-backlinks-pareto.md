# Development screen D5: is `extra_backlinks` a better recall/work trade-off overall?

Recorded on 23 November 2026, before it was run. Development data only.

## Question

Screens D3–D4 showed that `extra_backlinks` raises recall at a fixed search width for slightly more
work. Benchmarks compare indexes at matched work, along the whole recall–throughput curve. If the
option moves Epzero's curve up across the range, it is a product improvement and a candidate default,
not only a fix for arrival bias.

## Design

Epzero development build, uncompressed, M 16, efConstruction 100, one thread, seed 11, on the 300K
SIFT, GloVe and LongEval samples, as arrived and shuffled, `extra_backlinks` 0 and 8. Each build is
searched at ef 10, 20, 40, 80, 120, 200, 300, 400, 600 and 800, k = 10 and k = 100. Measures per
width: recall@10, recall@100, distance computations per query (from traces) and single-thread
queries per second. Comparison: recall at matched distance computations, by linear interpolation
between widths, at recall@10 targets 0.90, 0.95 and 0.98 (the work needed to reach each target).

## Prediction

With `extra_backlinks` 8, the work needed to reach recall@10 of 0.95 and 0.98 is at most that of
the baseline on all three datasets, as arrived and shuffled (the curve does not get worse anywhere
above recall 0.9), and is lower by at least 3% on at least two of the three datasets as arrived.

## Result (run 23 November 2026, after the design above was committed)

Change in the distance computations needed to reach a recall target, `extra_backlinks` 8 against 0
(negative = less work), linear interpolation between the measured widths; "n/a" where neither build
reaches the target. Raw curves: `Oryx/build/screen-d5/*.json`.

| Data | Order | recall@10 0.90 | 0.95 | 0.98 | 0.99 | recall@100 0.95 |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | as arrived | +0.4% | −0.1% | −6.3% | −10.1% | +5.1%* |
| SIFT | shuffled | +0.9% | −1.2% | −1.0% | −4.8% | +4.8%* |
| GloVe | as arrived | −1.8% | +1.0% | n/a | n/a | n/a |
| GloVe | shuffled | −4.2% | −6.2% | n/a | n/a | n/a |
| LongEval | as arrived | −3.3% | −7.5% | −21.4% | −1.0% | −4.4% |
| LongEval | shuffled | +3.1% | +0.1% | −15.6% | n/a | −1.9% |

\* Both builds already exceed the target at the smallest width searched for top-100 (ef 100), so this
is the extra work per step at equal width, not at equal recall.

**Against the prediction, narrowly:** the work to reach recall@10 0.95 was not lower everywhere: +1.0%
on GloVe as arrived and +0.1% on LongEval shuffled. The second part (at least 3% lower on two of three
datasets as arrived) held at 0.98 (SIFT −6.3%, LongEval −21.4%).

**Reading:** `extra_backlinks` is not a uniform speed-up. Each step scores slightly more neighbours
(about 1–5% more work at equal width), which costs a little at low recall and is repaid at high
recall (0.98 and above), where late, weakly linked documents are what is left to find. It is
neutral to slightly positive overall and clearly positive at high recall, with a large arrival-bias
benefit.

**Limits:** the width grid is coarse (interpolation error of a few percent); queries per second were
measured with builds running concurrently and are not used; one seed. A clean measurement (finer
grid, `extra_backlinks` 0, 4 and 8, sequential timing on an otherwise idle machine) is needed before
any throughput claim or a change of default.
