# Development screen D6: clean timing of `extra_backlinks`

Recorded on 23 November 2026, before it was run. Development data only.

## Why

Screen D5 measured work (distance computations) reliably but not throughput: builds ran concurrently
and the width grid was coarse. A throughput claim, or making `extra_backlinks` a default, needs
sequential runs on an otherwise idle machine and a finer grid.

## Design

`research/longeval/pareto_screen.py`, one process at a time, single thread, on the 300K SIFT, GloVe
and LongEval samples, as arrived, `extra_backlinks` 0, 4 and 8, seed 11; widths 10, 15, 20, 30, 40,
60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800. Measures: recall@10 and @100, distance
computations, queries per second. Compared at recall@10 0.90, 0.95, 0.98 and 0.99 by interpolation.
The machine is a laptop under WSL; timings are indicative, and any claim about throughput is
repeated on a PACE node before it is made.

## Prediction

At recall@10 0.95 and above, `extra_backlinks` 4 and 8 need no more distance computations than the
baseline and reach at least the baseline's queries per second (within 3%) on all three datasets; at
0.98 and above, LongEval and SIFT need at least 5% less work with 8.

## Result (run 23 November 2026, after the design above was committed)

Sequential, single thread, as arrived, seed 11. Change against `extra_backlinks` 0 at each recall@10
target (work = distance computations; negative work and positive QPS are better). Raw curves:
`Oryx/build/screen-d6/*.json`.

| Data | extra_backlinks | 0.90 work / QPS | 0.95 | 0.98 | 0.99 | Build |
| --- | --- | --- | --- | --- | --- | --- |
| SIFT | 4 | −1.2% / +2.6% | +0.6% / +4.3% | −4.4% / +7.3% | −10.7% / +13.3% | 29 → 29 s |
| SIFT | 8 | +0.2% / +1.6% | +1.2% / +1.6% | −4.8% / +2.6% | −7.3% / +11.6% | 29 → 31 s |
| GloVe | 4 | −3.0% / +3.6% | −2.0% / +5.7% | not reached | not reached | 38 → 40 s |
| GloVe | 8 | −1.1% / +0.1% | +1.5% / −2.3% | not reached | not reached | 38 → 41 s |
| LongEval | 4 | +1.0% / −0.1% | −4.8% / +2.2% | −19.8% / +24.1% | −5.7% / +6.1% | 261 → 267 s |
| LongEval | 8 | −2.3% / +4.2% | −10.2% / +9.3% | −21.3% / +23.2% | −3.1% / +3.1% | 261 → 273 s |

**Predictions, mostly held but not entirely:** throughput at recall 0.95 and above is never more than
3% below the baseline (lowest −2.3%, GloVe with 8), as predicted. "No more work at 0.95 and above"
failed narrowly in three cells (SIFT +0.6% and +1.2%, GloVe with 8 +1.5%, within the interpolation
error). "At least 5% less work at 0.98 and above with 8" held at 0.99 on SIFT (−7.3%) and at 0.98 on
LongEval (−21.3%), but not at SIFT 0.98 (−4.8%) or LongEval 0.99 (−3.1%).

**Reading:** with `extra_backlinks` 4, single-thread throughput is equal or higher at every recall
target on all three datasets (−0.1% to +24%), the arrival excess is roughly halved (screen D3), and
build time rises 0–5%. With 8 the arrival excess falls further (D3, D4) but the throughput gain is
smaller and slightly negative in one cell. **4 is the candidate default**, pending three seeds,
timing on a PACE node, the compressed (SQ8) configuration, and the confirmation readings.
