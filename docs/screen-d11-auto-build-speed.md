# Development screen D11: extra links and the refine pass in Epzero's automatic build

Recorded on 24 November 2026, before it was run. Product question, development data only.

## Why

In VIBE, Epzero's automatic build (`epzero.build`) took 2–3× Glass's build time (single-threaded) and was
behind the leaders at lower recall on two of three datasets. The automatic build ends with a refine pass
over every node. Screens D3–D6 showed that `extra_backlinks` raises recall and, at high recall,
throughput, for 0–5% build time. Two questions: does `extra_backlinks` 4 improve the automatic build's
speed at equal recall, and can it replace the refine pass to cut build time without losing speed?

## Design

`epzero.build` on the development branch (all other choices automatic: 8-bit codes, building on the
codes, principal directions if it picks them, re-rank, calibration), one thread for building and for
search, one process at a time on the laptop, on the 300K SIFT (L2), GloVe (inner product) and LongEval
(inner product, 2,560-d) samples, inserted in sample order. Arms:

| Arm | refine | extra_backlinks |
| --- | --- | --- |
| A (today's default) | on | 0 |
| B | on | 4 |
| C | off | 4 |
| D | off | 0 |
| E | off | 8 |

Measures: build time (the whole `epzero.build` call); then a sweep of 17 search widths at k = 10 and k = 100
with the automatic re-rank: recall, single-thread queries per second, distance computations. Compared
with arm A at recall@10 0.90, 0.95, 0.98 and 0.99 by interpolation.

## Prediction

- **B against A:** throughput at recall@10 ≥ 0.95 at least equal (within 3%), and higher by at least
  5% at 0.98 or 0.99 on at least two datasets; build time at most 10% longer.
- **C against A:** build time at most 75% of A's, and throughput at recall@10 0.95 within 5% of A's on
  all three datasets.
- **D** is slower than A at recall@10 ≥ 0.95 (the refine pass matters without extra links).

A setting is proposed as a default only if it also holds on PACE against Glass on the VIBE datasets.

## Result (run 24 November 2026, after the design above was committed)

The automatic build chose 8-bit codes, building on the codes, and refine in every case; no principal-
direction walk on SIFT and GloVe; on LongEval a walk on 448 directions, re-rank 300 and free-slot filling.
Change against arm A (today's default) in single-thread queries per second at recall@10 0.90 / 0.95 / 0.98
/ 0.99, by interpolation ("—" where the arm does not reach the target), and in build time. Raw:
`Oryx/build/screen-d11/*.json`.

| Arm | SIFT build | SIFT QPS | GloVe build | GloVe QPS | LongEval build | LongEval QPS |
| --- | --- | --- | --- | --- | --- | --- |
| A (default) | 60 s | — | 79 s | — | 629 s | — |
| B refine + 4 links | +3% | −11 / −6 / −11 / −7% | +3% | +4 / +5 / +10% / — | +2% | +2 / +1 / +1 / +4% |
| C no refine + 4 links | −33% | −10 / −10 / −14 / −11% | −37% | −8 / −14% / — / — | −34% | −9 / −10 / −6 / +5% |
| D no refine | −37% | −10 / −19 / −17 / −18% | −40% | −12 / −14% / — / — | −36% | −3 / −10 / +2 / −13% |
| E no refine + 8 links | −32% | −12 / −9 / −13 / −10% | −35% | −12 / −13% / — / — | −33% | −8 / −15 / −1 / −9% |

**Predictions:**
- "B at least equal at ≥ 0.95 and ≥ 5% faster at 0.98/0.99 on two datasets": **failed**. B is 6–11%
  slower on SIFT, 4–10% faster on GloVe, and within noise on LongEval.
- "C builds in ≤ 75% of A's time with QPS within 5% at 0.95": **failed on speed**. It builds in 63–66%
  of the time, but is 10–14% slower at 0.95.
- "D slower than A at ≥ 0.95": held (10–19% slower at 0.95).

**Reading:** in the automatic build (8-bit codes, the refine pass), `extra_backlinks` is not a general
speed-up, unlike in the plain configuration of screen D6. The refine pass already reconnects every node
from the finished graph, so extra links add little after it and on SIFT cost more per step. Without the
refine pass, extra links recover only part of the loss: they cannot replace it. **`extra_backlinks` stays
opt-in**, as the repair for indexes that grow after their build (screens D7–D10, the engine arm), where
later insertions are never refined. Cutting build time needs another approach, such as a cheaper refine
pass.

Also seen: on LongEval the automatic index's recall@100 stops at 0.986 even at ef 800, a ceiling like
AG News's 0.988 in VIBE (screen D12).
