# Analysis R2: what the arrival repair does to ranked relevance

Recorded on 26 November 2026, before it was computed. Exploratory; no new runs. It removes the last
limitation of the repair result: it was confirmed by neighbour recall only.

## Data

The full-scale engine arm, run 3 (`LONGEVAL_ENGINE_ARM.md`; Epzero's default build path, all 1,661,900
documents, seeds 11, 29, 47): arms `grown`, `repaired8w` (8 extra links, each later batch refined at
width 100) and `rebuilt`, the 381 test queries' returned IDs at ef 200 (`labels-ef200.npy`). Labels: the
official GPT-OSS-120B grades at snapshot 3 and the local judge's full grades of the judging pool (E1, job
7298863), read as in Analysis C (`order_relevance.py`).

**Known at the time:** the arms' neighbour-level results, and the coverage of their top 10: of the 3,973
distinct (query, document) pairs in the three arms' top 10 over the 374-query panel, 11 have neither an
official nor a local grade. No nDCG, gain or drop share of these arms had been computed.

## Measures (as Analysis C, with the arms in place of the orders)

At snapshot 3, ef 200, top 10, 374-query panel, seeds averaged within query; 10,000-draw paired query
bootstrap:

- nDCG@10 of each arm minus nDCG@10 of exact search, under the official grades and under the full
  grades; the contrasts `repaired8w − grown` and `rebuilt − grown`;
- the share of the exact top 10's relevant documents each arm drops, by arrival group;
- the substitutes: how many per query, officially graded, locally relevant.

The 11 ungraded pairs count as non-relevant, as in every scoring of the study; as a bound, the full-grade
contrast is also computed with each of them at grade 3 (the most favourable to the arm that returned it).

## Predictions

From Analysis C (the order effect is not detectable at the top 10 with 374 queries): the 95% interval of
`repaired8w − grown` lies within ±0.01 under both label sets; the repaired arm drops a smaller share of
the newest arrivals' relevant documents than the grown arm; and the bound moves no contrast by more than
0.002.

## Result (26 November 2026)

`research/longeval/repair_relevance.py`; aggregates `research/longeval/r2-repair-relevance.json`. 374 queries,
snapshot 3, ef 200, top 10, seeds 11, 29, 47 averaged within query.

| Arm | nDCG@10 minus exact: official / full grades | Relevant exact top-10 documents dropped, arrivals of snapshot 1 / 2 / 3 (official) | Substitutes per query; officially graded; locally relevant |
| --- | --- | --- | --- |
| grown | −0.0090 / −0.0067 | 1.7 / 4.6 / 4.3% | 0.34; 26%; 40% |
| repaired8w | −0.0082 / −0.0064 | 1.6 / 3.4 / 2.1% | 0.30; 23%; 32% |
| rebuilt | −0.0065 / −0.0051 | 1.4 / 4.0 / 2.4% | 0.31; 29%; 35% |

Repaired minus grown: +0.0008 [−0.0028, +0.0041] (official), +0.0003 [−0.0033, +0.0035] (full grades), −0.0002
[−0.0036, +0.0025] with the 11 ungraded pairs set to grade 3. Rebuilt minus grown: +0.0025 [−0.0001, +0.0055] and +0.0016
[−0.0005, +0.0042].

**Predictions: all three held.** Both intervals of repaired minus grown lie within ±0.01 (within ±0.004); the
repaired arm drops half as many of the newest arrivals' relevant documents as the grown arm (2.1% against 4.3%
official, 2.2% against 4.5% full grades), fewer than a rebuild (2.4%), with the oldest unchanged (1.6% against
1.7%); the bound moves no contrast by more than 0.0005. So the repair's neighbour-level effect reaches the
relevant documents of the newest arrivals, and costs nothing in mean nDCG@10, which it does not measurably raise.
