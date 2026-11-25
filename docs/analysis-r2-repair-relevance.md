# Analysis R2: what the arrival repair does to ranked relevance

Recorded on 26 November 2026, before it was computed. Exploratory; no new runs. It removes the last
limitation of the repair result: it was confirmed by neighbour recall only.

## Data

The full-scale engine arm, run 3 (`LONGEVAL_ENGINE_ARM.md`; Epzero's production path, all 1,661,900
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
