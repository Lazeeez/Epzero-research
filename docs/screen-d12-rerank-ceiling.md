# Development screen D12: the recall ceiling of the principal-direction walk

Recorded on 24 November 2026, before it was run. Product question, development data only.

## Why

In VIBE on AG News (1024-d), Epzero's recall at k = 100 stopped at 0.988 from ef 400 to ef 1000, whatever
M (12–32) and efConstruction (200, 500): it was not ranked at recall 0.99. The automatic build had chosen
to walk on 192 principal directions and to re-rank the best 300 candidates with full vectors, stopping
early when the rest cannot reach the top k (`rerank` 300, `rerank_z` > 0). A true neighbour that the
walk's codes place below the re-ranked candidates is never re-scored, however wide the search: either the
fixed 300, or the early stop, caps recall.

## Design

On a local dataset for which `epzero.build` chooses a principal-direction walk (from screen D11's recorded
plans; the LongEval sample if it is chosen there), the automatic index (default arm A of D11), searched
with one thread at ef 100–1000, k = 10 and k = 100, under three re-rank settings passed per search:

| Setting | rerank | rerank_z |
| --- | --- | --- |
| default | the plan's (300) | the plan's |
| deep | max(plan's, ef) | the plan's |
| full | ef | 0 (no early stop) |

Measures: recall@10, recall@100, queries per second, distance computations.

## Prediction

At k = 100, the default setting's recall stops rising before ef 1000 (a ceiling), and `deep` lifts it by at
least 0.003 at ef 800–1000; at ef ≤ 200, `deep` costs at most 5% throughput against the default (the early
stop still ends most re-ranks). If `deep` does not lift the ceiling but `full` does, the early stop is the
cause. If neither does, the walk's candidate set itself is the limit.
