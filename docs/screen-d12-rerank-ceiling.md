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

## Result (run 24 November 2026, after the design above was committed)

On the LongEval 300K sample the automatic build chose 8-bit codes, a walk on 448 principal directions,
`rerank` 300, `rerank_z` 3, refine and free-slot filling (build 634 s, one thread). Raw:
`BUILD_DIR/screen-d12/sample.json`. One thread; QPS from one timed pass per cell.

| ef | k = 100 recall: default / deep / full | k = 100 QPS: default / deep / full | k = 10 recall: default / deep / full | k = 10 QPS: default / deep / full |
| --- | --- | --- | --- | --- |
| 200 | 0.9505 / 0.9505 / 0.9505 | 2934 / 2813 / 2740 | 0.9748 / 0.9748 / 0.9761 | 3821 / 3840 / 2815 |
| 300 | 0.9704 / 0.9704 / 0.9704 | 2083 / 2038 / 1934 | 0.9829 / 0.9829 / 0.9840 | 2746 / 2897 / 2003 |
| 400 | 0.9763 / 0.9780 / 0.9782 | 1745 / 1637 / 1422 | 0.9861 / 0.9861 / 0.9874 | 2175 / 2207 / 1507 |
| 600 | 0.9828 / 0.9854 / 0.9859 | 1305 / 1216 / 1038 | 0.9892 / 0.9892 / 0.9908 | 1605 / 1614 / 1035 |
| 800 | 0.9861 / 0.9889 / 0.9894 | 970 / 989 / 791 | 0.9934 / 0.9934 / 0.9950 | 1201 / 1170 / 785 |
| 1000 | 0.9878 / 0.9908 / 0.9914 | 874 / 866 / 637 | 0.9945 / 0.9945 / 0.9963 | 960 / 991 / 640 |

**Predictions:**
- "Default recall@100 stops rising before ef 1000": **partly**. It flattens (+0.0017 from ef 800 to
  1000, never reaching 0.99) but does not stop.
- "`deep` lifts it by at least 0.003 at ef 800–1000": **held at ef 1000 (+0.0030), narrowly missed at
  800 (+0.0028)**.
- "At ef ≤ 200, `deep` costs at most 5%": held trivially. With ef ≤ 300 deep is the same setting as the
  default (300 ≥ ef); the measured −4% to +4% is timing noise.
- Diagnosis: `full` adds only 0.0005–0.0006 over `deep` at k = 100, so **the fixed depth of 300 is the
  cap, not the early stop**. The walk's candidates contain the missing neighbours; they are not re-scored.

**Reading:** at k = 100, re-scoring at least ef candidates makes recall@100 0.99 reachable (about ef 900)
where the default never gets there, and at ef 800–1000 it costs nothing measurable (989 against 970,
866 against 874): on the recall–speed curve, deep at ef 800 (0.9889, 989/s) beats the default at ef 1000
(0.9878, 874/s). At k = 10 the early stop ends the re-rank before the extra depth is used: deep is the
same recall as the default at the same speed. `full` (no early stop) is never worth it: at k = 10 it
costs 25–35% throughput for +0.001–0.002 recall.

**Candidate product change:** when the plan chose the re-rank depth (a walk on principal directions),
re-rank `max(rerank, ef)` candidates with the early stop. Before it becomes a default: AG News, where
the VIBE ceiling was seen, and the other VIBE walk datasets on a PACE Gold 6226 against Glass, three
seeds, the whole recall range, k = 10 and k = 100.
