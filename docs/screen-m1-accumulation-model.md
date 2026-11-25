# Screen M1: a model of passive accumulation, and how the disadvantage ages

Recorded on 26 November 2026, before any code for it was written or run. Exploratory, development data
and public data only. It answers two open points of the paper: the mechanism (section 10 of the long
manuscript) is described but not predicted, and how fast a late document's disadvantage fades was
inferred from one final-snapshot view (section 9).

## The model

Insert n nodes in a uniformly random order. When node j arrives it chooses k_j out-links among the j − 1
nodes already present. If the choice were neutral with respect to *when* those nodes arrived, a node at
position i < j would be chosen by j with probability k_j / (j − 1), and the expected number of later
arrivals that chose node i by time t would be

  E_i(t) = Σ_{j = i+1}^{t} k_j / (j − 1)  ≈  k̄ ln(t / i).

This is *exposure*: an early node has simply been present for more arrivals. Two further effects can
add to it or remove from it:

- **findability:** early nodes are better connected, so a new node's construction search may find
  them more often than k_j / (j − 1) (a rich-get-richer term);
- **pruning:** a later node j keeps its link to i only until j's own list fills with newer reverse
  links and the heuristic drops it.

Every link j → i with j inserted after i comes from j's own choice at insertion: j's list only
otherwise gains links from nodes newer than j. So a node's final in-degree splits exactly into
**S_i**, links from later nodes (surviving choices), and **R_i**, links from earlier nodes (reverse
links that i's own choices accepted and kept). Screen D2 measured R_i at insertion (12.5 of 12.6) and
the final gradient (28.3 to 13.7), but not this split, and never the choice counts.

## Design

Local, WSL, Epzero 0.5.0 (`~/epzero-dev-venv`), uncompressed, M 16, efConstruction 100, one thread,
one vector per `add` call, so that each node's list can be read right after its own insertion (its
choices; `index.neighbors(label, 0)`).

| Dataset | Folder | Metric | Queries | Orders | Seeds |
| --- | --- | --- | --- | --- | --- |
| LongEval 300K development sample | `~/epzero-longeval/pilot/sample` | ip | 381 test | appended, shuffled | 11, 29 |
| SIFT 300K public sample (random cohorts) | `~/epzero-longeval/pilot/public-sift` | l2 | 1,000 | appended, shuffled | 11, 29 |

Recorded per build (new module `research/longeval/accumulation.py`):

- for every node, its choices k_j and the count C_i of later arrivals that chose it;
- at checkpoints after 10%, 20%, …, 100% of the insertions: every node's in-degree split into S and R,
  and the queries searched at ef 200, k 100, against the exact top 100 *of the nodes inserted so
  far* (float32 scan, top 200 re-scored in float64, ties by position), scored by insertion decile and
  exact-rank band per query.

## Measures and predictions (fixed now)

Shuffled builds test the model (position independent of content); appended builds are reported
beside them.

1. **Exposure.** C_i(n) against E_i(n) by insertion decile. *Prediction:* exposure explains at least
   80% of the first-to-last-decile difference in C (the difference in E is at least 0.8 of the
   difference in C), in both datasets.
2. **Shape.** S_i at the end against ln(n / i) over 100 percentile bins of insertion position, the first
   bin excluded: *prediction:* linear with R² at least 0.95, in both datasets. R_i is roughly flat: its
   first-to-last-decile difference is less than a third of S's.
3. **Aging, in the graph.** The model makes a sharp prediction: the in-degree gap between two nodes
   inserted at positions i₁ < i₂ is k̄′ ln(i₂ / i₁) whatever the later growth, so it never closes. For
   the nodes in the first and fifth tenths of the first half of the build, the gap in mean in-degree at
   checkpoints 50% to 100%: *prediction:* it changes by less than 20% of its value at 50%.
4. **Aging, in search.** At each checkpoint, the within-query miss ratio (Mantel–Haenszel, strata query
   × band) of the newest tenth of the inserted nodes against the first tenth, and of a fixed group (the
   nodes in the fifth tenth of the full build) against the first tenth from checkpoint 50% on.
   *Predictions:* the newest tenth's ratio is above 1.2 at every checkpoint from 30% on; the fixed
   group's ratio at 100% is still more than half of its value at 50% (the disadvantage fades slowly, if
   at all).

A result against a prediction is reported as such. Aggregates only: no LongEval identifiers, texts or
vectors leave the WSL folders.
