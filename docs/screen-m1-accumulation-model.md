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

## Result (26 November 2026)

Eight builds (`research/longeval/accumulation.py`; aggregates `research/longeval/m1-accumulation.json`), Epzero
0.5.0, all completed. In the appended builds the two seeds give near-identical graphs (Epzero's seed sets
only the level draws). Per insertion tenth, shuffled builds, mean over the two seeds:

| | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th | 8th | 9th | 10th |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LongEval: chosen by later arrivals, C | 40.7 | 23.7 | 17.0 | 12.8 | 9.7 | 7.2 | 5.2 | 3.5 | 1.9 | 0.62 |
| LongEval: exposure prediction, E | 40.3 | 23.6 | 17.1 | 12.9 | 9.8 | 7.3 | 5.3 | 3.5 | 2.0 | 0.63 |
| LongEval: links from later nodes kept, S | 26.7 | 20.2 | 15.8 | 12.4 | 9.6 | 7.2 | 5.2 | 3.5 | 1.9 | 0.62 |
| LongEval: links from earlier nodes, R | 2.7 | 5.2 | 6.8 | 8.0 | 9.1 | 9.9 | 10.6 | 11.1 | 11.5 | 11.8 |
| SIFT: C | 38.3 | 23.0 | 16.9 | 12.8 | 9.7 | 7.3 | 5.2 | 3.5 | 2.0 | 0.63 |
| SIFT: E | 38.0 | 23.0 | 16.9 | 12.8 | 9.8 | 7.3 | 5.3 | 3.5 | 2.0 | 0.64 |
| SIFT: S | 26.0 | 20.0 | 15.8 | 12.5 | 9.6 | 7.3 | 5.2 | 3.5 | 2.0 | 0.63 |
| SIFT: R | 2.6 | 5.3 | 7.0 | 8.3 | 9.4 | 10.2 | 10.9 | 11.4 | 11.8 | 12.1 |

Each insertion chooses 12.2 (LongEval) or 11.9 (SIFT) neighbours on average.

**Predictions:**

1. **Exposure explains the choices — held, 8 of 8.** Exposure accounts for 99% of the first-to-last-tenth
   difference in C in shuffled builds (0.989–0.993) and 102–108% in appended ones, and matches C to within
   about 1% in every tenth of the shuffled builds. A new node chooses an existing node with probability close
   to k/(j − 1) whatever its age: there is no measurable rich-get-richer term in construction, only exposure.
2. **S is linear in ln(n/i) — held in 3 of 4 shuffled builds** (R² 0.957, 0.957, 0.9505; SIFT seed 29: 0.9496,
   just below 0.95), not in appended ones (0.93, where content order bends the curve). Slope 8.5–8.7 links per
   unit of ln(n/i), below the 12 chosen because pruning removes a third of the oldest tenth's incoming links
   (66–69% survive) and almost none of the rest's. **R is flat — failed, 0 of 8:** R rises from 2.6–3.2 in the
   first tenth to 11.7–12.1 in the last. A late node keeps nearly all of its own reverse links; an early node
   loses most of them as the neighbours it chose fill up with later arrivals.
3. **The in-degree gap never closes — failed, 0 of 8.** Between the first and fifth twentieths, the gap in
   mean in-degree shrinks by 26–37% from the 50% checkpoint to the end (LongEval shuffled 10.4 → 7.7). It
   closes, slowly, because the oldest nodes' incoming links are pruned.
4. **The newest are always disadvantaged — held, 8 of 8:** the newest tenth's within-query miss ratio against
   the first tenth is 1.3–1.9 (LongEval) and 2.3–4.5 (SIFT) at every checkpoint from 10% on. **A fixed group
   keeps more than half of its excess** (read as the excess ratio − 1; the design's literal wording, a ratio
   above half its value, holds trivially) **— held in 5 of 8:** the fifth tenth's ratio goes from 1.53–1.56
   at 50% to 1.17–1.47 at 100% on LongEval and from 2.27–2.88 to 1.42–1.94 on SIFT; the three failures are
   shuffled builds.

**The model, as measured.** A node inserted at position i of n ends with in-degree

  in(i) ≈ s(i) · k̄ · ln(n / i) + r(i),

where the first term is exposure (every later arrival chooses it with probability k̄ / (j − 1)), s(i) is the
share of those choices that survive pruning (0.66 for the oldest tenth, about 1 for the rest), and r(i) is the
reverse links it keeps (about k̄ for the newest, a few for the oldest). The newest nodes therefore have
in-degree close to k̄ (about 12), the oldest about 2.4 k̄ (about 29), on any data and in any order, and the
gap fades only as fast as pruning trims the oldest. That is why an extra-link repair works: it gives each
insertion the incoming links that exposure would otherwise give it only over the rest of the build.
