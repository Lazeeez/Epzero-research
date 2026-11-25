# Development screen D9: refining recent insertions in the default build path

Recorded on 23 November 2026, before it was run, after screen D8.

## Question

Screen D8 showed that in Epzero's default build path the refine pass after the bulk build is itself a
source of arrival bias: it treats only the documents present at the bulk build. D9 tests refining the
documents inserted since the last refine, once per snapshot (as a checkpoint would), using the new
`refine_links(first=...)` (Epzero branch `min-backlinks`), alone and with `extra_backlinks` 4.

## Design

As D7, grown mode (8-bit codes, `fast_build` on, M 16, efConstruction 100, seed 11; snapshot 1
bulk-built and refined; snapshots 2 and 3 inserted one at a time), and after each later snapshot's
insertions, `refine_links(threads=all, ef=50, first=position before that snapshot)`. Arms:
`extra_backlinks` 0 and 4. Compared with D7's grown (no refine of later snapshots) and rebuilt arms.
Measures as D7, plus the refine time.

## Prediction

Refining each snapshot's insertions cuts the grown excess at ef 200 (2.27 points) by at least half,
with the oldest cohort's miss rate rising at most 0.3 points and recall@100 not falling; the refine
time on all cores is at most the single-threaded insertion time of those documents. Adding
`extra_backlinks` 4 lowers the excess further.

## Result (run 23 November 2026, after the design above was committed)

| Arm (grown unless stated) | Newest-cohort excess, ef 100 / 200 / 400 (pp) | Miss rate by cohort, ef 200 | Recall@100 / @10, ef 200 | Distance computations, ef 200 | Extra time |
| --- | --- | --- | --- | --- | --- |
| no repair (D7) | 3.29 / 2.27 / 1.07 | 4.06 / 6.80 / 6.96% | 0.9476 / 0.9793 | 3,032 | — |
| refine each snapshot | 2.22 / 1.56 / 0.88 | 4.29 / 6.45 / 6.54% | 0.9478 / 0.9761 | 3,126 | refine 8 s (all cores) |
| `extra_backlinks` 4 (D7) | 1.96 / 1.44 / 0.57 | 4.22 / 6.34 / 6.41% | 0.9487 / 0.9769 | 3,047 | inserts +3 s |
| refine each snapshot + `extra_backlinks` 4 | 0.89 / 0.53 / 0.39 | 4.35 / 5.76 / 5.57% | 0.9509 / 0.9777 | 3,117 | refine 8 s, inserts +2 s |
| rebuilt (D7, reference) | 0.11 / 0.22 / 0.20 | 4.42 / 5.21 / 5.39% | 0.9522 / 0.9777 | 3,144 | bulk build 49 s |

**Against the prediction:** refining each snapshot's insertions alone cut the excess by only 31%
(at least half was predicted); the oldest cohort rose 0.23 points (within 0.3) and recall@100 did
not fall. The refine took 8 s on all cores against 67 s of single-thread insertion (as predicted).

**As predicted, the combination lowers it further, and by much more than either alone:** with
`extra_backlinks` 4, refining each snapshot cuts the excess by 77% (2.27 → 0.53 points at ef 200,
73% at ef 100, 64% at ef 400); the oldest cohort rises 0.29 points (just within 0.3); recall@100 rises
0.33 points to 0.9509, close to the rebuild's 0.9522, and recall@10 equals the rebuild's (0.9777).
Distance computations are 2.8% above the unrepaired graph and 0.9% below the rebuild.

**Reading:** the two repairs are complementary. `extra_backlinks` gives each insertion incoming links
from nearby nodes it did not choose (the later arrivals it has not had); refining recent insertions
gives them the second, better-informed pass that bulk-built documents get. Together they bring a
grown index close to a full rebuild in recall and in arrival fairness, for a few seconds of
multi-threaded work per checkpoint instead of a rebuild. One seed, development data: three seeds and
the confirmation readings come next.
