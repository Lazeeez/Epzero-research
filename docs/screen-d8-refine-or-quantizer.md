# Development screen D8: is the product path's larger excess the refine pass or stale 8-bit ranges?

Recorded on 23 November 2026, before it was run, after screen D7b.

## Question

Grown in the product path, the newest cohort's excess is 2.27 points at ef 200, against 1.62 in the
plain configuration and 0.22 for a full rebuild. Two explanations:

- **Refine density:** the bulk-built snapshot-1 graph is refined, so later insertions compete with
  better-linked incumbents.
- **Quantizer staleness:** the 8-bit ranges were trained on snapshot 1 only, so later documents may be
  coded less accurately.

## Design

As D7 (8-bit codes, `fast_build` on, M 16, efConstruction 100, seed 11, `extra_backlinks` 0):

- **grown, no refine:** snapshot 1 bulk-built without the refine pass, then snapshots 2 and 3 inserted
  one at a time;
- **rebuilt, stale ranges:** the 8-bit ranges trained on snapshot 1 only, then all documents
  bulk-built and refined.

## Prediction

None is preferred; the screen decides. Refine density is supported if grown without refine has a
clearly smaller excess than grown with refine (2.27), by at least 0.5 points at ef 200. Quantizer
staleness is supported if rebuilt with stale ranges has a clearly larger excess than the rebuilt
reference (0.22), by at least 0.5 points at ef 200.

## Result (run 23 November 2026, after the design above was committed)

| Arm | Newest-cohort excess, ef 100 / 200 / 400 (pp) | Miss rate by cohort, ef 200 | Recall@100 / @10, ef 200 |
| --- | --- | --- | --- |
| grown, refined bulk build (D7) | 3.29 / 2.27 / 1.07 | 4.06 / 6.80 / 6.96% | 0.9476 / 0.9793 |
| grown, no refine | 2.39 / 1.36 / 0.62 | 4.76 / 6.71 / 6.88% | 0.9439 / 0.9766 |
| rebuilt (D7) | 0.11 / 0.22 / 0.20 | 4.42 / 5.21 / 5.39% | 0.9522 / 0.9777 |
| rebuilt, 8-bit ranges from snapshot 1 | 0.15 / 0.49 / 0.23 | 4.33 / 5.32 / 5.50% | 0.9522 / 0.9766 |

**Refine density is supported:** without the refine pass the grown excess is 0.91 points smaller at
ef 200 (bar: 0.5). The refine pass improves the bulk-built documents (snapshot-1 miss rate 4.76% →
4.06%) but documents inserted afterwards are never refined, so the gap between them widens.

**Quantizer staleness is not supported:** 8-bit ranges trained on snapshot 1 only raise the rebuilt
excess by 0.27 points at ef 200 (bar: 0.5), and less at ef 100 and 400. On this organically growing
collection, stale code ranges matter little, as the literature review expected without drift.

**Consequence for the engine:** in the product path, the refine pass is itself a source of arrival
bias, because it treats only the documents present at the bulk build. Refining the documents
inserted since the last refine (for example at each checkpoint) would treat every document once.
That is the next design to screen (D9), against and together with `extra_backlinks`.
