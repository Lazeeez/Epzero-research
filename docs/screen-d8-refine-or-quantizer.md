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
