# Development screen D7b: why `extra_backlinks` 4 under-performs in the product path

Recorded on 23 November 2026, before it was run, after screen D7 (whose prediction failed).

## Design

As D7 (grown mode: snapshot 1 bulk-built with 8-bit codes and refined, snapshots 2 and 3 inserted one
at a time), with two arms: `extra_backlinks` 8 with `fast_build` on; `extra_backlinks` 0 and 4 with
`fast_build` off (the graph built on full vectors).

## Prediction

If approximate construction distances are the cause, `fast_build` off with 4 cuts the grown excess
(relative to `fast_build` off with 0) by at least half and does not lower recall@10. If the setting is
simply too small for this path, 8 with `fast_build` on cuts it by at least half. Either or both can
hold; the result decides the setting to carry forward.

## Result (run 23 November 2026, after the design above was committed)

Grown mode (snapshot 1 bulk-built with 8-bit codes and refined; snapshots 2 and 3 inserted one at a
time), ef 100 / 200 / 400:

| Build | extra_backlinks | Newest-cohort excess (pp) | Oldest-cohort miss, ef 200 | Recall@100 / @10, ef 200 |
| --- | --- | --- | --- | --- |
| 8-bit codes (D7) | 0 | 3.29 / 2.27 / 1.07 | 4.06% | 0.9476 / 0.9793 |
| 8-bit codes (D7) | 4 | 1.96 / 1.44 / 0.57 | 4.22% | 0.9487 / 0.9769 |
| 8-bit codes | 8 | 1.14 / 1.15 / 0.48 | 4.37% | 0.9487 / 0.9787 |
| full vectors | 0 | 3.57 / 2.20 / 1.08 | 4.10% | 0.9475 / 0.9774 |
| full vectors | 4 | 1.69 / 1.50 / 0.53 | 4.23% | 0.9490 / 0.9782 |
| rebuilt reference (D7) | 0 | 0.11 / 0.22 / 0.20 | 4.42% | 0.9522 / 0.9777 |

**Approximate construction distances are not the cause:** building on full vectors gives the same
baseline excess (2.20 against 2.27) and the same cut with 4 (−32% against −37%). **The setting is
too small for this path:** 8 cuts the excess by 49% at ef 200 (65% at ef 100, 55% at ef 400), close
to half, but the oldest cohort's miss rate then rises 0.31 points, just above the 0.3 limit.

**The recall@10 drop in D7 is within build-to-build noise:** the two baselines differ by 0.19 points
in recall@10 (0.9793 and 0.9774) with only the construction distances changed, and 4 with full
vectors raises it (+0.08).

**Reading:** in the product path the grown excess is larger than in the plain configuration (2.27
against 1.62 points), and `extra_backlinks` removes a smaller share of it at the same setting. Two
differences could explain the larger excess, neither yet tested: the refine pass makes the bulk-built
part of the graph denser, so later insertions compete with better-linked incumbents; and the 8-bit
ranges were trained on snapshot 1 only, so later documents may be coded less accurately (quantizer
staleness, which the literature review expected only under drift). Next: separate the two (rebuild
with snapshot-1 ranges; grow without the initial refine), and test a repair of recent insertions at
checkpoints as a complement.
