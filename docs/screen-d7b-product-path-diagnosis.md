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
