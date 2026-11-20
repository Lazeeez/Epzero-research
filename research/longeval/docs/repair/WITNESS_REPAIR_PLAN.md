# Calibration-only witness repair — initial screen, 2026-11-18

This is the second candidate mechanism after overlap-guided resumption failed its
matched-work screen. It is exploratory, not an established novel method. This plan
is written before executing its real-vector experiment.

## Hypothesis and mechanism

A successful higher-effort search supplies a directed witness path into a region
that a lower-effort search misses. Instead of guessing the next expansion at query
time, derive a small static edge overlay from calibration-query failures. Subsequent
queries use normal search over the same base graph plus these edges. They do not
receive query IDs, exact answers, calibration labels or a learned recall estimate.

For each calibration query whose low-effort recall is below .95:

1. Run low search with k=100, ef=100, and a 1,000-distance cap; run successful-path
   search with ef=3,200 and a 4,000-distance cap. Exact top-100 references are used
   offline to identify missed neighbors that the larger search reached.
2. Examine these missed neighbors in high-search discovery order. Follow their
   first-discovery parents back to the layer-0 start. Find a node expanded by low
   search and a later node on this path that low search never scored, whose distance
   improves low search's final worst result. The proposed edge must skip at least
   one path node and must not duplicate an existing edge. Prefer the latest eligible
   low-expanded source and the earliest eligible destination on its suffix.
3. Retain at most one proposal per calibration query. Rank unique proposed edges by
   distinct-query support, then total skipped path nodes, then numerical source and
   destination labels. Add at most 32 edges globally and two per source.

The destination can still be a calibration exact neighbor. Report this fraction;
compare with direct-answer shortcuts to detect simple answer memorization. Adding
edges can change a bounded search adversely; there is no monotonic-recall guarantee.
All calibration searches and exact-reference construction are learning costs, not
free query-time gains. The overlay changes the search graph but preserves the saved
base index; it is a read-only callback prototype, not a native repair implementation.

## Fixed first screen

Reuse the prior 8,192-row snapshot-1 sample and its seed 11 base graph. Fit using only
the 50 calibration queries. Freeze serialized overlay edges before any development
queries run. Keep all 50 separate development queries and both distance caps 1,000
and 2,000 at ef=100. The higher-search control uses ef=3,200 and identical actual
distance work for every intervention query. Required arms:

- Original graph at ef 100 and wide search.
- Witness overlay.
- Direct-answer overlay with the same number of edges at the same sources, using
  the proposing calibration query's nearest eligible exact neighbor.
- Random-destination overlay with the same per-source edge counts, seed 20261118.

Record per-query returned rankings, actual distance/adjacency work, overlay reads,
failure tails, learning costs, proposal provenance and graph/overlay hashes. Verify
scores independently. Off-mode replay must match native search before intervention.
Development queries and their exact references cannot enter the fitting function.
Held-out retrieval outcomes remain untouched. Do not tune this first screen after
seeing development results or promote a calibration-only gain.

A positive screen must improve separate development quality at matched actual work,
beat the simpler direct-answer/random controls, and justify a full-data, multi-seed
test. Otherwise retain the negative result. Sampling may remove the full-corpus
bottlenecks, so failure does not prove all graph repair impossible.

## Prior-art risk and required distinction

[CatapultDB](https://arxiv.org/html/2603.02164v1) already uses workload-derived
shortcuts to improve starting points. [RoarGraph](https://arxiv.org/html/2408.08933v1)
builds a query-guided graph. These defeat a broad claim that query-driven shortcuts
or workload-adaptive graphs are new. The possible narrower distinction to investigate
is budgeted repair of **witnessed low/high-effort path failures inside an existing
graph**, with separate-query transfer and explicit repair-cost amortization. That
distinction needs more algorithm-level prior-art review and successful experiments.

Approximate neighbor scoring is also established by
[FINGER](https://arxiv.org/abs/2206.11408),
[PEOs](https://arxiv.org/abs/2402.11354), and
[CRouting](https://arxiv.org/html/2509.00365v1). Merely using a cheap prediction to
skip distance calls would not establish a new method. This screen therefore changes
the repair mechanism rather than renaming these existing optimizations.

## Competitor work alongside this screen

Add pinned hnswlib 0.8.0 without changing the earlier experiment environment. Validate
its IP semantics, external-label mapping, deterministic single-thread behavior and
native timing harness against Epzero and FAISS. A small-sample operating-curve check
is infrastructure validation; it cannot replace the pending full-data PACE timing
jobs or establish a competitive paper result. Preserve binary/version identities.
