# Witness-repair screen and calibration coverage — 2026-11-18

**Decision: do not promote this repair rule to a large experiment.** The static
overlay improves its fitting queries, but provides almost no benefit to separate
development queries. All 32 selected edges have support from just one calibration
query, and all 32 destinations are exact neighbors of that query. This screen has
not demonstrated useful transfer or a novel competitive method.

The [pre-execution plan](WITNESS_REPAIR_PLAN.md) fixes the algorithm, edge budget,
operating points and controls. This follows the failed overlap-resumption screen;
it does not overwrite that evidence.

## Experiment and results

The existing 8,192-row snapshot-1 graph, seed11, was reused. Fifty calibration queries
proposed 32 edges, with at most two per source; the overlay was serialized and frozen
before evaluating 50 development queries. Search uses ef100 at two distance caps.
Matched controls use the same added-edge counts at the same sources, with either
direct calibration-answer destinations or deterministic random destinations. Wider
search on the original graph is capped to each arm's actual distance work.

| Population / cap | Original recall | Witness recall | Wide recall at witness's actual work | Witness wins / losses vs wide |
| --- | ---: | ---: | ---: | ---: |
| Calibration / 1,000 | 90.06% | 90.72% | 90.06% | 25 / 2 |
| Development / 1,000 | 88.86% | 88.88% | 88.86% | 1 / 0 |
| Calibration / 2,000 | 94.04% | 94.08% | 94.04% | 2 / 0 |
| Development / 2,000 | 93.40% | 93.40% | 93.42% | 0 / 1 |

The +0.02 percentage-point development change at cap1,000 is one additional exact
neighbor across 5,000 result positions. The development minimum recall and count
below .90 do not improve. At cap2,000 direct-answer edges achieve 93.42%, versus
93.40% for witness edges, with slightly different actual work. None establishes a
useful method advantage. These are exploratory descriptive findings, not significance
claims. The sampled graph may omit full-corpus traps.

The witness overlay is read by 24/50 development queries at cap1,000 and 28/50 at
cap2,000. Thus lack of transfer is not simply that no development query encountered
an added-edge source. Reading an edge does not imply admitting its destination.

Fitting low/high searches use 249,960 distance calls. Exact references were reused;
their calibration-side construction corresponds to 409,600 query/vector pairs and
must be counted in a future repair-cost comparison. The raw report's
`fitting_validation_distance_calls=63,535` is a **per-pass parity count**, not the
total validation overhead: native parity performs a native trace, replay and normal
search, totaling 190,605 distance calls across those three equal-work passes.
The recorded 6.75s fitting wall time includes Python tracing/validation and is not
a native repair-time benchmark.

## Full-corpus coverage audit

To check whether the small calibration population supplies broadly shared answer
destinations, the existing full-corpus exact and approximate runs were reanalyzed,
using external document IDs and the unchanged 50/50 query split.

| Snapshot | Development exact-neighbor occurrences present in calibration answers | Development queries with no such overlap |
| --- | ---: | ---: |
| 1 | 441 / 5,000 = 8.82% | 24 / 50 |
| 2 | 406 / 5,000 = 8.12% | 28 / 50 |
| 3 | 381 / 5,000 = 7.62% | 30 / 50 |

At snapshot3, calibration answers contain only 3 of 209 native-ef200 development
misses for seed29 and 3 of 206 for seed47. Seed11 is an important exception: 89 of
400 misses occur in calibration answers, including 87 of Q1's 100 missing neighbors.
Q2 has zero calibration-answer overlap. Do not claim uniformly low coverage for
every failed query or every construction seed.

This is evidence of sparse shared answer coverage, **not an upper bound on graph
repair**: a learned edge could help reach documents that were never calibration
answers. It motivates testing a structural repair objective or obtaining a larger,
disjoint training-query population before relying on learned answer destinations.
It does not justify consuming the held-out test queries as training data.

## Verification and next decision

Nine repair tests pass, covering calibration-only fitting, deterministic selection,
edge limits, invalid endpoints, path witnesses and base-graph preservation. The
independent verifier replays **352,607 calibration log events**, checks all 800 query
conditions and validates scores against float64 vector products. Maximum trace-score
error is 1.24e-7. All 32 proposals and the fixed overlay are reproducible; the saved
base index is unchanged. The full-corpus audit verifies its source run hashes against
the existing local PACE verification receipts.

Do not run another unrestricted heuristic sweep on these 100 queries. The next
method proposal must specify either a corpus-derived structural signal that can
transfer beyond calibration answer IDs, or a larger independent training-data
protocol with costs and competitor access declared. Compare its mechanism with
[CatapultDB](https://arxiv.org/html/2603.02164v1) and
[RoarGraph](https://arxiv.org/html/2408.08933v1) before claiming novelty in query-guided
shortcuts. Keep the new [native competitor harness](STANDARD_GRAPH_RESULTS.md) as
the performance baseline infrastructure.

Artifacts: `build/longeval-witness-pilot-01/`,
`build/longeval-calibration-coverage-01/`,
`build/longeval-method-screen-02-verification.json`, and
`build/longeval-witness-tests-01.xml`. Held-out retrieval remains untouched.
