# Candidate-retention experiment after the verified failure diagnostic

Status: implemented and locally tested; full-data PACE results pending. This is an
explicitly post-hoc mechanism experiment using selected development queries and candidate
identities. It does not define a deployable repair policy or a new primary search setting.

## Question

The prior traces identify documents `85864826` and `154424178` as candidates evicted
without a recorded expansion in failed seed-11 searches, but explored along successful
higher-effort routes. Does preserving one such evicted candidate for a later expansion
suffice to recover neighbors on the same graph? Does a wider ordinary search do as well
under the same distance-evaluation cap?

The existing result is documented in [HNSW_FAILURE_DIAGNOSTIC_RESULTS.md](HNSW_FAILURE_DIAGNOSTIC_RESULTS.md).
The two candidate identities were selected from that result. Any positive outcome here is
conditional on this selection and cannot estimate prevalence, novelty or policy benefit.

## Fixed scope and controls

- Reproduce the original incremental graphs and both baseline ranking hashes at all three
  snapshots for seeds 11 and 29. Intervene only at snapshot 3. Preserve the native engine,
  binary, input vectors and previous operating points.
- Use the two previously failing development queries and the same two deterministic
  SHA256-selected development controls. Apply **both** candidate identities to **all four**
  queries and both seeds; do not restrict reporting to the anticipated successful pairing.
- At each query, reproduce ordinary searches at ef=200, 201, 400, 1600, 1601, 3200.
  ef=201 and 1601 test an ordinary one-slot increase. No relevance judgments or final-test
  queries enter the experiment.
- At each lower ef in {200, 1600}, independently try each of the two selected candidates,
  plus one unrelated unexpanded eviction from the native baseline. Choose that sham by the
  smallest SHA256 of `longeval-retention-sham-v1:SEED:QUERY:EF:DOCUMENT`, excluding both
  selected candidates. Record an unavailable sham explicitly as a no-op.
- Pair each intervention with an ef=3200 ordinary replay capped at the intervention's
  distance-evaluation count. The cap includes all upper-layer and entry scoring, including
  repeated evaluations. A capped search may stop inside an expansion. Report actual work
  and whether the cap stopped it; do not claim equal time, memory, or expansion count.

There are 18 arms per query, 72 per seed and 144 in total: six ordinary baselines, six
interventions and six paired work controls per query. Each result includes the top-100
document IDs, scores, exact-neighbor recall and work counts. No new efficacy threshold,
statistical test or primary setting is selected from this experiment.

## Exact intervention

Run the original greedy upper-layer route and ordinary layer-0 beam. If the chosen
candidate was admitted, then evicted while still unexpanded, retain its label in one
auxiliary slot. Once the ordinary beam exhausts, expand that label once outside the beam
and resume ordinary beam processing of newly discovered neighbors. Do not reinsert the
candidate into results, rescore it, alter the graph, or force-admit any new neighbor.
If it is never admitted, never evicted, or was already expanded, the intervention is a no-op.

This tests sufficiency of one deferred expansion, not necessity or uniqueness of the
candidate. A failure does not disprove a retention explanation: a later path may require
multiple saved candidates, another expansion order, or more effort. A success does not show
that candidate selection can be performed without post-hoc knowledge.

## Execution and evidence gates

`retention_search.py` implements a separate Python traversal over native `neighbors` and
`get_vector` accessors. It obtains the unchanged loaded library's own inner-product distance
function through its exported GNU/Linux C++ ABI symbol. Unsupported layouts or missing
symbols fail rather than falling back to different scoring. This ABI dependency is restricted
to the research diagnostic and checked by a native fixture before downloads.

For every ordinary baseline, require exact equality of the complete emitted native score/
admission sequence, result labels and float32 distances, expansion count, distance count,
and ordinary native search output. Complete expansions and evictions are then observable
in the separate traversal, including expansions with no new neighbors. Every intervention
must preserve the same upper route. Inactive interventions must reproduce baseline rankings
and work counts. Re-fingerprint the graph after all arms.

Log every expansion, its ordered adjacency list, scored neighbor/admission, eviction,
deferred label, and initial score. Save complete compressed logs with stream and file hashes.
No truncation is allowed: fail above two million events, one million distance evaluations,
8 MiB compressed per arm, or 256 MiB of compressed logs per job. Full work files and all
large downloads remain on compute-node temporary disk. Only bounded evidence is copied to
the shared result directory.

Local validation: 19 tests pass, including native parity at dimensions 16 and 2560,
nonsequential labels, equal-distance ordering, a fixture in which a deferred eviction reaches
a hidden neighbor, inactive interventions, stopping inside an expansion, corrupted native
counts, complete-log guards, all-arm replay from saved evidence, and rejection of changed
baseline graphs/rankings. This does not replace full-data native parity on PACE.

## Interpretation and subsequent work

Report all arms, including shams, no-ops, failed rescues, seed-29 controls, and wider-search
comparators. Explicitly separate graph construction, candidate selection, search work and
Python instrumentation overhead. Do not use this implementation to claim native latency.

If targeted rescues work, the next question is whether a query-independent rule can identify
useful deferred candidates within a fixed budget, evaluated on separate development data
and eventually a frozen held-out protocol. If wider ordinary search already dominates the
selected interventions at comparable work, retain that negative result. Neither outcome
removes the remaining IVF, resource-control, representation-freeze, inference and novelty work.
