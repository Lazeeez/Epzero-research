# Full-cohort search-work diagnostic

Status: implemented and locally tested; PACE execution pending. This supplements the
verified three-seed HNSW matrix and selected-candidate retention results. It is a declared
development diagnostic, not a frozen primary paper analysis or a native latency benchmark.

## Question and scope

Does the seed-11 recall collapse remain unusual when search is compared using actual
distance-evaluation counts across every development query and all three snapshots? How do
mean and per-query recall change when a snapshot-1 work budget is retained versus selecting
a new budget using the separate calibration cohort?

Use **all 100 training queries**: the existing 50 calibration and 50 development IDs, separately
reported. Use Epzero incremental construction with seeds **11, 29, 47** at all three snapshots.
Do not select only the previously failing queries, omit seed 11, change construction, or use
any held-out query/judgment. The same queries recur across snapshots and seeds; these are
repeated observations, not 900 independent queries.

The original 18 trajectories already establish replay equality and the limited observed
benefit of shuffling. This experiment holds construction mode fixed while studying search
work. It is not a new cross-backend matched-cost result or a substitute for IVF controls.

## Conditions and selection

Reproduce all original snapshot-state records and both baseline ranking hashes before each
observation. At every query/snapshot, run ordinary native search and native tracing at:

- **ef=200**, the original frozen-effort setting;
- **ef=3200**, the declared wide-search comparator used in the prior mechanism experiment.

Require trace/search ranking equality. Retain the complete native trace at both efforts.
From the wide trace, reconstruct the top 100 at distance budgets:
**1,000; 2,000; 4,000; 8,000; 16,000; 32,000; 64,000**.
These budgets are declared before inspecting full-cohort outcomes of this experiment. They
are development choices informed by earlier diagnostics, not preregistered confirmatory values.

Include upper-layer distance evaluations and both entry evaluations in each count. An
exhausted search may use fewer evaluations than its budget; report actual work and whether
the full native score stream is covered. At an exactly equal budget, unscored final expansions
may still remain, so score-stream coverage is not a claim about completing every expansion.
Budgets ending before 100 candidates are available are explicitly infeasible. Preserve partial
outputs and affected query IDs, report no cohort mean for an infeasible grid point, and never
select a budget by dropping those queries.

For each seed, choose the smallest feasible budget attaining mean exact-top-100 overlap of
0.95 on snapshot 1's calibration cohort. Keep that budget fixed across later snapshots. Also
report the smallest feasible calibration budget separately at each snapshot. A target miss is
recorded as unavailable, not silently replaced with a larger untested budget. Development
outcomes never enter these selections. Retain the complete development/calibration grids,
per-query recall, minima and zero-recall counts, including unfavorable outcomes.

The full matrix contains 1,800 native query conditions and 6,300 budget-prefix query
conditions: 100 queries, three snapshots, three seeds, two native efforts and seven budgets.
Each job exports 54 cohort ranking files and 600 complete compressed native traces.

## Meaning of the work-budget results

This is **offline analysis of a native traversal prefix**. It does not add a capped-search API
to the engine or measure the latency of a native search stopped at that point. The diagnostic
executes the full native search to obtain its trace; its instrumented wall time cannot be used
as budgeted-query latency.

For uncompressed float32 inner-product search, each emitted neighbor event corresponds to
one distance evaluation. The two extra evaluations are the initial upper entry and the layer-0
entry. Fail unless `distance_computations == len(events) + 2`. Reconstruct the best 100 from
the layer-0 entry and scored neighbors, preserving native discovery-order ties. Require the
complete reconstructed ranking to equal the native result exactly.

Validate prefix semantics against explicit stopping in the independently observed traversal
for the same four previously declared diagnostic queries at every snapshot and seed. All
feasible budgets must produce exactly the same labels, distances and work counts. First
require full native event/ranking/expansion/count parity for that traversal. Fixtures also check
parity at dimensions 16 and 2560, ties, early infeasibility and exhaustion. This validates the
prefix interpretation without claiming the Python traversal has native performance.

Native complete-search expansion counts remain available. The native neighbor trace omits
expansions that discover no new neighbors, so this experiment does **not** infer exact
prefix expansion counts from it. Distance counts, expansion counts, memory and latency remain
distinct measures. No claim of performance superiority follows merely from a recall/work curve.

## Evidence and execution

Keep original engine sources, binary, representations and baseline operating points unchanged.
Use the established full-data trajectory observer, with separate new output directories.
Validate baseline source/binary/package/query/reference identities and the native distance ABI
before downloading the corpus. Re-fingerprint the graph after each snapshot's observation.

Complete traces are bounded at two million events and 8 MiB compressed per trace, with
384 MiB of compressed traces per job. Exceeding any limit fails the job; no trace suffix is
silently discarded. Stream/file hashes, upper routes, result runs, per-query budget counts,
cohort selection records, tests and process resource reports are retained. All corpus archives,
vectors, audit arrays and original trace work files stay on compute-node temporary storage.
Only bounded evidence is copied to the shared result directory.

The implementation is `work_budget.py`. The focused suite has 26 passing tests, including
the prior retention and trajectory regressions. Real-data results require independent local
verification after PACE returns; passing fixture tests does not establish the outcome.

## Decision after the results

Compare both mean and tail behavior, explicitly distinguishing fixed ef, fixed work budget,
calibration-selected budget and actual exhausted-search work. Retain the wider-search
baseline: selected candidate rescues previously matched it exactly at equal distance counts.
Do not infer a need for a new retention policy if ordinary search explains the observed gains.

Next, specify feasible IVF training samples and separate frozen-centroid updates, same-training
rebuild controls and fresh-current-snapshot training. Training population, centroid count,
probe grid, seeds, and cost accounting must be explicit before those full-data runs. IVF
training/search work is not directly equated with HNSW work without defining the counted
operations. The current experiment will supply the broader HNSW work/recall reference for
that comparison; it does not complete the remaining IVF or native resource gates.
