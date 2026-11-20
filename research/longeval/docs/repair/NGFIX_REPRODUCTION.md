# NGFix/RFix direct competitor — integration record, 2026-11-18

Status: **PACE job7141656's native synthetic/sample evidence is independently
verified**. See [NGFIX_RESULTS.md](NGFIX_RESULTS.md):1,600 real plus256 synthetic
conditions,185,600 independently checked score occurrences and500 matching packaged
files. Development transfer is modest; using fitted queries to select ef fails the
development recall targets. This is an integration reproduction, not a full-data
paper reproduction or speedup claim. The full log has five Slurm working-directory
fallback warnings; exact cause remains unresolved despite passing integrity checks.

**V03 supersedes v02 for launch:** [failure record and corrected commands](NGFIX_LAUNCH_FAILURE_7141318.md).
Job7141318's pasted log confirms the missing-module failure. The independently reproduced
working-directory import defect is fixed with Python safe-path mode;22 tests pass.
The native source, compiler flags and inputs below remain unchanged.

The previous `build/longeval-ngfix-screen-02.tar.gz` (2,543,714 bytes), SHA-256
`5532d5b8a7954efe33a291e1a583cfccaa743a9831f6aaaa859670fed66661f6`.
The isolated package's 498 files are verified, its native wrapper compiles using
only bundled non-system dependencies, and all 20 packaged input/output validation
tests pass on Linux. Version01 failed an isolated missing-header check; do not use it.
See [submission and retrieval instructions](../../../../scripts/pace/LONGEVAL_NGFIX.md).

[Paper](https://arxiv.org/html/2510.22316v1),
[author code](https://github.com/yuhuifishash/NGFix/tree/9acd987cf1c420069f8fd602c0d4e4aa81053558).
The untouched checkout is `build/third-party/ngfix`, pinned to
`9acd987cf1c420069f8fd602c0d4e4aa81053558`.

## Why this changes the novelty boundary

The paper defines escape hardness through the smallest maximum query-neighbor
rank along a graph path. NGFix repairs local neighborhood connectivity; RFix
improves routes toward a query neighborhood. The method also maintains repair
edges under insertions. Thus historical-query graph repair, search-trap escape,
two-phase diagnosis, and incremental repair are already established directions.
Simply combining our witness overlay with a hardness score is insufficient novelty.

Our possible distinction remains an open research question: obtaining effective
repair outside the small fitted answer population, under explicit edge and work
budgets. This is an experimental objective, not a claimed contribution. Corpus
proxies, reverse edges, multi-entry search and query-guided graph construction
also require direct prior-art comparison before any novelty claim.

## Controlled adapter

`ngfix_driver.cpp` uses the original full-precision inner-product implementation.
It builds the upstream **bottom-layer graph**, not Epzero's hierarchical graph.
It reloads the same serialized base for four controls: base, NGFix, RFix, both.
M=16, efConstruction=100, MEX=48 are initial integration settings, not a tuned
performance comparison or the paper's complete experimental configuration.

The wrapper applies the author's exact-reference sequence: NGFix(100,100),
NGFix(10,10), RFix(10), as applicable. Calibration references have depth 200.
Construction and fitting are serial; query rows and groundtruth rows are loaded
together and explicitly validated. This avoids a potential mismatch in the
upstream example's parallel stream reader: its query read and groundtruth read
use separate lock acquisitions. No upstream files were modified.

Query settings are ef=100/200/400/800, k=100; each condition has one warmup and three
timed calls. Variant order rotates by query. Ranking equality is checked outside
timing. The wrapper writes all IDs/scores, per-call timing/counters, stage times,
and before/after index states for independent verification. Upstream counters
exclude the initial entry distance and are not equated to Epzero counters.

The local wrapper compiled with GCC13.3.0, `-O3 -std=c++20 -fopenmp
-march=skylake-avx512`, the pinned checkout and extracted Ubuntu Boost1.83 headers.
The local executable is `build/longeval-ngfix-driver-avx512`. This machine lacks
AVX-512, so that binary has not been executed locally. The separately compiled PACE
binary completed native fixture and real-sample execution with its identity recorded.
The wrapper keeps four graphs in memory and is currently a sample driver;
full-corpus integration requires a deliberate memory and scheduling design.

## Prepared sample contract

`ngfix_inputs.py` passes 11 binary format/validation tests. Its real-data preparation
rechecks the existing 8,192-vector sample and pinned query embeddings. It maps
sample row labels to dense competitor IDs, orders calibration queries before
development, computes float64 exact top-200 for **only 50 calibration queries**,
and verifies their top-100 against the existing reference. All binary matrices
roundtrip exactly. The 409,600 reference vector-pair calculations are recorded.

Prepared inputs: `build/longeval-ngfix-inputs-01/`. The native process receives
`base.fbin`, `queries.fbin`, and `calibration_top200.ibin`. Development references
remain outside the fitting process in `queries.json`. No held-out test queries
are included. Fifty fitting queries are an engineering screen; poor performance
with that budget would not establish superiority over properly trained NGFix.

## Execution gate and remaining work

1. Complete: native synthetic and sample execution, independent local ranking/score
   verification, and returned compiler/binary/hardware/source/input identity checks.
   Graph-state equality was checked on PACE; graph bytes were not returned.
2. Complete: .95/.99 selections and build/repair costs recorded. Those selectors
   reuse fitting queries and do not certify development recall.
3. Remaining: disjoint fitting/tuning/evaluation, broader independent training,
   and matched hardware/measurement boundaries against standard competitors.
4. Remaining: full-data and additional-dataset evaluation after a valid protocol
   and a distinct candidate pass the promotion gate.

Returned runtime evidence is verified as detailed in the results record. Any large
transfer, input preparation or download must occur in compute-node temporary local storage.
The existing native-cost-02 package remains a separate, unchanged baseline job.

The new job enforces that storage requirement and performs a synthetic native
correctness run before downloading snapshot1. It tests four controls on the
verified sample and records 1,600 query conditions / 4,800 timed calls. The independent
verifier checks every returned score, calibration-only setting selection, repeated
counter stability, and unchanged query-time graph hashes. Tests explicitly reject
mispaired calibration references, missing timings, wrong scores and mutated states.
These are verifier tests, not substitute native benchmark results. The original
100-query sample and its previous exposure remain disclosed.
