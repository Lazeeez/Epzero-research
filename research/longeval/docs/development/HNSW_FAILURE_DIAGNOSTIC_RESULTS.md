# Verified HNSW failure diagnostic — 2026-11-17

PACE jobs **7126117 (seed 11)** and **7126118 (seed 29)** reproduced their original
Epzero incremental trajectories and passed independent local verification. Both seed-11
snapshot-3 failures recover when search effort increases on the same graph, with the same
upper-layer route and layer-0 starting node. The retained traces identify candidate evictions
on successful paths. This is a post-hoc development diagnostic, not a general aging result,
an algorithm fix, or a new primary operating point.

This report supersedes the pending-execution status in
[HNSW_FAILURE_DIAGNOSTIC_PLAN.md](HNSW_FAILURE_DIAGNOSTIC_PLAN.md). The submitted protocol
and source bundle are preserved unchanged.

## Recovery on the original graph

Q1 is `161820e44e7022cf44fd99a59bce5792`; Q2 is
`21e8c323e168b0813eff14fdac972f1f`. Recall here is overlap with the exact top 100,
not relevance recall against judgments. Each mean uses 50 queries in its own cohort.

| Snapshot 3, seed 11: ef | Q1 recall | Q2 recall | Development mean | Calibration mean |
|---:|---:|---:|---:|---:|
| 100 | 0% | 0% | 88.74% | 90.26% |
| 200 | 0% | 0% | 92.00% | 95.52% |
| 400 | 100% | 0% | 95.76% | 97.66% |
| 800 | 100% | 0% | 96.66% | 98.66% |
| 1600 | 100% | 0% | 97.36% | 99.20% |
| 3200 | 100% | 99% | 99.60% | 99.52% |

These are first recovery points on the tested grid, not exact minimum effort thresholds.
A 97.36% development mean still coexists with one zero-recall query. A mean calibration
target therefore does not guarantee adequate recall for every query.

For the seed-29 control, Q1 recall across the same grid is 98%, 99%, 100%, 100%, 100%,
100%; Q2 is 88%, 95%, 97%, 99%, 99%, 99%. Its development means are 92.38%, 95.82%,
97.70%, 98.54%, 99.28%, 99.56%. The seed-11 collapse is not shared by this control.

Seed 11's entry document is `141567420` at every snapshot. Q1's layer-0 start remains
`14918060` across snapshots and effort levels; nevertheless Q1 recovers by ef=200 at
snapshot 1, has 98% recall already at ef=100 at snapshot 2, and needs the ef=400 grid
point at snapshot 3. Q2's layer-0 start changes across snapshots, ending at `310911385`.
At snapshot 3, increasing effort leaves each query's upper route and start unchanged.
Recovery thus does not require changing that route, although its role in search difficulty
has not been isolated experimentally.

## What the traces explain

The complete failing traces encounter no exact top-100 neighbor. Retained successful
trace edges supply a directed path from each actual layer-0 start to an exact neighbor:
21 edges for Q1 at ef=400, and 11 for Q2 at ef=3200. These are witnessed paths, not
shortest paths or evidence of global graph connectivity. Both include documents with
lower query similarity than the starting document.

Replaying the native distance-sorted candidate pool reproduces every retained admission
flag in the four compared traces. Along each witnessed path, one candidate is admitted
at both efforts, evicted at the lower effort without a recorded expansion, and expanded
at the higher effort:

| Query | Candidate document | Lower / higher ef | Admission event | Lower-effort eviction | Higher-effort first recorded expansion |
|---|---|---|---:|---:|---:|
| Q1 | `85864826` | 200 / 400 | 2023 | 2998 | 3309 |
| Q2 | `154424178` | 1600 / 3200 | 1345 | 1918 | 2266 |

Event numbers refer to each trace's emitted sequence, including upper-layer events;
they are not expansion counts. First exact-neighbor discoveries occur at events 3562
and 2376 respectively. The high-effort paths continue through these candidates toward
those discoveries. No witnessed path node was rejected at its first low-effort discovery
and admitted at its first high-effort discovery: retention, rather than initial admission,
is the observed difference at these two candidates.

This supports candidate eviction as a concrete explanation for why these observed routes
were unavailable to the smaller search pool. It does not establish a unique bottleneck,
the minimum required effort, why construction produced this topology, or absence of other
implementation issues. Expansions that produce no new-neighbor event are invisible in
this trace format, so an absent expansion event alone is not proof of no expansion.

For context, Q1 uses 3,213 distance evaluations and 207 expansions at ef=200, versus
8,222 and 626 at ef=400. Q2 uses 2,100 and 1,626 at ef=1600, versus 36,756 and 5,049
at ef=3200. These are native work counters, not isolated latency measurements. A
16-fold increase in ef must not be described as a measured 16-fold cost increase.

No exhaustive reachability scan was triggered: both targets exceeded the protocol's
0.95 threshold at ef=3200. Recovery demonstrates reachability of recovered neighbors
from these search starts; it does not certify the entire graph as connected.

## Verification and limits

Both jobs passed ten native fixture tests. Local checks verified source and input identity,
reproduction of all original baseline state records and six ranking files per job, and
all 72 new effort-grid ranking files across both jobs. Calibration and development recall
were independently recounted; scores were recomputed against the original query/document
vectors. The original calibration grid is now independently checked for these two
incremental trajectories. This does not extend that check to the other 16 trajectories.

All 144 retained trace prefixes were checked, totaling 1,368,870 events. Of those traces,
120 are complete and their full-stream hashes were recomputed; 24 retain only the first
20,000 events. Their omitted suffixes cannot be independently verified from the bundles.
The complete low-effort failure traces and the decisive recovery paths are available.
Maximum absolute score discrepancies are approximately 1.06e-7 for run scores and
8.64e-8 for trace distances. Post-diagnostic fingerprints report unchanged graph state;
the full graph itself was not copied for independent reconstruction.

Diagnostic process peak RSS was 17.264 GiB and 17.262 GiB, with approximately 52m25s
wall time each. These include rebuilding the trajectory and diagnostics, not just search.
Both bundles additionally record the FAISS native-module hash; this does not retroactively
establish the binary identity of earlier FAISS runs.

## Evidence and reproduction

The detailed receipts are
`build/longeval-pace-diagnostic-evidence-{7126117,7126118}/local-verification.json`.
They bind returned files and verification code. Aggregate artifacts are under
`build/longeval-diagnostic-comparison/`: `comparison.json`, `effort-sweeps.csv`,
`trace-summaries.csv`, `admission-analysis.json`, and `beam-retention-analysis.json`.
The comparison includes every witnessed path node, its document ID and recomputed score.

SHA-256 identities:

- Received seed-11 archive: `22545f270843f583c9766209ff9028820dcab4986d88e26d5a46f57195e7402b`.
- Received seed-29 archive: `b831ccce33c79a616040682a1e2e0be68d0098b654667389b2463a52042b426e`.
- Aggregate comparison: `12155c2629d08a6d474843c1c78b583101061e307e8d9a332a8be59edf57bd2d`.
- Beam-retention analysis: `4b9ad89af2af2d2c1c0d7b06b85f6ea8f2866ad0111a9233aafdb1c73074faf1`.

Analysis scripts are `build/verify-longeval-diagnostics.py`,
`build/longeval_diagnostic_verification.py`, `build/summarize-longeval-diagnostics.py`,
`build/analyze-longeval-trace-admission.py`, and `build/replay-longeval-trace-beams.py`.
Verification and path scoring require the local mapped corpus and archived query inputs.
The retention replay needs the verified trace bundles and aggregate comparison only.

## Research consequence and next work

Keep the original ef=200 matrix intact. Do not select ef=3200 as a primary setting from
these development outcomes. The next mechanism experiment should explicitly record all
expansions and evictions, reproduce the same graph and baseline rankings, and test a
separately labeled intervention on candidate retention with work accounting and successful
query/seed controls. Such an intervention can test the suggested mechanism more directly;
these two selected failures cannot establish its prevalence or general usefulness.

Remaining paper work includes IVF and matched-cost controls, a frozen representation and
inferential protocol with practical-effect margins, held-out evaluation, and a defensible
contribution assessed against related work. Neither novelty nor publication readiness is
established by this diagnostic.
