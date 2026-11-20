# Independently tuned NGFix screen — job 7148412

**Result:** all 24 selected split/control/target combinations meet their development
mean-recall target. The protocol no longer selects search effort from fitted
queries. NGFix adds small recall gains at essentially similar observed latency;
this screen establishes neither a competitive speedup nor a new method.

The frozen [protocol](NGFIX_DISJOINT_PLAN.md) uses three overlapping 30-fitting /
20-tuning /50-development splits of the same exposed 100 queries. The four controls
are base, NGFix, RFix and both; ef is selected from 100/200/400/800 using tuning
recall alone. All controls select ef 200 for 95% and ef 400 for 99% in every split.
The development set is the same 50 queries throughout; these are not independent
datasets, construction seeds or 300 distinct development observations.

## Selected operating points

Times are the mean of 50 per-query medians over three timed native C++ calls,
following one warmup. All rows use k=100 on the 8,192-vector sample.

| Split | Target / selected ef | Base development recall | NGFix development recall | Base ms | NGFix ms |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | 95% / 200 | 97.54% | 97.72% | 1.908 | 1.914 |
| 1 | 95% / 200 | 97.54% | 97.70% | 1.905 | 1.915 |
| 2 | 95% / 200 | 97.54% | 97.58% | 1.908 | 1.918 |
| 0 | 99% / 400 | 99.42% | 99.46% | 3.129 | 3.148 |
| 1 | 99% / 400 | 99.42% | 99.46% | 3.128 | 3.149 |
| 2 | 99% / 400 | 99.42% | 99.42% | 3.134 | 3.137 |

At ef 200, NGFix improves 6/4/2 development queries in splits 0/1/2, with no recall
losses. Gains are 0.18/0.16/0.04 percentage points. The minimum remains 89% in all
three splits. Queries below 95% individual recall are 7 for base in each split
and 6/7/7 for NGFix. Meeting a mean target is not a per-query guarantee.

At ef 400, NGFix improves 2/2/0 queries, with no recall losses. The split 0 minimum
increases from 97% to 98%; other minima remain 97%. The complete four-ef curves and
tail counts are retained in the cohort table, rather than reporting selected
points alone. At ef 100 the development gains are 0.30/0.20/0.16 percentage points.

RFix alone has identical rankings, scores and counters to base in every 400
query/ef conditions per split, as well as the same within-run state hash. The
combined control has identical rankings/scores/counters to NGFix in all 400
conditions per split, but different raw saved-state hashes. No additional retrieval
benefit is observed from RFix here. Raw hash inequality is not proof of different
active edges, as discussed below.

At the selected settings, NGFix's observed mean times are 0.08–0.69% higher than
base. These tiny differences lack independent timing uncertainty estimates; do
not characterize them as a reliable slowdown or a fully tuned Pareto comparison.
There is no observed 10% latency improvement of the kind required by our method
promotion gate. Earlier Python-binding Epzero/FAISS timings have a different
measurement boundary and must not be directly ranked against these native calls.

## Verification

Archive: `build/longeval-ngsplit-7148412-evidence.tar.gz`, SHA-256
`93d0e457ec6a2cec5c8365917bccd985cda531fa5eb76a4a81b9effe8afa3d95`.
All 46 original returned files are preserved. The returned manifest matches the
submitted package byte for byte, and all 506 packaged files verify. The frozen
protocol, reconstructed sample, query order, fitting-only reference matrices,
snapshot ingestion, sources and both verifier identities match local evidence.

Independent local verification covers 4,800 real and 256 synthetic conditions,
15,168 timed calls, and 505,600 returned score occurrences recomputed against
float64 dot products. Occurrences include repeated vector pairs. It also recomputes
all role-specific recalls, tuning selections and development target decisions.
All 19 PACE Python tests and three native malformed-input rejection checks pass.
The synthetic fixture recovers exact top 100 at ef 800 in every role/control.

All native executions record affinity `[0]`, Xeon Gold 6226 and GCC 13.3.0. The
reported executable hash matches both the rejection tests and previous job 7141656:
`59cf56197f6c1ae72ce6a822116d49c6a983eae09a43a408904d34500ed982de`.
The runtime CWD is `/tmp`, with local XFS mount evidence. The complete log has
**zero** working-directory fallback warnings, compared with five in the prior run.
This confirms the corrected launch behavior on this job; it does not retrospectively
establish the site-level cause or timing impact of the previous warnings.

## Saved-state identity limitation and diagnostic

The base files have equal sizes but different SHA-256 values in all three splits,
also differing from job 7141656. An additional cross-run byte-identity assertion
therefore **failed** during analysis. The mismatch is retained in the report;
it was not converted into a claim of identical graphs.

The pinned upstream `ngfixlib/graph/node.h` constructor allocates neighbor capacity
without initializing every unused slot. `StoreIndex` at lines 262–267 writes the
entire capacity, including inactive slots. A local C++ diagnostic using that
unchanged header creates two nodes with identical metadata and sole active neighbor,
intentionally fills inactive slots differently, and obtains different serialized
bytes. The first differing byte is 8; both node records are 20 bytes. This proves
that raw serialization need not uniquely describe active graph structure. The
diagnostic does not execute ANN kernels or read the unreturned PACE graphs.

Source inspection also finds that `HNSW_NGFix` allocates vector/delete-flag storage
without initialization; `SetData` fills vector coordinates but not their preceding
delete flag. We have no returned flag census. Future adapters should explicitly
initialize/validate those flags and export canonical fingerprints of vectors,
live flags, ordered active adjacency and meaningful repair metadata. Preserve the
original implementation as a separate version when doing so.

All 1,200 base query/ef ranking, score and distance-counter conditions across the
three splits exactly reproduce job 7141656 after query-ID alignment. Within each
run, the four controls reload one common saved base, and the node verified byte
identity before/after query execution. These facts support the reported observed
comparisons, but do not prove whole-graph identity across builds or identify the
actual bytes responsible for the hash differences. Index and executable bytes
were not returned; the local analysis cannot rehash them. Do not promote these
sample results into a full-data graph-determinism claim.

Diagnostic evidence: `build/ngfix-serialization-slack-check-01/report.json` and
`build/ngfix-serialization-slack-check.cpp`. Original upstream source is unchanged.

## Costs and interpretation

Base construction takes 6.915–6.920s. NGFix repair loops take 47.16–49.58ms; RFix
takes 6.48–6.80ms; both take 53.32–55.73ms. These omit exact-reference preparation,
graph loading and serialization. NGFix adds 8,904–10,104 serialized bytes to the
84,571,512-byte base. Whole sample-worker peak RSS is 420,392–420,404KiB, including
four graphs and verification work; this is not per-index memory.

Compared with the prior 50-query fit, this screen changes the fitting budget to 30
as well as separating tuning. It does not isolate a causal effect of tuning
separation alone. Thirty fitting queries and one small sample cannot establish
superiority over a properly trained competitor or reproduce the paper's scale.

The immediate fitting/tuning protocol defect is addressed. The remaining research
priority is a distinct coverage method with matched-work and native performance
gains, followed by broader training and independent datasets. No new PACE sweep
of the rejected prototypes is justified by this result. Before future graph-based
claims, add the explicit flag initialization and canonical state diagnostics above.
All 381 final held-out retrieval outcomes remain untouched.

Reproduce analysis with `build/analyze-longeval-ngsplit-7148412.py` after the bounded
receipt script and local input preparation. Output directory:
`build/longeval-ngsplit-comparison-7148412/`, containing four verification reports,
4,800 per-query rows, 144 cohorts, 108 paired comparisons, 24 selected points,
stage/resource tables and the machine-readable report. The prior 33-file protocol
and feasibility state was preserved and reverified before status edits in
`build/longeval-disjoint-and-region-01-preserved.tar.gz`, SHA-256
`f4eb3a580dfbcd08581781daf4aad7d653e168a8b77ebd1ea91247f1c7ad084e`.
