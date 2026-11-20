# HNSW native-call cost comparison

The original baselines predate the observer hook already validated by the earlier
diagnostic jobs. Preflight pins both the original trajectory-module hash and that
known observer revision; it does not exempt arbitrary changes to the module.

This closes the timing-protocol gap left by the verified IVF experiments. It is a
development measurement, not a held-out evaluation or final publication claim.

Reproduce the six verified incremental HNSW trajectories: Epzero and FAISS at seeds
11, 29, 47, three full snapshots each. Original state hashes, frozen and calibration-
selected ranking files must match. Use the same immutable engine, source, query vectors,
exact references, construction settings and insertion history. No search tuning changes.

For each snapshot, time all 50 development queries at original frozen ef=200, current
calibration-selected ef, and diagnostic wide ef=3200. Keep duplicate settings as separate
repeated observations rather than dropping one condition after inspecting timings.

The protocol matches the IVF single-query measurements: one untimed warmup immediately
followed by three timed native calls, original query-contract order, one thread, exclusive
Gold 6226 allocation, core/NUMA binding, 64 GB RAM. Record each native-call duration in
nanoseconds. Query conversion, result validation, graph tracing, fingerprints and counters'
reset/read operations are outside timing. Native-call timing includes the Python binding
and keyword-argument overhead of the library's API. It is not server throughput or cold
search latency. Summaries use per-query medians of three calls, then mean/p95 over queries.

Epzero trace rankings must equal normal search rankings; collect its distance/expansion
counts outside timing. FAISS ndis/nhops are reset/read for each native call and must repeat.
Keep their names distinct: their accounting conventions are not certified equal. A small
eight-vector fixture, for example, can report FAISS ndis=7 because the native counter need
not include every distance evaluation. Do not label a native-counter ratio an exact
cross-library work ratio. Existing complete Epzero traces remain the stronger work audit.

Restore FAISS efSearch and verify native state identity after the timing stage. No new
judgments or test queries are read. Original baseline references and independently verified
receipts are packaged. On PACE, native binary/dependency identities must match before
downloads. Small synthetic tests validate both backends and reject a changed baseline.

Run both backends sequentially per seed, reusing only the node-local corpus. Seed 29 runs
FAISS first; 11 and 47 run Epzero first. Each backend uses a fresh process. Three allocations
do not provide a randomized multi-machine timing experiment. Compare to IVF with explicit
CPU/protocol alignment and retain allocation/order limitations; do not infer precise small
speed differences or universal performance rankings.

Next after returned verification: finalize primary population/contrasts/practical margin,
freeze held-out protocol and run the held-out encoding/oracle/evaluation pipeline. The
IVF development findings are retained regardless of whether final evaluation agrees.
