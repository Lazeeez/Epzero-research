# Focused development diagnostic after the three-seed matrix

Status: implemented and tested; PACE submission is the next step. No full-data diagnostic result
is claimed yet. Implementation is failure_diagnostic.py with an optional post-snapshot observer
in trajectory.py. Ten native fixture tests pass; CLI input validation and native-mismatch
rejection also pass using a clearly marked local-runtime fixture.
This is a post-hoc mechanism investigation, not a primary efficacy comparison or retuning of
the existing result matrix. No new broad seed sweep is proposed at this point.

## Question and controls

Determine why Epzero seed 11's snapshot-3 incremental/replay graph misses the exact top 100
entirely for two development queries, while seeds 29 and 47 and FAISS do not. Distinguish an
insufficient search budget, an unfavorable upper-layer route, limited graph reachability and
an implementation problem without assuming any of these explanations beforehand.

Reproduce the original Epzero incremental trajectories for seeds 11 and 29 with the same
data, ID order, model outputs, graph settings and one-thread adapter. Before interpreting any
new diagnostic, require each snapshot's topology fingerprint and ef=200 development run hash
to equal its previously verified result. A mismatch is a reproduction failure to investigate,
not evidence for a mechanism. Record any additional instrumentation/source identity separately.

## Observations to collect

1. Record development and calibration rankings separately at a declared effort grid
   [100, 200, 400, 800, 1600, 3200]. Preserve both cohorts and every query. Label this diagnostic
   effort sweep independently from the earlier selected-effort comparisons; do not replace
   original outcomes or choose a new primary threshold from these results.
2. Trace the two previously failing queries at each effort and snapshot, plus a deterministic
   control sample selected by query ID before inspecting trace results. Record entry label/layer,
   upper-layer transitions, layer-0 route, expansions, distance evaluations and resulting IDs.
   Confirm traced results equal ordinary search results at the same effort. The existing
   `Index.trace` binding already exposes these fields; its profile-derived options must be
   checked for equivalence to the explicit uncompressed baseline search.
3. Compare the exact neighbors' membership and graph neighborhoods with the visited path.
   If failures persist at higher effort, measure directed reachability from the relevant
   search entry points. A trace that does not visit a node does not prove that node unreachable.
   Save compact statistics and decisive evidence rather than inferring disconnected components
   from recall alone.
4. Record the actual FAISS native-module hash for future cross-backend diagnostics, and save
   calibration rankings so subsequent target claims can be independently recounted. These
   provenance improvements do not retroactively change what the original bundles contain.

Implementation details: trace the two target queries and the two other development IDs with
the lowest SHA256(longeval-trace-control-v1:ID). Save the first 20,000 events per trace, hash
and summarize the entire event stream, and record saved/omitted counts. Compress prefixes on
node-local work storage and permit at most 4 MiB per copied prefix. At snapshot 3, if either
target's recall remains below 0.95 at ef=3200, exhaust layer-0 directed reachability from each
distinct affected query's layer-0 start. Save reachable counts, a digest of visited-label flags,
and exact-neighbor reachability for all four traced queries. Re-fingerprint the index after
diagnostics and fail if it changed. All diagnostic effort-grid rankings are retained separately
for development and calibration; they do not replace the original experiment's operating point.

## Execution and interpretation constraints

Keep archives, extracted vectors, native-index saves and large trace/graph work files on
compute-node temporary storage. Persist bounded summaries, hashes and compressed diagnostic
evidence in the shared result directory. Report any trace truncation explicitly. Preserve the
original engine and 18-run development matrix until a separately identified diagnostic or fix
is tested; do not silently patch the algorithm and combine results under one experiment label.

Use meaningful fixture tests for trace/search parity and fingerprint matching before another
full-data submission. Reproduce the failing graph before attributing improvement to any change.
If a native defect is found, report the original failure and separately test a correction across
both affected and unaffected seeds; novelty must then be reassessed around the demonstrated
finding. If it is ordinary seed/order sensitivity, investigate its mechanism and practical
impact without describing three seeds as a reliable population failure-rate estimate.
