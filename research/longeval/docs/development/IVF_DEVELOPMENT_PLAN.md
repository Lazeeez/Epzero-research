# Full-data IVF development controls

This additive experiment uses the existing audited insert-only snapshots, CUDA FP32
training-query vectors, and full-corpus float64 exact references. It does not read
test queries or relevance judgments. The full protocol is `ivf-development.json`.

## Arms

For each seed 11, 29, 47 run these in separate sequential processes:

- `frozen_incremental`: train on snapshot 1, retain the index, append lexical additions.
- `frozen_rebuild`: reproduce the same snapshot-1 trained template; at every snapshot
  allocate a fresh index with those centroids and replay the chronological insertion order.
- `refreshed_rebuild`: train on each current snapshot and rebuild with that same order.

The first two isolate keeping the object alive versus rebuilding under identical training
and insertion history. Their serialized states and rankings must agree at each snapshot.
All three snapshot-1 states must agree. A mismatch invalidates the control execution.
Retraining changes both the sample and centroids; this is a maintenance-policy contrast,
not identification of a centroid-only causal effect. No favorable outcome is required.

## Fixed choices

IndexIVFFlat with exact flat IP coarse quantizer; full original 2,560-dimensional float32
vectors; no compression, normalization or max-codes scan cap. `nlist=1024`, 20 spherical
k-means iterations, one restart, explicit seed, single computation thread. Sample 65,536
current live IDs by the smallest SHA-256 priorities of
`longeval-ivf-training-v1:SEED:ID`, then train in lexical ID order. This is independent of
matrix row order and query outcomes. FAISS internal subsampling is disabled by a sufficient
max-points-per-centroid setting. Persist sample IDs, centroid arrays, and their hashes.

The selected training matrix is 640 MiB; gather it in 1,024-row batches. Only one full
native index lives in each process, plus a small empty trained template. New IDs are
lexically ordered within each snapshot; full rebuilds replay the concatenated additions.

Calibrate mean ANN recall@100 to 0.95 on the 50 calibration queries. Evaluate the complete
probe grid 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024. Choose the smallest feasible point
meeting the target; if none meets it, retain the best feasible recall, tie-breaking toward
lower probes, and mark target failure. Incomplete query results make the entire point
infeasible; retain incomplete-query IDs and do not silently omit them from the population.

On the 50 development queries export three regimes: the snapshot-1 selected probe count,
the independently current-calibrated probe count, and all lists. Frozen nprobe does not
imply fixed work. All-list search should match exhaustive neighbors except numerical
boundary substitutions, which are reported explicitly with a 1e-5 absolute score tolerance.
All returned scores are independently checked against stored vectors using float64 dots.

## Costs and integrity

Native searches use one query per call, one warmup and three timed repeats for development
regimes. Record individual nanoseconds including the Python/native boundary. Calibration
grid uses one timed repeat after warmup. Explicit coarse-list checks are outside timing.
Counters are reset per call. Cross-check FAISS scanned-vector counts against the sum of
probed-list sizes; report 1,024 coarse-centroid comparisons separately and their total.
No concurrent FAISS searches run. These are warm-cache native-call observations, not service
throughput or a comparison to earlier instrumented HNSW timing.

Record training, allocation, insertion, and verification separately using the existing
worker resource ledger. RSS includes vectors, ID maps, and the harness; it is not isolated
index memory. Measure full-process resource use per arm with `/usr/bin/time`. Search must
leave the native serialization unchanged after restoring nprobe to 1. Read actual list
labels to verify membership and reject duplicates. Hash every exported bounded artifact.

Use a single exclusive Gold 6226 CPU node per seed, 64 GB RAM, 160 GB temporary disk,
12-hour wall limit. Stage the full corpus once per seed on compute-node temporary disk,
run three arms sequentially, then retain only bounded results. No large shared downloads.
Tests and pinned-input preflight precede corpus downloads.

FAISS background: [accuracy and clustering FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)
and [implementation notes](https://github.com/facebookresearch/faiss/wiki/Implementation-notes).
The executed installed version and binary identities are recorded in provenance; current
online documentation is not a substitute for validating that installed implementation.
