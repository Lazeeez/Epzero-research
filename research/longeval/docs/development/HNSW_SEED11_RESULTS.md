# Two-backend HNSW development results, seed 11 — 2026-11-17

All six trajectories have completed and passed local artifact verification: three construction
modes each for Epzero and FAISS HNSW. They use the same archived document vectors, CUDA float32
training-query embeddings, 50 calibration queries, 50 development queries, exact references,
M=16, efConstruction=100 and construction seed 11. No final-test queries or judgments enter
this comparison. These results supersede the pending-FAISS status in EPZERO_CONTROL_RESULTS_SEED11.md.

## Fixed-effort recall

Mean development ANN recall@100 with ef=200:

| Backend | Construction | Snapshot 1 | Snapshot 2 | Snapshot 3 |
|---|---|---:|---:|---:|
| Epzero | Incremental | 0.9642 | 0.9632 | 0.9200 |
| Epzero | Replay | 0.9642 | 0.9632 | 0.9200 |
| Epzero | Shuffled | 0.9634 | 0.9624 | 0.9548 |
| FAISS HNSW | Incremental | 0.9636 | 0.9606 | 0.9560 |
| FAISS HNSW | Replay | 0.9636 | 0.9606 | 0.9560 |
| FAISS HNSW | Shuffled | 0.9634 | 0.9610 | 0.9530 |

For both implementations, replay exactly matches incremental construction's recorded index
fingerprint, insertion order and all six ranked-output file hashes. Epzero fingerprints topology;
FAISS fingerprints its canonical serialization with efSearch standardized to 200. The shared
external ID sets and construction sequences also match across implementations.

Thus, preserving the native index across the three snapshot boundaries introduces no additional
observed difference from rebuilding in the same order in these runs. This result is limited
to the tested construction settings; it is not a claim about deletion, updates, other seeds or
other persistence strategies.

The two severe Epzero failures do not reproduce in FAISS for seed 11:

| Development query | Epzero incremental/replay | FAISS incremental/replay | Epzero shuffled | FAISS shuffled |
|---|---:|---:|---:|---:|
| 161820e44e7022cf44fd99a59bce5792 | 0.00 | 0.99 | 0.98 | 0.98 |
| 21e8c323e168b0813eff14fdac972f1f | 0.00 | 0.93 | 0.91 | 0.91 |

At snapshot 3, changing order raises Epzero mean recall by 3.48 percentage points, while FAISS
changes by -0.30 points. This does not support a general benefit from shuffling or a general
ANN aging claim. The severe failure is currently localized to the tested Epzero seed/order
configuration. Its mechanism and recurrence remain unresolved; neither an engine defect nor
graph disconnection has been established. Query-specific analysis is post-hoc and does not
replace or exclude any development queries.

## Calibration-selected operating points

Incremental and replay select ef=200 at every snapshot for both backends. Shuffled selects
[200, 400, 400] for Epzero and [200, 200, 400] for FAISS. At snapshot 3, FAISS shuffled's
calibration recall is 0.9430 at ef=200 and 0.9708 at ef=400; development recall at the selected
ef=400 is 0.9762. Epzero shuffled reaches 0.9750 development recall at ef=400.

These selected-effort comparisons combine order with changed search effort. The discrete grid
meets a minimum calibration target rather than equating achieved recall. Even equal ef across
backends does not establish equal compute, latency or memory. The fixed-effort table above
keeps effort unchanged within each backend's construction comparison.

## Returned evidence and resource observations

FAISS jobs: incremental 7114741, replay 7114742, shuffled 7114743. Each passed nine native
trajectory/memory tests on PACE. Local verification checked the submitted bundle, source
identities, package versions, original audited IDs/vectors, declared seeded order and parent-state
records, query identities and precision, exact-reference hashes and all six exported runs.
Recall was independently recounted from ranked IDs; all 30,000 exported scores per FAISS job
were recomputed against the original vectors. Maximum absolute discrepancy was 1.613e-7.

| FAISS mode | Trajectory process wall time | Peak worker RSS |
|---|---:|---:|
| Incremental | 51m 41.81s | 20.854 GiB |
| Replay | 1h 51m 57s | 21.041 GiB |
| Shuffled | 1h 56m 29s | 21.219 GiB |

The existing 64 GB exclusive allocation is adequate for these observed runs. Whole-worker
memory includes inputs and instrumentation; these are not index-only measurements. Jobs used
different physical Gold 6226 nodes. No publication-quality performance comparison is claimed.
Every large download and extraction stayed in compute-node temporary storage.

Limitations of the evidence chain: calibration rankings and full native indexes were not saved.
Calibration selection can be checked against the reported grid, but its recall cannot be
independently recounted from these bundles. Index fingerprints can be compared, not independently
reconstructed. The harness records an Epzero binary hash and FAISS package version; it does not
record a FAISS binary hash. The unchanged Epzero binary and package versions match bootstrap.

| Returned FAISS archive | SHA-256 |
|---|---|
| 7114741 | `61aeb0ecd35e080bc0769b1dc370251f453c8911a5ce887dc28fdcb4d8b55dc8` |
| 7114742 | `274bc759e4e114f4a671f36a6cc3828004a56f2b39c6739d3d33b7782c546839` |
| 7114743 | `2fe557cc7d800982e0782fbdf2dfb2d67fe63130639ad70f7c362f45342e3162` |

Receipts are in `build/longeval-pace-trajectory-evidence-{JOB_ID}/local-verification.json`.
The six-arm comparison is `build/longeval-hnsw-controls-seed11/comparison.json`, generated by
`build/compare-longeval-hnsw-seed11.py`. It binds the verification/report/archive hashes and
retains paired per-query differences. Verification code is `build/verify-longeval-faiss-seed11.py`.

## Next experiment batch

Repeat both backends and all three construction modes at seeds 29 and 47, which were already
listed with seed 11 in the earlier pilot configuration. Keep the configuration, cohorts, exact
references and outcome reporting unchanged. This is 12 additional development trajectories;
it checks seed dependence without selecting replacement queries or a favorable seed. The
current PACE bundle already accepts these seeds, so no new upload is required.

Inspect every run, including target misses and failures. Mechanism diagnostics, remaining IVF
and resource controls, practical-effect margins, the primary inferential plan, held-out test
evaluation, novelty assessment and paper writing remain outstanding. None of the current
descriptive results establishes significance or publication readiness.
