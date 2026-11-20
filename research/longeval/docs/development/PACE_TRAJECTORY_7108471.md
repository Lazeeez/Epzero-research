# First full-data trajectory: PACE job 7108471

Status: returned artifacts verified locally, including independent development recall and
returned-score recomputation. This supersedes the pre-submission status
in DEVELOPMENT_TRAJECTORY_STATUS.md without changing that checksummed transfer bundle.

Epzero incremental construction with seed 11 completed across all three snapshots. Frozen
and calibration-selected search effort were both ef=200, so their exported run files match.

| Snapshot | Documents | Calibration recall@100 | Development recall@100 | Build/addition time |
|---|---:|---:|---:|---:|
| 1 | 869,902 | 0.9550 | 0.9642 | 1,425.47 s |
| 2 | 1,322,720 | 0.9558 | 0.9632 | 785.50 s |
| 3 | 1,661,900 | 0.9552 | 0.9200 | 577.49 s |

The calibration target is met on the 50 calibration queries; it is not a guarantee that the
separate 50 development queries achieve 0.95. All development recalls above were recomputed
directly from saved ranked IDs against the independently prepared full-corpus exact references.
The saved calibration grid supports the selected effort, but calibration rankings were not
exported, so those recall values cannot be independently recounted from this bundle.

The full vector audit passed again: every vector was finite, nonzero and within the declared
unit-norm tolerance; no shared vector changed bitwise and no document IDs were removed.
Both the original 127-file source manifest and 44-file additive manifest passed on PACE.

Reported job wall time was 01:17:52, including tests, downloads, extraction, audit and trajectory.
This is not index-construction time alone. The exclusive allocation reports 24 CPUs and 64 GB
RAM on a single compute node, partition coc-cpu. The epilog's CPU-time and memory fields are not
being interpreted as measured index costs. The process resource files report preparation
26:56.56, trajectory 50:16.05 and trajectory peak RSS 18,103,000 KiB (17.2644 GiB), with zero
swaps. This is whole-worker memory, including the index, Python state and harness. Epzero
reserves final index capacity from snapshot 1. Instrumented build/addition time totals
46:28.47; it excludes allocation, validation, fingerprinting and query evaluation.

Local evidence: `build/longeval-pace-trajectory-evidence-7108471/`. Its `local-verification.json`
records all returned file hashes, verification code identity and recomputed metrics. Verification
passed for the nine native tests, submitted additive manifest, source hashes, unchanged native
binary/package identities from bootstrap, query inputs, vector/text audit identities, exact
references, chronological insertion/live-set digests and parent-state chain. All 15,000 frozen-run
scores were checked against original float32 document/query vectors with float64 accumulation;
maximum absolute discrepancy was below 9.69e-8. Identical matched-recall files cover the other
three exports. Graph hashes are bound by the returned records, not independently reconstructed:
the full native graph was not saved.

Returned archive SHA-256: `09ff77cabdc43eb1769040f3369ac8f7e1907f9b26e964b01842a3d378d7c280`.
Trajectory report SHA-256: `b69f377e0e43ec4d9670d196e97f6229a81baeb671292a54d97bc7261b03c418`.

## Development failure characterization

Snapshot 2 to 3 development recall drops by 4.32 percentage points. Two queries fall from
1.00 and 0.92 to zero recall, contributing 3.84 points (88.9% of the net decline). This is a
post-hoc diagnostic; no queries are excluded or substituted in the reported population.

| Query ID | Exact top-100 retained from snapshot 2 | Exact score at rank 100 | Best returned ANN score |
|---|---:|---:|---:|
| 161820e44e7022cf44fd99a59bce5792 | 85 | 0.470949 | 0.293062 |
| 21e8c323e168b0813eff14fdac972f1f | 88 | 0.452028 | 0.241947 |

Recomputing scores directly confirms substantial misses, not just alternate IDs tied at the
top-100 boundary. The exact-reference scores reproduce within 4.45e-16. The second query's
100 ANN results are all new snapshot-3 documents even though 88 exact neighbors predate that
snapshot. These observations do not establish graph disconnection, an entry-point problem or
a general index-history effect. Evidence: `zero-recall-diagnostic.json` beside the verification.

Next run same-seed replay and shuffled controls with the existing, unchanged job and source
bundle. Compare graph/order hashes and per-query outputs at the same snapshot; a cross-snapshot
recall change alone confounds corpus growth and index construction. Follow with the independent
FAISS backend and additional seeds. Statistical inference, primary-test evaluation and novelty
remain unestablished.
