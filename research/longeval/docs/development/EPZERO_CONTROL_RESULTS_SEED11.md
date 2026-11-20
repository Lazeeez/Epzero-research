# Verified Epzero construction controls, seed 11

The incremental (7108471), replay (7111579) and shuffled (7111580) full-data development
trajectories are complete and their returned evidence has been checked locally. This supersedes
the pending-control status in PACE_TRAJECTORY_7108471.md and DEVELOPMENT_CONTROL_RUNS.md.

## What the controls establish

At every snapshot, replay has the same topology fingerprint, chronological insertion-order
digest and live set as incremental construction. All six exported ranking files have identical
SHA-256 hashes between those two arms. Rebuilding from scratch with the same order therefore
reproduces the snapshot-3 failures in this run. There is no observed additional effect of keeping
the native index alive across snapshot boundaries under this configuration.

Shuffling the construction order changes the topology and ranked results. With ef held at 200,
snapshot-3 mean development ANN recall rises from 0.9200 to 0.9548. This supports construction-order
sensitivity for this backend, seed and development population. It does not establish a general
degradation mechanism, statistical significance, or a novel published contribution.

## Same-effort development comparison

All values below are mean ANN recall@100 on the same 50 development queries, independently
recounted against the frozen float32 exact references. Search effort is ef=200 throughout.

| Snapshot | Incremental | Replay | Shuffled | Shuffled minus incremental |
|---|---:|---:|---:|---:|
| 1 | 0.9642 | 0.9642 | 0.9634 | -0.08 percentage points |
| 2 | 0.9632 | 0.9632 | 0.9624 | -0.08 percentage points |
| 3 | 0.9200 | 0.9200 | 0.9548 | +3.48 percentage points |

At snapshot 3, shuffled improves 17 queries, worsens 15 and leaves 18 unchanged. The two
previously identified zero-recall queries recover as follows:

| Query ID | Incremental / replay | Shuffled, ef=200 | Shuffled, ef=400 |
|---|---:|---:|---:|
| 161820e44e7022cf44fd99a59bce5792 | 0.00 | 0.98 | 1.00 |
| 21e8c323e168b0813eff14fdac972f1f | 0.00 | 0.91 | 0.96 |

Those two queries contribute +3.78 percentage points to the same-effort aggregate difference;
the remaining 48 contribute -0.30 points. This is a post-hoc explanation of the aggregate,
not a new evaluation cohort or evidence of broad uniform improvement. No queries were removed.

## Calibration-selected comparison

The separate 50 calibration queries select ef=200 for incremental/replay at every snapshot.
Shuffled selects 200, 400 and 400; its calibration recall at ef=200 is 0.9500, 0.9480 and 0.9448.
At the selected efforts, shuffled development recall is 0.9634, 0.9806 and 0.9750.

The snapshot-3 comparison of 0.9750 versus 0.9200 changes both construction order and selected
search effort. It is not a same-cost order effect. The discrete grid satisfies a minimum
calibration target, not equality of achieved recall; its target does not guarantee recall on
the development cohort. No higher-effort development rankings exist for incremental/replay in
these bundles, so their behavior at ef=400 has not been established here.

## Verification and resources

Both new bundles passed the nine native tests on PACE. Local verification checked the submitted
44-file overlay, original source identities, unchanged native binary/packages, exact references,
query roles/precision, vector and text audit identities, snapshot membership and expected seeded
insertion order. All six runs in each new bundle were recounted and their scores recomputed
against original document/query vectors. Maximum absolute score discrepancy was below 9.69e-8.

Calibration choice was checked against the reported grid, but calibration rankings were not
saved, so their recall could not be independently recounted. Native graphs were also not saved:
topology comparisons use their recorded fingerprints, not an independent graph reconstruction.

| Arm | Trajectory wall time | Sum of build/addition stages | Peak worker RSS |
|---|---:|---:|---:|
| Incremental | 50m 16s | 46m 28s | 17.264 GiB |
| Replay | 1h 50m 36s | 1h 46m 55s | 17.250 GiB |
| Shuffled | 1h 55m 31s | 1h 51m 44s | 17.400 GiB |

Jobs used separate exclusive Gold 6226 nodes, one computation thread and the same 64 GB
allocation. These are instrumented process observations, not matched-cost or publication-quality
latency comparisons. Every new large download and extraction remained in compute-node temporary
storage. Query populations, configurations and reported outcomes have not been retuned.

Evidence is under `build/longeval-pace-trajectory-evidence-{JOB_ID}/`. The paired comparison and
all input receipt hashes are in `build/longeval-epzero-controls-seed11/comparison.json`, generated
by `build/compare-longeval-epzero-controls-11.py`.

| Returned archive | SHA-256 |
|---|---|
| Replay 7111579 | `2587b0ac0dd68fc7b691158a644688ec6f6c5182dd1c55f781ad27129dff515e` |
| Shuffled 7111580 | `3a356df21107b60802c7224b0ce1c739bbdc6cfde46feef393fc3b2dd3df5e16` |

## Next research step

Repeat all three construction modes using the independent FAISS HNSW backend, seed 11, with
the existing frozen development configuration. Then repeat seeds 29 and 47 (already declared
in the earlier pilot plan), inspect whether failures recur, and investigate mechanisms without
changing the reported original results. Primary inferential choices, practical-effect margins,
test comparisons, remaining IVF/resource controls and novelty assessment are still outstanding.
