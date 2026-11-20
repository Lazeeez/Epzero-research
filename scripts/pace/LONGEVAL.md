# LongEval research jobs on PACE

The LongEval study (`research/longeval/`) runs on the PACE ICE cluster. Its scripts are separate
from the engine benchmarks in this folder, so its pinned environment and data never overwrite the
older `.pace-bench` setup. Every job is described by a runbook, with the exact commands to submit it
and to collect its evidence.

## Rules every job follows

- **Data stays on the node.** Large downloads, extracted vectors and indexes stay on the
  compute node's temporary disk. Only bounded evidence (reports, run files, checksums) is copied
  back, and none of it is committed.
- **One thread per index.** Construction, search and BLAS run single-threaded. Parallelism comes
  from running independent jobs.
- **One CPU type.** CPU jobs run on exclusive Gold 6226 nodes, and jobs on different CPU types never
  overlap.
- **Self-checking packages.** Each package verifies its own checksums, runs its tests and checks
  its inputs before any download. The local analysis then re-verifies the evidence it returns.

## Jobs, in order

| Runbook | Job script | Packager | Purpose | Jobs |
| --- | --- | --- | --- | --- |
| [LONGEVAL_BOOTSTRAP.md](LONGEVAL_BOOTSTRAP.md) | `longeval-bootstrap.sbatch` | — | Environment and native-build bootstrap | 7098485 |
| [LONGEVAL_AGREEMENT.md](LONGEVAL_AGREEMENT.md) | `longeval-agreement.sbatch`, `longeval-encoder.sbatch` | — | Document-encoder agreement with the released vectors | 7106389 |
| — | `longeval-query-sensitivity.sbatch` | — | Training-query encoding (FP32/BF16) | 7107422 |
| [LONGEVAL_TRAJECTORY.md](LONGEVAL_TRAJECTORY.md) | `longeval-trajectory.sbatch` | — | Development HNSW trajectories (3 modes × 2 backends × 3 seeds) | 7108471, 7111579–7111580, 7114741–7114743, 7122361–7122372 |
| [LONGEVAL_FAILURE_DIAGNOSTIC.md](LONGEVAL_FAILURE_DIAGNOSTIC.md) | `longeval-failure-diagnostic.sbatch` | — | Failure diagnostic on reproduced graphs | 7126117, 7126118 |
| [LONGEVAL_RETENTION.md](LONGEVAL_RETENTION.md) | `longeval-retention.sbatch` | `package-longeval-retention.py` | Candidate-retention experiment | 7128488, 7128489 |
| [LONGEVAL_WORK_BUDGET.md](LONGEVAL_WORK_BUDGET.md) | `longeval-work-budget.sbatch` | `package-longeval-work-budget.py` | Full-cohort search-work diagnostic | 7130605–7130607 |
| [LONGEVAL_IVF.md](LONGEVAL_IVF.md) | `longeval-ivf.sbatch` | `package-longeval-ivf.py` | Development IVF maintenance controls | 7133623–7133625 |
| [LONGEVAL_NATIVE_COST.md](LONGEVAL_NATIVE_COST.md) | `longeval-native-cost.sbatch` | `package-longeval-native-cost.py` | Development native HNSW timing | 7139467–7139469 |
| [LONGEVAL_NGFIX.md](LONGEVAL_NGFIX.md) | `longeval-ngfix.sbatch` | `package-longeval-ngfix.py`, `package-longeval-ngfix-v03.py` | NGFix/RFix integration screen | 7141656 (7141318 failed at launch) |
| [LONGEVAL_NGFIX_DISJOINT.md](LONGEVAL_NGFIX_DISJOINT.md) | `longeval-ngfix-disjoint.sbatch` | `package-longeval-ngfix-disjoint.py` | Independently tuned NGFix screen | 7148412 |
| [LONGEVAL_PSEUDOQUERY.md](LONGEVAL_PSEUDOQUERY.md) | `longeval-pseudoquery.sbatch` | `package-longeval-pseudoquery.py` | Pseudo-query encoding (GPU) | 7151090 |
| [LONGEVAL_TEST_QUERIES.md](LONGEVAL_TEST_QUERIES.md) | `longeval-test-queries.sbatch` | `package-longeval-test-queries.py` | Held-out test-query encoding (GPU) | 7164678 |
| [LONGEVAL_HELDOUT.md](LONGEVAL_HELDOUT.md) | `longeval-heldout.sbatch` | `package-longeval-heldout.py` | **Held-out arms under the frozen protocol** | 7168978–7168992 |
| [LONGEVAL_HELDOUT_COST.md](LONGEVAL_HELDOUT_COST.md) | `longeval-heldout-cost.sbatch` | `package-longeval-heldout-cost.py` | Held-out native timing (secondary) | 7178503–7178505 |
| [LONGEVAL_HELDOUT_EXT.md](LONGEVAL_HELDOUT_EXT.md) | `longeval-heldout-ext.sbatch` | `package-longeval-heldout-ext.py` | Shuffled order and ef sweep (exploratory) | 7179544–7179549 |

The shared helpers are:

- `longeval-node-env.sh`: the node-local staging directory;
- `longeval-setup.sh` and `longeval-gpu-setup.sh`: environment setup;
- `longeval-stage.py`: staging;
- `longeval-preflight.sh`: a read-only inventory of the cluster;
- `longeval-gpu-verify.py`: verifies the source manifest;
- `longeval-cpu.sbatch`: the early pilot, audit and feasibility jobs.

The section below keeps the original setup notes.

## Preflight and staging

Run `bash scripts/pace/longeval-preflight.sh` on a login node. It reads available partitions,
GPU features, account/QOS associations, current jobs and scratch space; it performs no installations
or benchmark work. Choose actual partition/account/CPU/GPU constraints from those results.

Use a new research checkout under scratch, at the recorded base Git commit, and overlay the local
modified research/engine source with its per-file SHA-256 manifest. A Git clone alone omits the
current uncommitted changes. Preserve the existing benchmark checkout and environments.

Use a dedicated research Python environment with `research/longeval/requirements.txt` and a local
Epzero source build. Stage model files and packages through the login/data-transfer path before
submitting offline compute jobs. Do not run full audits, index builds, model inference or tests
on login nodes. All native build/test work belongs in an allocation. Confirm scratch quota before
staging: dense archives plus extracted payloads need about 80 GB, and the encoder adds about 8 GB,
before packages, indexes and result files. File-level checksums must survive transfer.

The existing `bench-env.sh` recreates `.pace-bench`; it is not the setup command for this work.
The current LongEval build recipe and native tests are documented in
`research/longeval/docs/data/NATIVE_VALIDATION.md`. Keep construction/search/BLAS at one thread for the
declared history controls, even on a many-core node. Parallelism is across independent jobs.

## CPU jobs

`longeval-cpu.sbatch` runs one of `pilot`, `audit`, or `feasibility` in a fresh worker, reserves an
exclusive node, requests 64 GB RAM and four hours, and binds the Slurm task's CPU and local memory.
These are prepared defaults that must be checked against the actual allocation before submission.
Set an existing source-built Python path and a new output directory. Example after preflight:

```sh
export LONGEVAL_PYTHON="$PWD/.pace-longeval/bin/python"
export LONGEVAL_MODE=feasibility
export LONGEVAL_DATA="$PWD/data/longeval/dense/snapshot-1"
export LONGEVAL_AUDIT="$PWD/data/longeval/audit/snapshot-1/scan.json"
export LONGEVAL_BACKEND=epzero
export LONGEVAL_COUNT=64000
export LONGEVAL_OUTPUT="$PWD/pace-results/longeval/NEW-RESOURCE-RUN"
sbatch --partition=CHOSEN_CPU_PARTITION --account=CHOSEN_ACCOUNT scripts/pace/longeval-cpu.sbatch
```

Use `LONGEVAL_DATASET` for the synthetic pilot. For `audit`, set `LONGEVAL_DATA` to the parent of
the three verified extraction directories. A resource probe uses held-out document vectors;
it does not become a real-query relevance experiment by running on PACE. Larger IVF runs still
require an explicit bounded training contract. Current full-corpus research trajectories are
not implemented by these job templates.

## GPU encoder diagnostics

`longeval-encoder.sbatch` requests one GPU, four CPUs, 32 GB host RAM and one hour. Select a GPU
with at least 16 GiB VRAM and BF16 support, and prepare a CUDA-enabled Torch 2.9.0 environment with
the matching driver/runtime compatibility checked on that node. The local CPU-only wheel is not
suitable for this job. Record the exact installed package lock and GPU identity.

Set `LONGEVAL_ENCODER_PYTHON`, `LONGEVAL_MODEL`, `LONGEVAL_DOCUMENT_SAMPLE` and a new
`LONGEVAL_OUTPUT`, then submit with the chosen GPU partition and account. The script works offline,
uses the pinned model files, records hardware and dependencies, and applies no quantization.
CUDA outputs require their own agreement check; they are not assumed identical to CPU outputs.
The provisional local document sample can support an early diagnostic only; source verification
and agreement with the archived vectors remain research gates.
