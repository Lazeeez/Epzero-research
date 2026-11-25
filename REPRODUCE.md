# Reproducing the LongEval study

There are four levels of checking. The first takes a few minutes on any laptop. The last
recomputes every number in the paper from the official LongEval release.

| Level | What it shows | Needs | Time |
| --- | --- | --- | --- |
| [1](#1-one-command-check) | The code is correct on synthetic data, and the frozen protocol is untouched | Python 3.11+ | ~3 min |
| [2](#2-read-the-evidence) | How each reported number was produced and verified | a browser | — |
| [3](#3-data-and-exact-search) | The official data, its audit, and exact search on real vectors | Linux, ~85 GB disk, 24 GB RAM (it was run on 23 GB) | ~2 h |
| [4](#4-full-reproduction) | Every result: query encoding, all index arms, the held-out analysis | one NVIDIA GPU with 20 GB or more; Linux nodes with 64 GB RAM; Slurm optional | ~1 day |

LongEval data cannot be redistributed, so levels 3 and 4 download them from the official release.
The scripts pin every file by checksum and refuse anything that differs.

## 1. One-command check

```bash
git clone https://github.com/Lazeeez/ann-longeval-research.git && cd ann-longeval-research
python3 -m venv .venv && source .venv/bin/activate
pip install -r research/longeval/requirements.txt     # includes the released Epzero 0.5.0 engine
bash scripts/verify.sh
```

`verify.sh` runs three checks:

1. **Tests:** about 260 unit and regression tests. Tests marked `linux_native` skip on other
   platforms, and the `hnswlib` tests skip unless that library is installed.
2. **Frozen protocol:** checks all 55 files in the
   [freeze receipt](research/longeval/FINAL_PROTOCOL_V1_FREEZE.json) byte for byte. The protocol and
   its analysis code were hashed before any held-out query was run.
3. **Synthetic pilot:** runs the pipeline end to end on three synthetic snapshots (build every
   index, search, score, verify every artifact).

On Windows, run the same steps in WSL, or run the three commands inside `verify.sh` by hand.

## 2. Read the evidence

| Question | Where |
| --- | --- |
| What was decided before the held-out run? | [FINAL_PROTOCOL_V1.md](research/longeval/FINAL_PROTOCOL_V1.md) |
| What are the primary results, and how were they verified? | [HELDOUT_RESULTS.md](research/longeval/HELDOUT_RESULTS.md) |
| What was found after the freeze, and how is it labelled? | [HELDOUT_EXPLORATORY.md](research/longeval/HELDOUT_EXPLORATORY.md) |
| Which PACE job produced which result? | [scripts/pace/LONGEVAL.md](scripts/pace/LONGEVAL.md) and [docs/RESEARCH_LOG.md](research/longeval/docs/RESEARCH_LOG.md) |
| How does the paper position itself? | [reports/LongEval paper novelty review.md](reports/LongEval%20paper%20novelty%20review.md) |

Every PACE job wrote a checksum list (`EVIDENCE-SHA256SUMS`) with its outputs. The result
documents give the SHA-256 of each evidence archive, as it was checked locally.

## 3. Data and exact search

On Linux, from the repository root, with the level-1 environment:

```bash
# Queries and judgments (small files), then an audit of IDs, roles and judgment coverage.
python -m research.longeval fetch-release build/longeval-release/files
python -m research.longeval audit-release build/longeval-release/files --output build/release-audit.json

# The three snapshots' document vectors (39 GB of archives, ~40 GB extracted), audited vector by vector.
python -m research.longeval prepare-dense build/dense --output build/preparation
```

`prepare-dense` can be resumed: if it stops, run it again. It verifies every archive against
its pinned checksum, extracts it, and scans every vector. The data steps are described in
[INGESTION.md](research/longeval/docs/data/INGESTION.md) and
[RELEASE_AUDIT.md](research/longeval/docs/data/RELEASE_AUDIT.md).

Exact float64 search (`research/longeval/oracle.py`) then streams the vectors from disk, so it
needs little memory. The held-out exact references took 1–2.5 minutes per snapshot on one thread
for 381 queries (`python -m research.longeval.heldout exact --help`). They need the encoded
queries from level 4.

## 4. Full reproduction

The experiments ran as self-contained Slurm packages on the PACE cluster. Each package verifies
its own checksums and inputs before downloading anything. Each runbook below gives the exact
commands, resources and expected durations, in the order the results depend on each other:

| Step | Runbook | Hardware | Produces |
| --- | --- | --- | --- |
| Training-query encoding | `scripts/pace/longeval-query-sensitivity.sbatch` (see [QUERY_VALIDATION_STATUS.md](research/longeval/docs/data/QUERY_VALIDATION_STATUS.md)) | 1 GPU ≥ 20 GB | Qwen3-Embedding-4B query vectors, FP32 |
| Development trajectories | [LONGEVAL_TRAJECTORY.md](scripts/pace/LONGEVAL_TRAJECTORY.md), [LONGEVAL_IVF.md](scripts/pace/LONGEVAL_IVF.md) | 64 GB CPU node per job | RQ1/RQ3 development evidence |
| Repair screens (RQ4) | [LONGEVAL_NGFIX.md](scripts/pace/LONGEVAL_NGFIX.md) and the other method runbooks | CPU node | Equal-work repair comparisons |
| Test-query encoding | [LONGEVAL_TEST_QUERIES.md](scripts/pace/LONGEVAL_TEST_QUERIES.md) | 1 GPU ≥ 20 GB | 381 test-query vectors |
| **Held-out arms** | [LONGEVAL_HELDOUT.md](scripts/pace/LONGEVAL_HELDOUT.md) | 15 jobs, 64 GB CPU node each | The primary results |
| Held-out timing | [LONGEVAL_HELDOUT_COST.md](scripts/pace/LONGEVAL_HELDOUT_COST.md) | 3 CPU jobs | Latency and distance counts |
| Shuffled order and ef sweep | [LONGEVAL_HELDOUT_EXT.md](scripts/pace/LONGEVAL_HELDOUT_EXT.md) | 6 CPU jobs | Exploratory results |

**Analysis, run locally from the returned evidence:**

```bash
python -m research.longeval.heldout_analysis heldout INPUTS build/longeval-release/files EVIDENCE OUTPUT \
    research/longeval/trajectory-development.json research/longeval/ivf-development.json
```

Here `INPUTS` is the held-out inputs folder that the held-out packager assembles, and `EVIDENCE`
is the extracted evidence archive. Before reading any judgment, the analysis:

- recounts every run's neighbour recall against the exact references;
- checks every checksum;
- refuses evidence produced from a configuration other than the frozen one.

**Running elsewhere:**

- **Another Slurm cluster:** change the `#SBATCH --account`, `--qos`, `--partition` and
  `--constraint` lines.
- **Without Slurm:** every job script is a plain sequence of `python -m research.longeval...`
  commands, which run unchanged on any Linux machine with enough memory.
- **Same CPU:** use the same CPU model for every index arm (Gold 6226 here) so timings are
  comparable.
- **Reproducibility:** each index is built single-threaded with a fixed seed. Rebuilding it on
  the same CPU type reproduced the exact graph and rankings, verified by state hashes. On another
  CPU type, expect small differences from floating-point and SIMD code paths. Compare those
  results statistically, not bit for bit.
