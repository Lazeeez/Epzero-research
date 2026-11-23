# Further construction seeds for the held-out primary arms

An exploratory follow-up to the held-out study, not part of `FINAL_PROTOCOL_V1`. Recorded on
23 November 2026, before any of these jobs was submitted.

## Why

The frozen analysis averaged three construction seeds (11, 29, 47) within each query and resampled
queries only. The sensitivity analysis (`research/longeval/HELDOUT_EXPLORATORY.md`, section 17)
found that Epzero's change in the gap comes mostly from one seed (−0.011 for seed 29, against −0.003
and −0.004), and that with seeds resampled as well its interval, [−0.0146, +0.0011], includes 0.
FAISS's calibration also depended on the seed: two of its three seeds moved to ef 400. Three seeds
cannot tell construction randomness from the effect.

## Design

Seven further seeds, fixed now: **101, 211, 307, 401, 503, 601, 701**. With the three frozen
seeds this gives ten per arm. Each job runs one seed through the frozen code, unchanged:

| Arm | Code path | Extra record |
| --- | --- | --- |
| Epzero HNSW, incremental | `trajectory.trajectory_core` with the held-out inputs and configuration | per-query recall of the 50 calibration queries at every width of the calibration grid, at every snapshot |
| FAISS HNSW, incremental | the same | the same |
| IVF, frozen training, incremental | `heldout.run_ivf` | — |

The observer only searches; it must leave the index fingerprint unchanged and reproduce the
calibration step's recall exactly, or the job stops. The identical-history rebuild is not repeated:
it was identical to incremental maintenance in all 27 comparisons.

## What will be estimated

Fixed now, before any output exists:

- **Primary.** For each arm, D (the change in the ANN-minus-exact nDCG@10 gap from snapshot 1 to 3,
  GPT-OSS-120B labels, the 374-query panel) over all ten seeds, with a two-level bootstrap
  (queries and seeds resampled), beside the frozen three-seed value.
- **Calibration.** For each arm and snapshot, how many of the ten seeds calibration moves beyond
  ef 200 (nprobe for IVF); D separately for seeds that widened and seeds that did not.
- **Calibration queries.** From the per-query calibration recall: its spread, the gap between
  calibration and test recall at the same width, and the calibration-size study of section 8 redone
  by resampling the calibration queries instead of the test queries.
- **Spread.** The standard deviation of D across seeds, per arm.

**Prediction**, from the three-seed result and the sensitivity analysis: FAISS and IVF stay within
±0.004 over ten seeds; Epzero's ten-seed D is smaller in magnitude than its three-seed −0.0059. A
result against either prediction is reported as such.

## Submit

Build the package locally (`python scripts/pace/package-longeval-heldout-seeds.py BUILD_DIR`), copy
`longeval-heldout-seeds-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with
`longeval-heldout-seeds-01-receipt.json`, then:

```bash
sha256sum longeval-heldout-seeds-01.tar.gz
tar -xzf longeval-heldout-seeds-01.tar.gz
(cd longeval-heldout-seeds-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 101 211 307 401 503 601 701; do
  sbatch longeval-heldout-seeds-01/scripts/pace/longeval-heldout-seeds.sbatch "$seed"
done
```

Seven jobs, one exclusive Gold 6226 node each (one thread, 64 GB, 160 GB node-local disk, 12-hour
limit). As with every CPU job of the study, keep all jobs on the same CPU type.

## After completion

```bash
sacct -X --name=longeval-seeds -S 2026-11-23 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-seeds-*.log
tar -czf longeval-heldout-seeds-01-evidence.tar.gz build/pace-longeval-heldout-seeds/jobs longeval-seeds-*.log
sha256sum longeval-heldout-seeds-01-evidence.tar.gz
```

The archive holds run files derived from LongEval data: copy it to the local `build` directory and
never commit it.
