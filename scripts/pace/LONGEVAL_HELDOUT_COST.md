# Held-out native HNSW timing

This job measures the secondary cost outcomes promised by FINAL_PROTOCOL_V1 (clarification 2):
native query latency and search work for the 381 test queries. It covers both HNSW
implementations at:

- fixed ef 200;
- the calibration-selected ef;
- a wide diagnostic ef of 3200;
- since package 02, fixed ef 100, 400, 800 and 1600 as well.

Package 01 ran as jobs 7178503–7178505 on 20 November 2026. It timed the first three settings
only, so the paper could not give the latency of ef 400 and ef 800, the widths at which the
effectiveness gap nearly vanishes. Package 02 reruns the same jobs with the four further widths.
The rebuild must again reproduce every analysed state and ranking. The timings at ef 200, the
calibrated ef and ef 3200 are measured again; they should come close to those of package 01, and
both are reported.

Each job rebuilds the held-out incremental indexes for one seed. Every state and every ranking
must match the analysed held-out runs exactly, or the job stops. The timing protocol is the
development native-cost protocol, unchanged: one warmup and three timed single-query calls, one
thread, an exclusive Gold 6226 node. Epzero's distance counts come from untimed traces; FAISS
reports its native `ndis` and `nhops` counters. Nothing here can change a primary result.

## Submit

Copy `longeval-heldout-cost-02.tar.gz` (built locally with
`python scripts/pace/package-longeval-heldout-cost.py BUILD_DIR`) to the PACE LongEval checkout. Compare its
SHA-256 with `build/longeval-heldout-cost-02-receipt.json`, then:

```bash
sha256sum longeval-heldout-cost-02.tar.gz
tar -xzf longeval-heldout-cost-02.tar.gz
(cd longeval-heldout-cost-02 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  sbatch --job-name="le-hc2-${seed}" longeval-heldout-cost-02/scripts/pace/longeval-heldout-cost.sbatch "$seed"
done
```

These are three jobs, one per seed. Each runs Epzero and then FAISS, in separate processes; seed
29 runs FAISS first, as in development. Each gets an exclusive Gold 6226 node, 64 GB RAM and 160
GB of node-local disk, with an 8-hour limit. Expect about 3 hours.

## After completion

```bash
sacct -X -u "$USER" -S 2026-11-22 --format=JobID,JobName%32,State,ExitCode,Elapsed | grep -E 'JobID|le-hc2'
jobs=$(sacct -X -n -u "$USER" -S 2026-11-22 --format=JobID,JobName%32 | awk '$2 ~ /^le-hc2-/ {print $1}')
for j in $jobs; do grep -H '^PASS\|^exit_code' "longeval-heldout-cost-$j.log"; done
tar -czf longeval-heldout-cost-02-evidence.tar.gz \
  $(for j in $jobs; do echo "build/pace-longeval-heldout-cost/jobs/$j longeval-heldout-cost-$j.log"; done)
sha256sum longeval-heldout-cost-02-evidence.tar.gz
```

Copy the archive to the local `build` directory and check its SHA-256. It contains run files
derived from LongEval data, so it must never be committed.
