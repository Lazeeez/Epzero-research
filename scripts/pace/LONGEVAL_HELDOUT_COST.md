# Held-out native HNSW timing

This job measures the secondary cost outcomes promised by FINAL_PROTOCOL_V1 (clarification 2):
native query latency and search work for the 381 test queries. It covers both HNSW
implementations at:

- fixed ef 200;
- the calibration-selected ef;
- a wide diagnostic ef of 3200.

Each job rebuilds the held-out incremental indexes for one seed. Every state and every ranking
must match the analysed held-out runs exactly, or the job stops. The timing protocol is the
development native-cost protocol, unchanged: one warmup and three timed single-query calls, one
thread, an exclusive Gold 6226 node. Epzero's distance counts come from untimed traces; FAISS
reports its native `ndis` and `nhops` counters. Nothing here can change a primary result.

## Submit

Copy `build/longeval-heldout-cost-01.tar.gz` to the PACE LongEval checkout. Compare its
SHA-256 with `build/longeval-heldout-cost-01-receipt.json`, then:

```bash
sha256sum longeval-heldout-cost-01.tar.gz
tar -xzf longeval-heldout-cost-01.tar.gz
(cd longeval-heldout-cost-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  sbatch --job-name="le-hc-${seed}" longeval-heldout-cost-01/scripts/pace/longeval-heldout-cost.sbatch "$seed"
done
```

These are three jobs, one per seed. Each runs Epzero and then FAISS, in separate processes; seed
29 runs FAISS first, as in development. Each gets an exclusive Gold 6226 node, 64 GB RAM and 160
GB of node-local disk, with an 8-hour limit. Expect about 3 hours.

## After completion

```bash
sacct -X -u "$USER" -S 2026-11-20 --format=JobID,JobName%32,State,ExitCode,Elapsed | grep -E 'JobID|le-hc'
grep -h '^PASS\|^exit_code' longeval-heldout-cost-*.log
tar -czf longeval-heldout-cost-01-evidence.tar.gz build/pace-longeval-heldout-cost/jobs longeval-heldout-cost-*.log
sha256sum longeval-heldout-cost-01-evidence.tar.gz
```

Copy the archive to the local `build` directory and check its SHA-256. It contains run files
derived from LongEval data, so it must never be committed.
