# Held-out ANN arms (FINAL_PROTOCOL_V1)

This runs the frozen protocol's approximate-search arms on the 381 test queries:

- Epzero HNSW and FAISS HNSW, each kept incrementally and rebuilt from the identical history;
- FAISS IVF-Flat, with training frozen and with training refreshed;
- three construction seeds per arm.

The package is self-contained. It contains the frozen research code, the protocol and its
freeze receipt, the test-query vectors from job 7164678, and the exact references computed
after the freeze. It contains no relevance judgments. Judgments are read only by the local
analysis after the evidence is returned.

## Submit

Copy `build/longeval-heldout-01.tar.gz` to the PACE LongEval checkout. Compare its SHA-256 with
`build/longeval-heldout-01-receipt.json`, then run:

```bash
sha256sum longeval-heldout-01.tar.gz
tar -xzf longeval-heldout-01.tar.gz
(cd longeval-heldout-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  for backend in epzero faiss_hnsw; do
    for mode in incremental replay; do
      sbatch --job-name="le-ho-${backend}-${mode}-${seed}" \
        longeval-heldout-01/scripts/pace/longeval-heldout.sbatch hnsw "$backend" "$mode" "$seed"
    done
  done
  sbatch --job-name="le-ho-ivf-${seed}" longeval-heldout-01/scripts/pace/longeval-heldout.sbatch ivf "$seed"
done
```

This submits 15 jobs: 12 HNSW and 3 IVF. Each IVF job runs its three modes in sequence. Every job
gets an exclusive Gold 6226 node with one computation thread, 64 GB RAM, 160 GB of node-local
temporary disk and a 12-hour limit. All jobs use the same CPU type, so they may run in parallel.

Expected durations, from the development runs:

- HNSW incremental: about 1.5 hours;
- HNSW identical-history rebuild: about 2.5 hours;
- IVF (all three modes): about 3 hours.

Each job first verifies the package checksums, then runs the unit tests and the input preflight.
Only after those pass does it download and audit the dense data. The downloads stay on the
node's temporary disk.

## After completion

```bash
sacct -X -u "$USER" -S 2026-11-19 --format=JobID,JobName%32,State,ExitCode,Elapsed | grep -E 'JobID|le-ho'
grep -h '^PASS\|^exit_code' longeval-heldout-*.log
tar -czf longeval-heldout-01-evidence.tar.gz build/pace-longeval-heldout/jobs longeval-heldout-*.log
sha256sum longeval-heldout-01-evidence.tar.gz
```

The evidence archive is about 150 MB. Copy it to the local `build` directory and check its
SHA-256 after copying, because an earlier large transfer was truncated. It holds run files
derived from LongEval data, so it must never be committed to the public repository.

If a job fails, resubmit only that arm with the same command. Its new output goes to a new
job folder. Report both attempts: the analysis takes exactly one completed job per condition.
