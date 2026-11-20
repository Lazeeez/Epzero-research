# Held-out extension: shuffled insertion order and ef sweep (exploratory)

This is post-freeze, exploratory work. It tests two findings from `HELDOUT_EXPLORATORY.md` on
the 381 test queries:

1. **Shuffled order.** Does rebuilding each snapshot in a random insertion order remove HNSW's
   excess misses of documents inserted later? Does it change the effectiveness gap?
2. **ef sweep.** How does the effectiveness gap depend on neighbour recall at each snapshot? The
   job searches at ef 100, 200, 400, 800, 1600 and 3200, for both the incremental and the
   shuffled indexes.

Each job handles one backend and one seed:

- The **incremental** arm must reproduce the analysed held-out states and rankings exactly, or
  the job stops.
- The **shuffled** arm then rebuilds each snapshot from scratch, in a random order seeded by the
  construction seed.

The searches are batched and not timed. Nothing here changes a primary result.

## Submit

This can run at the same time as the timing jobs (`le-hc-*`), which use the same CPU type. Copy
`build/longeval-heldout-ext-01.tar.gz` to the PACE LongEval checkout and compare its SHA-256 with
`build/longeval-heldout-ext-01-receipt.json`. Then:

```bash
sha256sum longeval-heldout-ext-01.tar.gz
tar -xzf longeval-heldout-ext-01.tar.gz
(cd longeval-heldout-ext-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  for backend in epzero faiss_hnsw; do
    sbatch --job-name="le-hx-${backend}-${seed}" \
      longeval-heldout-ext-01/scripts/pace/longeval-heldout-ext.sbatch "$backend" "$seed"
  done
done
```

These are six jobs. Each gets an exclusive Gold 6226 node, 64 GB RAM and 160 GB of node-local
disk, with a 12-hour limit. Expect about 4 hours: data preparation, then about 1.2 hours for the
incremental rebuild and about 2 hours for the shuffled one.

## After completion

```bash
sacct -X -u "$USER" -S 2026-11-20 --format=JobID,JobName%32,State,ExitCode,Elapsed | grep -E 'JobID|le-hx'
grep -h '^PASS\|^exit_code' longeval-heldout-ext-*.log
tar -czf longeval-heldout-ext-01-evidence.tar.gz build/pace-longeval-heldout-ext/jobs longeval-heldout-ext-*.log
sha256sum longeval-heldout-ext-01-evidence.tar.gz
```

The archive will be large, roughly 300–500 MB. Copy it to the local `build` directory and check
its SHA-256. It contains run files derived from LongEval data, so it must never be committed.
