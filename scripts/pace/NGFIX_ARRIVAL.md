# Screen A2b on PACE: NGFix against the arrival bias

Design and predictions: `docs/screen-a2b-ngfix.md` (recorded before any code). NGFix's kernels need AVX-512, so it
runs on a Gold 6226 node; the laptop cannot run it.

## Package (locally)

```bash
python scripts/pace/package-ngfix-arrival.py ~/epzero-longeval/pilot BUILD_DIR
```

BUILD_DIR must hold the verified `longeval-ngfix-screen-02.tar.gz`; its pinned NGFix source and Boost headers are
reused unchanged, and the packager refuses an adapter that differs from that package's. The archive holds the
LongEval sample's vectors (derived from LongEval data): copy it only to the PACE LongEval checkout, never commit it.

## Submit (from the PACE LongEval checkout)

```bash
sha256sum ngfix-arrival-01.tar.gz        # compare with ngfix-arrival-01-receipt.json
tar -xzf ngfix-arrival-01.tar.gz
(cd ngfix-arrival-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch ngfix-arrival-01/scripts/pace/ngfix-arrival.sbatch
```

One exclusive Gold 6226 node, one thread, 64 GB, 12-hour limit. The job compiles the adapter with GCC 13.3.0 and
`-march=skylake-avx512`, runs this screen's tests, then four runs (GloVe and LongEval, as arrived and shuffled),
each prepared, built, repaired, searched and scored on the node.

## After completion

```bash
sacct -X --name=ngfix-arrival -S 2026-11-29 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' ngfix-arrival-*.log
tar -czf ngfix-arrival-01-evidence.tar.gz build/pace-ngfix-arrival/jobs ngfix-arrival-*.log
sha256sum ngfix-arrival-01-evidence.tar.gz
```

The evidence holds scored aggregates, timings and resource reports only (no vectors or rankings).
