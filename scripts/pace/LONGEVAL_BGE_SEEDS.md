# Further construction seeds for E4's held-out arms on the BGE vectors

An exploratory follow-up to E4, under amendment 7 of `EXTENSION_AMENDMENTS.md`. Recorded on 26 November
2026, before any of these jobs was built or submitted.

## Why

With the frozen three seeds, Epzero's D was −0.0059 under Qwen and +0.0016 under BGE, and their paired
difference, +0.0075 [−0.0017, +0.0205], is not within the ±0.01 margin: whether Epzero's arm agrees across
encoders is the one E4 question left open. Under Qwen, ten seeds (`LONGEVAL_HELDOUT_SEEDS.md`) moved
Epzero's D to −0.0034 [−0.0092, +0.0015], showing that seed 29 had pulled the three-seed value down. The
same seven further seeds on the BGE vectors give both encoders ten seeds.

## Design

Seeds **101, 211, 307, 401, 503, 601, 701** (those of the Qwen follow-up). One job per seed runs the three
primary arms in turn through `bge_heldout.py`'s entry points, unchanged, with the frozen configurations:
Epzero HNSW and FAISS HNSW maintained incrementally (`hnsw --mode incremental`) and IVF with frozen
training (`ivf --mode frozen_incremental`). Each job prepares the BGE inputs from the verified vectors (job
7290861) as the 15 held-out jobs did; amendment 6's content check applies to its inputs. The rebuild arms
are not repeated: rebuilds were identical to incremental maintenance in all 27 BGE comparisons.

## What will be estimated (fixed now)

- **Primary.** Per arm, D over the ten BGE seeds with a two-level bootstrap (queries and seeds resampled,
  10,000 draws), as `seeds_analysis.py` does for Qwen, beside the three-seed value.
- **Encoder contrast.** Per arm, the paired difference of ten-seed D, BGE minus Qwen, per query (seeds
  averaged within each encoder), with a two-level bootstrap resampling queries and, within each encoder,
  seeds.
- **Calibration.** How many of the ten seeds calibration moved beyond ef 200 (nprobe for IVF).

**Predictions:** the ten-seed BGE D is within ±0.006 for every arm (point estimates); Epzero's encoder
contrast is smaller in magnitude than the three-seed +0.0075, and its interval lies within ±0.01. A result
against a prediction is reported as such.

## Submit

Build the package locally (`python scripts/pace/package-longeval-bge-seeds.py BUILD_DIR`), copy
`longeval-bge-seeds-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with
`longeval-bge-seeds-01-receipt.json`, then:

```bash
sha256sum longeval-bge-seeds-01.tar.gz
tar -xzf longeval-bge-seeds-01.tar.gz
(cd longeval-bge-seeds-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 101 211 307 401 503 601 701; do
  sbatch longeval-bge-seeds-01/scripts/pace/longeval-bge-seeds.sbatch "$seed"
done
```

Seven jobs, one exclusive Gold 6226 node each (one thread, 64 GB, 60 GB node-local disk, 12-hour limit;
each arm took 19–42 minutes in the held-out jobs). As with every CPU job of the study, keep all jobs on the
same CPU type and do not overlap them with jobs on other CPU types.

## After completion

```bash
sacct -X --name=longeval-bge-seeds -S 2026-11-26 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-bge-seeds-*.log
tar -czf longeval-bge-seeds-01-evidence.tar.gz build/pace-longeval-bge-seeds/jobs longeval-bge-seeds-*.log
sha256sum longeval-bge-seeds-01-evidence.tar.gz
```

The archive holds run files derived from LongEval data: copy it to the local `build` directory and never
commit it.
