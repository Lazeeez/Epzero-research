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
sacct -X --name=longeval-bge-seeds -S 2026-11-25 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-bge-seeds-*.log
tar -czf longeval-bge-seeds-01-evidence.tar.gz build/pace-longeval-bge-seeds/jobs longeval-bge-seeds-*.log
sha256sum longeval-bge-seeds-01-evidence.tar.gz
```

The archive holds run files derived from LongEval data: copy it to the local `build` directory and never
commit it.

## Submitted (26 November 2026)

The archive's SHA-256 on PACE matched the receipt (9601874904d87e62…6ae4ed) and the package checksums
verified. Jobs 7325332–7325338, seeds 101, 211, 307, 401, 503, 601, 701 in that order.
All seven were pending on resources at first check. PACE's clock was still on 25 November (UTC), so `sacct`
needs `-S 2026-11-25`; an evidence archive made before the jobs ran holds nothing and is deleted.

## Result (27 November 2026)

All seven jobs passed (1 h 22 min to 1 h 24 min each, 34 of 34 environment tests in every job). Evidence
archive SHA-256 ee5e5ad6…a6850cb54, checked locally, then unpacked and run with `bge_seeds_analysis.py`;
aggregates `research/longeval/e4-bge-seeds-ten.json`. All 22 BGE jobs (15 held-out, 7 seed) passed the content
check against the same prepared inputs, and no report was excluded. The three frozen seeds reproduce E4's D for
every arm. D is the change from snapshot 1 to 3 in the ANN-minus-exact nDCG@10 gap (GPT-OSS-120B labels,
374-query panel); intervals resample queries and seeds (10,000 draws).

| Arm | D, three frozen seeds | **D, ten seeds** | 95% interval (queries and seeds) | SD across seeds | Per seed, range |
| --- | --- | --- | --- | --- | --- |
| Epzero HNSW | +0.0016 | **+0.0020** | [−0.0034, +0.0085] | 0.0021 | −0.0020 to +0.0049 |
| FAISS HNSW | −0.0027 | **+0.0000** | [−0.0057, +0.0061] | 0.0022 | −0.0050 to +0.0019 |
| IVF, frozen training | +0.0005 | **+0.0002** | [−0.0015, +0.0018] | 0.0008 | −0.0012 to +0.0010 |

**Encoder contrast** (BGE minus Qwen, ten seeds each, per query with seeds averaged within each encoder; queries
resampled jointly and seeds within each encoder):

| Arm | BGE D | Qwen D | **BGE minus Qwen** | 95% interval |
| --- | --- | --- | --- | --- |
| Epzero HNSW | +0.0020 | −0.0034 | **+0.0054** | [−0.0030, +0.0165] |
| FAISS HNSW | +0.0000 | −0.0021 | **+0.0021** | [−0.0055, +0.0106] |
| IVF, frozen training | +0.0002 | −0.0009 | **+0.0011** | [−0.0020, +0.0043] |

**Predictions.** (1) Ten-seed BGE D within ±0.006 for every arm: **held** (+0.0020, +0.0000, +0.0002), and
every interval lies within ±0.01. (2) Epzero's encoder contrast smaller in magnitude than the three-seed
+0.0075: **held** (+0.0054). (3) Its interval within ±0.01: **failed**; the upper end is +0.0165. Whether
Epzero's arm agrees across encoders within the margin is still not settled. With seed variation included,
FAISS's contrast interval also reaches just past the margin (+0.0106); E4's three-seed contrasts resampled
queries only, so the two are not directly comparable.

**Calibration.** Under BGE, calibration chose ef 400 for both HNSW arms and nprobe 128 for IVF at all three
snapshots of all ten seeds: the width never changed as the collection grew. (Under Qwen, Epzero stayed at ef
200 on nine of ten seeds and FAISS moved from 200 to 400 on six.) The analysis's "widened" count, defined as
"beyond ef 200 at any snapshot", is therefore ten of ten for both HNSW arms, but none of them widened over
time.

**Reading** (after the result, not tested): under BGE the two HNSW arms ran at the same width throughout, and
their ten-seed D agrees (+0.0020 and +0.0000). Under Qwen, at a fixed ef 400, Epzero's frozen D was −0.0026
against FAISS's −0.0022. So Epzero's encoder contrast is partly a contrast between search widths (200 under
Qwen, 400 under BGE), not only between encoders.
