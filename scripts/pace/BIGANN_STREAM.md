# A public streaming workload: Big-ANN's clustered MSTuring-10M runbook

Recorded on 24 November 2026, before any code for it was written or any data downloaded. Exploratory
follow-up, not part of `FINAL_PROTOCOL_V1`; decided by the author (the advisor was unavailable).

## Why

Decision 2 of `docs/arrival-bias-research.md` names a public streaming workload as one of the two
confirmation readings for any engine repair, beside the BGE vectors. It is also the claim's test on
the community's own streaming benchmark. The NeurIPS'23 Big-ANN streaming track ranks indexes by
recall averaged over a runbook's search steps; nobody reports it by when the target arrived.

## Workload

`msturing-10M-clustered` from `big-ann-benchmarks` (`neurips23/runbooks/clustered_runbook.yaml`):
10,000,000 MSTuring vectors (100 dimensions, float32, Euclidean), already permuted so that each of 32
k-means clusters (`clustered_data_gen.py`: `kmeans2`, `minit="++"`) is one contiguous block; the
runbook inserts the 32 blocks in file order (32 contiguous ranges from 0 to 10,000,000, a search after
each, no deletions). Files, downloaded on the compute node from the track's public storage and hashed
into the evidence: `msturing-10M-clustered.fbin`, `testQuery10K.fbin` (10,000 queries),
`clu_msturing10M_gt100` (the track's exact top-100 for the full set), and the runbook YAML (read as
data only; no code from the repository is run). Never used before in this project.

**Arrival cohorts:** a cluster's cohort is set by where its block starts: 1 if it starts before
3,333,334, 2 before 6,666,667, else 3 (thirds aligned to cluster boundaries, as LongEval's snapshots
are arrival blocks). **Exact targets:** computed in the job (exact float32 top-200, re-scored in float64,
ties by position) and checked against the track's file: mean top-100 overlap at least 0.999 and every
query at least 0.98, or the job stops.

## Part A: insertion order (both implementations)

As the LongEval order study (`research/longeval/insertion_pilot.py run`, unchanged): Epzero (no
compression) and FAISS HNSW, M 16, efConstruction 100, single-threaded, one vector at a time; orders
`appended` (the runbook's order), `reversed` (cohort 3's blocks, then 2's, then 1's; file order within)
and `shuffled` (one random permutation); seeds 11, 29, 47; searched at ef 100, 200, 400 and 800 (k 100)
with all 10,000 queries.

In this stream a vector's exact neighbours mostly arrive in the same block as it does, unlike LongEval,
where new documents land among old ones. Two readings of the mechanism (section 7.3 of the research
document) then predict different things:

- **Position alone** (passive accumulation by global insertion position): the newest cohort loses more
  neighbours when inserted last, as on LongEval and the random-cohort controls.
- **Timing minority** (a vector is disadvantaged when its neighbours arrived at a different time):
  little difference between cohorts, because each cluster arrives together; the disadvantage appears
  instead **within each block**, between a block's early and late members.

**Measures (fixed now; ef 200, seeds averaged within query, query bootstrap, 5,000 draws):**

1. **Primary, cohort level:** the paired change in the gap between the newest and oldest cohorts'
   miss risk from `appended` to `reversed` (`order_pairs.py`; LongEval: −4.04 points for Epzero).
2. **Within-block position:** in `appended` builds, the within-query miss-rate difference between
   targets inserted in the last tenth of their own cluster's block and those in its first tenth
   (strata query × exact-rank band), and the same in `reversed`.
3. **Global position:** in `shuffled` builds, the within-query difference between the last and the first
   insertion tenth (position independent of content; SIFT, GloVe and LongEval all showed it).

**Predictions (the author's, from the timing-minority reading):** (3) is positive with an interval
excluding 0 for both implementations (position matters at 10M on public data). (2) is positive with an
interval excluding 0 in `appended` builds for both. (1) is smaller in magnitude than half of LongEval's
(|gap change| < 2.0 points) for both, and may include 0. If instead (1) is at least as large as on
LongEval, position alone explains the bias and the timing-minority reading is not supported.

## Part B: the engine repair (Epzero's product path)

As the full-scale engine arm (`research/longeval/engine_arm.py`, unchanged; M 16, efConstruction 100,
8-bit codes, `fast_build`): cohort 1 bulk-built on all cores and refined, cohorts 2 and 3 inserted one
at a time in the runbook's order. Arms `grown`, `repaired` (`extra_backlinks` 4, each later cohort
refined), `repaired8` (8 links) and `rebuilt`, seeds 11, 29, 47, ef 100, 200, 400 (k 100).

**Analysis (fixed now), each repair against `grown`, ef 200:**

- **Effect (decides the confirmation reading, decision 3):** the paired change in the gap between the
  newest and oldest cohorts' miss risk from `grown` to the repair (`order_pairs.py` on the arms' returned
  IDs) is negative with a 95% interval excluding 0.
- **No harm (reported with intervals; decision 3's margins):** upper 95% bound of the rise in the oldest
  cohort's miss rate at most 0.75 points, and of the fall in recall@100 at most 0.5 points
  (`engine_arm_analysis.py`, which also reports the within-query excess as a secondary measure).
- Also: the within-block position difference (measure 2) in each arm; distance computations; build,
  insertion and refine times.

**Predictions:** both repairs raise recall@100 and cut the within-block position difference of the
inserted cohorts; if Part A's primary contrast is near zero, the cohort-gap effect may be too small to
detect, and the reading is then reported as uninformative for the cohort claim rather than as a pass or
a failure. `repaired8` meets the no-harm margins.

## Jobs

One package (`bigann-stream-01`, built locally, nothing derived from LongEval in it), run from the PACE
LongEval checkout on exclusive Gold 6226 nodes only (`scripts/pace/bigann-stream.sbatch`): Part A one job
per implementation and seed (three orders each, as in the LongEval order rerun), Part B one job per seed
(four arms), nine jobs in all. Every job downloads and prepares the stream itself; the analysis checks
that all jobs saw files with the same hashes and computed identical cohorts and exact targets.

*Changed on 24 November 2026, before any job was submitted:* Part A runs in the study's own Python
environment (its Epzero release and FAISS, exactly as the LongEval order study); only Part B builds
Epzero on the node from the `min-backlinks` branch (commit in the manifest), as in the engine arm. The
download is checked against the sizes and MD5 checksums the track's storage reports for the data,
queries and ground truth, and the runbook is fetched at commit `ec58203` of `big-ann-benchmarks` and
checked against its SHA-256 (`b65641b2�`).

## Submit

Build the package locally (`python scripts/pace/package-bigann-stream.py BUILD_DIR ORYX_DIR`), copy
`bigann-stream-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with the receipt, then:

```bash
tar -xzf bigann-stream-01.tar.gz
(cd bigann-stream-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  for backend in epzero faiss_hnsw; do sbatch bigann-stream-01/scripts/pace/bigann-stream.sbatch order "$backend" "$seed"; done
  sbatch bigann-stream-01/scripts/pace/bigann-stream.sbatch engine "$seed"
done
```

## After completion

```bash
sacct -X --name=bigann-stream -S 2026-11-23 --format=JobID,State,ExitCode,Elapsed,Constraints
grep -H '^PASS\|^exit_code' bigann-stream-*.log
tar -czf bigann-stream-01-evidence.tar.gz build/pace-bigann-stream/jobs bigann-stream-*.log
sha256sum bigann-stream-01-evidence.tar.gz
```

Locally: merge the jobs' `builds/epzero`, `builds/faiss_hnsw` and `builds/engine` folders, prepare the
same sample locally or reuse one job's `sample.json` hashes for the check, and run
`python -m research.longeval.bigann_stream analyze SAMPLE epzero=... faiss_hnsw=... --engine ... --output OUT.json`.
