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

## Part B: the engine repair (Epzero's default build path)

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
checked against its SHA-256 (`b65641b2…`).

*Changed on 24 November 2026, after screen D15 (`docs/screen-d15-refine-new-harder.md`):* Part B gains
a fifth arm, `repaired8w` (8 links, each later cohort's insertions refined at width 100), which is
the third full-scale engine arm's candidate. It is analysed as the other repairs, with the same effect
and no-harm rules. Package `bigann-stream-02` holds it. Part A and the other arms are unchanged.
If package -01's jobs were already submitted, they stand, and only `repaired8w` is run from -02
(`ENGINE_ARMS=repaired8w`). It is then compared with -01's `grown` from a different job, and that
contrast is reported as across jobs.

## Submit

Build the package locally (`python scripts/pace/package-bigann-stream.py BUILD_DIR ORYX_DIR`), copy
`bigann-stream-02.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with the receipt, then:

```bash
tar -xzf bigann-stream-02.tar.gz
(cd bigann-stream-02 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  for backend in epzero faiss_hnsw; do sbatch bigann-stream-02/scripts/pace/bigann-stream.sbatch order "$backend" "$seed"; done
  sbatch bigann-stream-02/scripts/pace/bigann-stream.sbatch engine "$seed"
done
```

If -01's nine jobs are already queued or done, submit only the new arm instead:

```bash
for seed in 11 29 47; do
  ENGINE_ARMS=repaired8w sbatch bigann-stream-02/scripts/pace/bigann-stream.sbatch engine "$seed"
done
```

## After completion

```bash
sacct -X --name=bigann-stream -S 2026-11-23 --format=JobID,State,ExitCode,Elapsed,Constraints
grep -H '^PASS\|^exit_code' bigann-stream-*.log
tar -czf bigann-stream-02-evidence.tar.gz build/pace-bigann-stream/jobs bigann-stream-*.log
sha256sum bigann-stream-02-evidence.tar.gz
```

Locally: merge the jobs' `builds/epzero`, `builds/faiss_hnsw` and `builds/engine` folders, prepare the
same sample locally or reuse one job's `sample.json` hashes for the check, and run
`python -m research.longeval.bigann_stream analyze SAMPLE epzero=... faiss_hnsw=... --engine ... --output OUT.json`.

## Result (jobs 7269863–7269871 and 7279048–7279050, 24 November 2026)

All twelve jobs passed, and every evidence file verified against its job's checksums. The nine -01
jobs ran package -01 (engine commit `4d4a64cc`); the three `repaired8w` jobs ran -02 (`2d279c0e`).
All twelve downloaded byte-identical files (sizes, MD5s and the pinned runbook's SHA-256 checked) and
computed byte-identical `sample.json`. Their exact neighbours matched the track's ground truth (9,998 of
10,000 queries identical as sets, mean overlap 0.999998). The sample was prepared again locally from
the same files, and its `sample.json` is byte-identical to the jobs'. Evidence:
`Oryx/build/bigann-stream-evidence.tar.gz` (SHA-256 `d27efac3…a3d`). Aggregates:
`research/longeval/bigann-stream-analysis.json`. Ef 200, k 100, 10,000 queries, seeds 11, 29, 47
averaged within query, 5,000 bootstrap draws.

**Part A: insertion order**

| Measure (percentage points) | Epzero | FAISS HNSW |
| --- | --- | --- |
| (1) Gap change, appended → reversed | **−11.22** [−11.52, −10.91] | **−9.08** [−9.38, −8.79] |
| Cohort risk change, appended → reversed: 1 / 2 / 3 | +6.68 / −1.01 / −4.54 | +5.40 / −0.88 / −3.68 |
| (2) Within-block, last minus first tenth, appended | +1.09 [+0.77, +1.41] | +0.67 [+0.35, +0.97] |
| (2) Within-block, reversed | +0.86 [+0.54, +1.16] | +0.25 [−0.06, +0.56] |
| (3) Global position, last minus first tenth, shuffled | +7.71 [+7.49, +7.92] | +6.01 [+5.80, +6.21] |

**Predictions (timing-minority reading):**

- (3) "positive with an interval excluding 0 for both": **held**, and large (+6.0 to +7.7 points).
- (2) "positive with an interval excluding 0 in appended builds for both": **held**, but small (+0.7 to
  +1.1 points).
- (1) "|gap change| < 2.0 points for both": **failed**. It is −11.2 and −9.1 points, more than twice
  LongEval's −4.04 for Epzero. Under the rule fixed above, position alone explains the bias, and **the
  timing-minority reading is not supported**. Arriving together with its own neighbours does not protect
  a cluster: whichever cohort comes last is missed more, and the oldest cohort loses 5–7 points when it
  is inserted last instead of first.

**Part B: the engine repair** (repaired8w against the -01 jobs' grown, from different jobs on different
nodes; the effect is far larger than build-to-build variation)

| Arm | Newest-cohort excess | Oldest-cohort miss | Recall@100 |
| --- | --- | --- | --- |
| grown | 5.63 pp | 20.99% | 74.74% |
| repaired (4 links + refine new) | 1.31 pp | 19.79% | 79.29% |
| repaired8 | 0.50 pp | 19.90% | 79.58% |
| **repaired8w (8 links + width-100 refine)** | **−0.65 pp** | 19.63% | **80.88%** |
| rebuilt | 2.24 pp | 19.10% | 79.25% |

Effect (paired gap change, grown → repair, decision 3): repaired −3.19 [−3.43, −2.95], repaired8 −3.77
[−4.01, −3.51], **repaired8w −4.95 [−5.21, −4.68]**. All three pass the effect, and both no-harm margins:
the oldest cohort's miss rate falls in every arm (repaired8w −1.36 [−1.59, −1.14]), and recall rises
(repaired8w +6.14 [+5.97, +6.31] points). Against the rebuild, repaired8w has a lower excess (−2.88
[−3.07, −2.70]) and higher recall (+1.63 [+1.52, +1.73]), with the oldest cohort 0.53 points worse.

**Predictions:** "both repairs raise recall@100": **held** (+4.6 to +6.1 points). "Cut the within-block
position difference": **failed** (grown +1.19; repaired +1.60, repaired8 +1.20, repaired8w +0.95, with
overlapping intervals). "repaired8 meets the no-harm margins": **held**. Part A's primary contrast is far
from zero, so the reading is informative.

**Reading:** on the community's own streaming benchmark, both implementations show the arrival bias
by global insertion position, more strongly than on LongEval. The rebuild is not free of it either
(2.24 points): its bulk build inserts in file order, which is arrival order here. The width-100 repair
removes it, lowers the oldest cohort's misses, and lifts recall@100 at ef 200 from 74.7% to 80.9%,
above a rebuild. This is the public-workload confirmation reading of decision 2, and it **passes**.
