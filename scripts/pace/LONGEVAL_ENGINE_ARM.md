# The repaired-Epzero arm at full scale: design, predictions and analysis

An exploratory follow-up, not part of `FINAL_PROTOCOL_V1`. Recorded on 24 November 2026, before any of
these jobs was submitted. Decided by the author; the advisor was unavailable.

## Why

On the 300K development sample, two engine options together removed most of the arrival bias in
Epzero's default build path (screens D3–D10, `docs/screen-d*.md`): `extra_backlinks` 4 (each insertion is
also linked from the closest construction candidates it did not choose) and refining each later
snapshot's insertions (`refine_links(first=...)`). Over three seeds the newest cohort's excess fell
from 2.22 to 0.63 points (a rebuild gives 0.12). This run tests the same repair, chosen on development
data, at full scale and on queries never used to choose it.

## Design

All of snapshot 3 (1,661,900 documents, the released Qwen3-Embedding-4B vectors, inner product), as
the collection grew: the snapshot-1 documents are trained on and bulk-built with 8-bit codes on all
cores, then refined; the snapshot-2 and snapshot-3 documents are inserted one at a time, in arrival
order (`research/longeval/engine_arm.py`). Epzero is built on the node from the `min-backlinks`
branch at the commit recorded in the package manifest. Graph: M 16, efConstruction 100, `fast_build`
on, refine width 50. Three arms:

| Arm | `extra_backlinks` | Later snapshots refined | Bulk build |
| --- | --- | --- | --- |
| grown | 0 | no | snapshot 1 |
| repaired | 4 | yes, after each | snapshot 1 |
| rebuilt (reference) | 0 | — | all of snapshot 3 |

Seeds 11, 29 and 47 (one job per seed, all three arms). Searched at ef 100, 200 and 400, k 100,
re-ranking the beam with full vectors, on the 381 test queries (development: the repair was chosen on
them) and on the **100 confirmation queries** (the LongEval-Sci training queries; never used to choose
anything about the engine or any arrival-order result). Exact top-100 targets are computed in the job
and checked against the study's references, as in the order rerun.

## Analysis, fixed now

`research/longeval/engine_arm_analysis.py`, ef 200, seeds averaged within query, paired query
bootstrap (10,000 draws), with the margins of decision 3 (`docs/arrival-bias-research.md`):

- **Effect:** the newest cohort's within-query excess (snapshot 3 against 1) is lower in `repaired`
  than in `grown`, with the 95% interval of the difference excluding 0, and is at most 1.0 point in
  `repaired`.
- **No harm:** the upper 95% bound of the rise in the oldest cohort's miss rate is at most 0.75
  points, and of the fall in recall@100 at most 0.5 points.
- **Confirmation reading** (100 queries): decided by the effect criterion; no harm reported with its
  interval. **Development reading** (381 test queries): all three criteria.
- Reported beside them: `repaired` against `rebuilt`, ef 100 and 400, distance computations per query,
  build, insertion and refine times.

## Predictions

From screens D9 and D10: the repair passes on both readings; at ef 200 the newest cohort's excess in
`repaired` is less than half of `grown`'s; recall@100 is higher in `repaired` than in `grown`; the
refine of later snapshots takes less time on all cores than their single-threaded insertion. A result
against a prediction is reported as such. If the confirmation reading fails, the repair is not
claimed, and the paper reports it as a development result only.

## Submit

Build the package locally (`python scripts/pace/package-longeval-engine-arm.py BUILD_DIR EPZERO_SOURCE`),
copy `longeval-engine-arm-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with the
receipt, then:

```bash
sha256sum longeval-engine-arm-01.tar.gz
tar -xzf longeval-engine-arm-01.tar.gz
(cd longeval-engine-arm-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  sbatch longeval-engine-arm-01/scripts/pace/longeval-engine-arm.sbatch "$seed"
done
```

Three jobs, one exclusive Gold 6226 node each. Keep all jobs on the same CPU type.

## After completion

```bash
sacct -X --name=longeval-engine-arm -S 2026-11-24 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-engine-arm-*.log
tar -czf longeval-engine-arm-01-evidence.tar.gz build/pace-longeval-engine-arm/jobs longeval-engine-arm-*.log
sha256sum longeval-engine-arm-01-evidence.tar.gz
```

The archive holds run files derived from LongEval data: copy it to the local `build` directory and
never commit it. Locally, merge the three jobs' `builds` folders and run
`python -m research.longeval.engine_arm_analysis BUILDS OUT.json` and again with `--confirm`.

## Result (jobs 7254793–7254795, 24 November 2026)

All three jobs passed; every evidence file verified; the test and confirmation queries' exact
neighbours matched the study's references; the node-built Epzero carried both repair options
(`engine-check.json`). Aggregates: `research/longeval/engine-arm-01-confirm.json` and
`engine-arm-01-test.json`. Ef 200, seeds 11, 29, 47 averaged within query, 10,000 bootstrap draws.

| Reading | Arm | Newest-cohort excess | Oldest-cohort miss | Recall@100 |
| --- | --- | --- | --- | --- |
| Confirmation (100) | grown | 1.46 pp | 3.41% | 95.76% |
| | repaired | −0.32 pp | 3.69% | 95.84% |
| | rebuilt | −0.80 pp | 4.14% | 95.69% |
| Development (381) | grown | 3.41 pp | 4.85% | 93.61% |
| | repaired | 1.54 pp | 4.89% | 94.10% |
| | rebuilt | 0.43 pp | 5.19% | 94.19% |

Repaired minus grown, with 95% intervals:

- **Confirmation:** excess −1.78 [−3.06, −0.76]; oldest +0.28 [−0.13, +0.87]; recall +0.08 [−0.42, +0.45].
- **Development:** excess −1.87 [−2.41, −1.36]; oldest +0.04 [−0.18, +0.25]; recall +0.49 [+0.27, +0.75].

**Decision under the rules fixed beforehand:**

- **Confirmation reading: passed.** The excess falls with an interval excluding 0 and ends at −0.32
  points, under 1.0. No harm is reported, not gated: the oldest cohort's upper bound (+0.87) is above
  0.75 with 100 queries.
- **Development reading: failed** on the absolute criterion. The excess falls by 55% with an
  interval excluding 0, and there is no harm (oldest +0.04, recall +0.49), but the repaired excess
  (1.54 points) is above the 1.0-point target. At full scale the grown excess (3.41) is larger than
  on the 300K development sample (2.22).

**Predictions:** "passes on both readings": **failed** (the development reading). "Repaired excess
less than half of grown": held (0.45 of it on the development queries; below zero on the confirmation
queries). "Recall@100 higher in repaired": held on both. "Refine of later snapshots faster than their
insertion": held (30 s on all cores against 733 s single-threaded).

**Cost:** later insertions 702 → 733 s (+4%); refines 30 s; distance computations at ef 200 2,937 →
3,009 (+2.5%). The rebuild reaches a lower excess but raises the oldest cohort's miss rate (+0.34 on
the development queries, +0.73 on the confirmation queries) and costs a 182 s bulk build.

**Reading:** the repair reliably removes about half or more of the arrival bias at full scale and
improves recall, at small cost, and passes the confirmation reading. It does not yet bring the newest
cohort's excess under one point on all queries. Under decision 1 the paper may present it as the
remedy only after the remaining confirmation readings (BGE vectors and a public streaming workload);
a stronger setting (`extra_backlinks` 8 cut the excess further on the development sample, screen
D7b) is the next development step.

## Second run: 8 links (package `longeval-engine-arm-02`)

Recorded on 24 November 2026, before submission. Development screen D13
(`docs/screen-d13-stronger-repair.md`) found that `extra_backlinks` 8 with each later snapshot refined
cuts the 300K sample's excess from 2.22 to 0.28 points over three seeds (4 links: 0.63; rebuild: 0.12),
and its rule sent 8 links to a full-scale run.

**Design.** As the first run, two arms per job: `grown` (unchanged; a same-job reference) and
`repaired8` (`extra_backlinks` 8, each later snapshot refined), seeds 11, 29 and 47, the same Epzero
source branch (commit in the manifest). `rebuilt` is taken from the first run (no history, so not
affected by the change).

**Analysis, fixed now.** `engine_arm_analysis.py --arm repaired8` (and with `--confirm`), on the second
run's `grown` and `repaired8` folders plus the first run's `rebuilt` folders, with decision 3's
margins as before: effect (interval of repaired8 minus grown excluding 0, and repaired8's excess at most
1.0 point), no harm (oldest cohort's rise ≤ 0.75, recall@100's fall ≤ 0.5, upper 95% bounds). The
confirmation reading is decided by the effect. **Reproduction check:** the second run's `grown`
per-query tables are compared with the first run's; any difference is reported (the bulk build runs on
all cores), and the paired contrasts use the same-job `grown`.

**Predictions.** Passes both readings: on the test queries repaired8's excess is at most 1.0 point
(first run, 4 links: 1.54), with no harm; on the confirmation queries the effect holds. Search work at
ef 200 within 5% of the 4-link arm's; insertion time at most 1.1× grown's.

**Submit**, as the first run, with the new package:

```bash
sha256sum longeval-engine-arm-02.tar.gz
tar -xzf longeval-engine-arm-02.tar.gz
(cd longeval-engine-arm-02 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  sbatch longeval-engine-arm-02/scripts/pace/longeval-engine-arm.sbatch "$seed"
done
```

**After completion**, as the first run (`sacct -X --name=longeval-engine-arm -S 2026-11-23 ...`), into
`longeval-engine-arm-02-evidence.tar.gz` (never committed).

### Result of the second run (jobs 7265455–7265457, 24 November 2026)

All three jobs passed; every evidence file verified against its job's checksums; the manifests match
package -02 (engine commit `4d4a64cc`); the exact neighbours of both query sets matched the study's
references; tests passed. Aggregates: `research/longeval/engine-arm-02-test.json` and
`engine-arm-02-confirm.json` (second run's `grown` and `repaired8`, first run's `rebuilt`). Ef 200, seeds
11, 29, 47 averaged within query, 10,000 bootstrap draws.

| Reading | Arm | Newest-cohort excess | Oldest-cohort miss | Recall@100 |
| --- | --- | --- | --- | --- |
| Confirmation (100) | grown | 1.52 pp | 3.74% | 95.47% |
| | repaired8 | −0.36 pp | 3.72% | 95.82% |
| | rebuilt (first run) | −0.80 pp | 4.14% | 95.69% |
| Development (381) | grown | 3.43 pp | 4.77% | 93.69% |
| | repaired8 | 1.17 pp | 5.04% | 94.01% |
| | rebuilt (first run) | 0.43 pp | 5.19% | 94.19% |

repaired8 minus grown, with 95% intervals:

- **Confirmation:** excess −1.87 [−3.00, −0.86]; oldest −0.02 [−0.98, +0.70]; recall +0.35 [−0.30, +1.19].
- **Development:** excess −2.26 [−2.80, −1.75]; oldest +0.27 [+0.04, +0.48]; recall +0.32 [+0.10, +0.57].

**Decision under the rules fixed beforehand:** the confirmation reading **passes** (the effect; the
no-harm bounds, reported, are also inside the margins this time). The development reading **fails** on
the absolute criterion: the excess falls by 66% with an interval excluding 0 and no harm by the margins
(oldest upper bound +0.48 ≤ 0.75, recall up), but 1.17 points is above the 1.0 target.

**Predictions:** "passes both readings": **failed** (development, 1.17 > 1.0). "Search work at ef 200
within 5% of the 4-link arm's": held (3,016 against 3,009). "Insertion time at most 1.1× grown's": held
(755 against 710 s single-threaded, 1.06×; with the 30 s of refine on all cores, 1.11×).

**Against the first run's 4-link arm** (development): excess 1.54 → 1.17, oldest 4.89% → 5.04%, recall
94.10% → 94.01%. Doubling the links removes a further 0.37 points; on the 300K sample it removed 0.35
(0.63 → 0.28). At full scale the repair closes 75% of the gap between grown and rebuilt (3.43 → 1.17
against 0.43), with 8 links; 62% with 4.

**Reproduction check:** the second run's `grown` is not bit-identical to the first run's (the bulk build
runs on all cores). Averaged over seeds, the development excess is 3.43 against 3.41, the oldest-cohort
miss 4.77% against 4.85%, recall@100 93.69% against 93.61%; on the confirmation queries 1.52 against
1.46 points, with one seed (29) differing by about a point of recall on 100 queries. Build-to-build
variation of this size is why the paired contrasts use the same-job `grown`.

**Reading:** a stronger setting helps, but with diminishing return; the remaining excess at full scale
is not removed by more links alone. Bulk build and insertion leave the newest documents with a
disadvantage that only a rebuild (0.43, itself not zero) removes. Under decision 2 the test queries are a
development reading; the confirmation readings that decide the paper are the confirmation queries
(passed by both settings), the BGE vectors and a public streaming workload.

## Third run: a wider refine of each new batch (package `longeval-engine-arm-03`)

Recorded on 24 November 2026, before submission. Screen D14 (`docs/screen-d14-refine-around.md`)
found that refining the older nodes around each batch widens the gap. Screen D15
(`docs/screen-d15-refine-new-harder.md`) found that refining each later snapshot's insertions at width
100 instead of 50 takes the 300K sample's excess from 0.28 to −0.08 points over three seeds (a rebuild
gives 0.12). Recall@100 was 0.9519, against a rebuild's 0.9520, for 3% more search work and 15 s of
refine on all cores instead of 8. D15's rule sent it to a full-scale run.

**Design.** As the second run, two arms per job: `grown` (unchanged; a same-job reference) and
`repaired8w` (`extra_backlinks` 8, each later snapshot's insertions refined at width 100). The refine
after the bulk build stays at width 50 (`REFINE_EF`, what `epzero.build()` uses). Seeds 11, 29 and 47.
The Epzero source is the same branch, now at 2d279c0e; its only change since 4d4a64cc is the off-by-default
`refine_links(around)` of screen D14, which this run does not use. `rebuilt` is taken from the first run.

**Analysis, fixed now.** `engine_arm_analysis.py --arm repaired8w` (and with `--confirm`), on the third
run's `grown` and `repaired8w` folders plus the first run's `rebuilt` folders. The margins are decision
3's, as before:

- effect: the interval of repaired8w minus grown excludes 0, and repaired8w's excess is at most 1.0 point;
- no harm: the upper 95% bound of the oldest cohort's rise is at most 0.75, and of recall@100's fall at
  most 0.5.

The confirmation reading is decided by the effect. Reported beside them: repaired8w against the second
run's repaired8 (different jobs, so means only, not paired), and the reproduction check of `grown`
against the first two runs.

**Predictions.** Passes both readings: on the test queries repaired8w's excess is at most 1.0 point
(repaired8: 1.17), with no harm by the margins; on the confirmation queries the effect holds. The
oldest cohort's rise against grown on the test queries is at most the rebuild's (first run: +0.34)
plus 0.25 points. Search work at ef 200 within 5% of repaired8's (3,016). Refine of later snapshots
at most 2.5× the second run's 30 s on all cores. If the test-query excess is still above 1.0, a wider
refine is reported as closing part of the gap, and the paper presents the repair under decision 1 as
before.

**Submit**, as before, with the new package:

```bash
sha256sum longeval-engine-arm-03.tar.gz
tar -xzf longeval-engine-arm-03.tar.gz
(cd longeval-engine-arm-03 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  sbatch longeval-engine-arm-03/scripts/pace/longeval-engine-arm.sbatch "$seed"
done
```

**After completion**, as before (`sacct -X --name=longeval-engine-arm -S 2026-11-24 ...`), into
`longeval-engine-arm-03-evidence.tar.gz` (never committed).

### Result of the third run (jobs 7273883–7273885, 24 November 2026)

All three jobs passed, and every evidence file verified against its job's checksums. The manifests
match package -03 (engine commit `2d279c0e`). The prepared sample is byte-identical to the second
run's, and the exact neighbours of both query sets matched the study's references. All 21 tests passed
in each job. Evidence: `BUILD_DIR/longeval-engine-arm-03-evidence.tar.gz` (SHA-256 `35e2e007…f48f`),
never committed. Aggregates: `research/longeval/engine-arm-03-test.json` and
`engine-arm-03-confirm.json` (third run's `grown` and `repaired8w`, first run's `rebuilt`). Ef 200,
seeds 11, 29, 47 averaged within query, 10,000 bootstrap draws.

| Reading | Arm | Newest-cohort excess | Oldest-cohort miss | Recall@100 |
| --- | --- | --- | --- | --- |
| Confirmation (100) | grown | 1.12 pp | 3.93% | 95.29% |
| | **repaired8w** | **−0.37 pp** | 3.54% | 96.15% |
| | rebuilt (first run) | −0.80 pp | 4.14% | 95.69% |
| Development (381) | grown | 3.36 pp | 4.78% | 93.71% |
| | **repaired8w** | **0.57 pp** | 4.98% | 94.27% |
| | rebuilt (first run) | 0.43 pp | 5.19% | 94.19% |

repaired8w minus grown, with 95% intervals:

- **Development:** excess −2.79 [−3.41, −2.21]; oldest +0.20 [−0.02, +0.42]; recall +0.56 [+0.33, +0.81].
- **Confirmation:** excess −1.49 [−2.61, −0.39]; oldest −0.39 [−1.32, +0.22]; recall +0.86 [+0.27, +1.67].

repaired8w minus rebuilt: development excess +0.14 [−0.48, +0.79], oldest −0.20 [−0.49, +0.08], recall
+0.08 [−0.20, +0.38]; confirmation excess +0.43 [−0.64, +1.48], oldest −0.60 [−1.10, −0.11], recall
+0.45 [+0.02, +0.90].

**Decision under the rules fixed beforehand:** both readings **pass**. On the development queries the
excess falls by 83%, with an interval excluding 0, to 0.57 points, under the 1.0-point target. The
oldest cohort's upper bound (+0.42) is inside the 0.75 margin, and recall rises. On the confirmation
queries the effect holds, with an interval excluding 0.

**Predictions:** "passes both readings": **held**. "Oldest cohort's rise at most the rebuild's (+0.34)
plus 0.25": **held** (+0.20). "Search work at ef 200 within 5% of repaired8's (3,016)": **held** (3,092,
+2.5%). "Refine of later snapshots at most 2.5× 30 s": **held** (54 s on all cores, 1.8×).

**Against a rebuild:** the repaired index is not distinguishable from a full rebuild on any measure.
The excess differs by +0.14 points on the development queries (interval −0.48 to +0.79). The oldest
cohort's miss rate is no higher than the rebuild's (lower on the confirmation queries, with an interval
excluding 0), and recall is the same on the development queries and higher on the confirmation queries. This costs 1.09× the single-threaded insertion time (773 against
706 s) and 54 s of refine on all cores, against a 182 s rebuild on all cores at each checkpoint.

**Against the earlier repairs** (development, different jobs, means only): excess 1.54 (4 links) →
1.17 (8 links) → 0.57 (8 links, width-100 refine). The width-100 refine removes about half of what
8 links left, as screen D15 predicted in direction (on the 300K sample it went below zero).

**Reproduction check:** the third run's `grown` agrees with the first two (development excess 3.36
against 3.43 and 3.41, oldest miss 4.78% against 4.77% and 4.85%, recall 93.71% against 93.69% and
93.61%). The builds are not bit-identical, because the bulk build runs on all cores.

**Reading:** with `extra_backlinks` 8 and each batch of insertions refined at width 100, a grown
index's arrival bias at full scale is no larger than a rebuild's, on both query sets. The repair keeps
the oldest documents' miss rate no higher than a rebuild's and costs a fraction of one. Under decision 2 the
remaining confirmation readings are the BGE vectors and the public streaming workload; both are
running.
