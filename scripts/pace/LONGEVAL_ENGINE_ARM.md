# The repaired-Epzero arm at full scale: design, predictions and analysis

An exploratory follow-up, not part of `FINAL_PROTOCOL_V1`. Recorded on 24 November 2026, before any of
these jobs was submitted. Decided by the author; the advisor was unavailable.

## Why

On the 300K development sample, two engine changes together removed most of the arrival bias in
Epzero's product path (screens D3–D10, `docs/screen-d*.md`): `extra_backlinks` 4 (each insertion is
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

Build the package locally (`python scripts/pace/package-longeval-engine-arm.py BUILD_DIR ORYX_DIR`),
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
