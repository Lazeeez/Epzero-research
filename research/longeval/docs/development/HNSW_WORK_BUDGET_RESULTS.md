# Verified full-cohort search-work results

PACE jobs **7130605 (seed 11)**, **7130606 (seed 29)** and **7130607 (seed 47)** are
complete and independently verified. All 100 training queries were evaluated at three
snapshots, with the 50 calibration and 50 development queries kept separate. This report
supersedes the pending-execution status in [HNSW_WORK_BUDGET_PLAN.md](HNSW_WORK_BUDGET_PLAN.md).
The submitted protocol, engine and original results are unchanged.

**All nine calibration conditions select 4,000 distance evaluations. At that budget, no
calibration or development query has zero recall in any tested seed/snapshot condition.**
The seed-11 final-snapshot development mean is 96.18%, compared with 96.68% for each
other seed. Individual-query shortfalls remain substantial, so the mean target does not
establish a per-query recall floor.

## Fixed work budget across snapshots

The smallest feasible declared grid point reaching mean calibration recall of 0.95 is
4,000 at every snapshot and seed. Thus the snapshot-1 frozen budget and the independently
selected current-snapshot budget coincide throughout this experiment. All grid points were
feasible; no queries were dropped. Calibration mean recall at the selected budget ranges
from 96.26% to 96.56% across the nine conditions.

Development ANN recall@100 at 4,000 distance evaluations:

| Seed | Snapshot 1 | Snapshot 2 | Snapshot 3 | Snapshot 3 minus snapshot 1 |
|---:|---:|---:|---:|---:|
| 11 | 97.10% | 97.30% | 96.18% | -0.92 pp |
| 29 | 97.18% | 97.20% | 96.68% | -0.50 pp |
| 47 | 97.08% | 97.16% | 96.68% | -0.40 pp |

All these query conditions use exactly 4,000 evaluations; none has exhausted its full native
score stream by that point. The unchanged selection is a statement about this coarse grid
and mean target. It does not prove that the optimal budget is constant or that recall is
unchanged as the corpus grows. The table shows residual development differences.

## Original frozen effort versus the work-budget condition

Final-snapshot development results:

| Seed | Original ef=200 mean recall | Original mean distance evaluations | Budget=4,000 mean recall | Minimum recall at 4,000 | Queries below 95% at 4,000 |
|---:|---:|---:|---:|---:|---:|
| 11 | 92.00% | 2,749.92 | 96.18% | 64% | 10 / 50 |
| 29 | 95.82% | 2,794.00 | 96.68% | 66% | 8 / 50 |
| 47 | 95.88% | 2,773.50 | 96.68% | 69% | 9 / 50 |

The final-snapshot spread between seed means decreases from 3.88 percentage points in the
original ef=200 condition to 0.50 points at the shared work budget. This is not an equal-work
comparison between those two settings: the 4,000-evaluation condition uses more work on
average and follows an ef=3200 search trajectory. It also has a different beam capacity.
It must not be presented as a free accuracy gain or a measured latency improvement.

The two previously failing seed-11 queries are:

| Query | Original ef=200 recall | Original distance evaluations | Recall at 4,000 evaluations |
|---|---:|---:|---:|
| `161820e44e7022cf44fd99a59bce5792` | 0% | 3,213 | 85% |
| `21e8c323e168b0813eff14fdac972f1f` | 0% | 615 | 87% |

Both cease being total misses at the selected work budget; neither reaches 95% there.
This is distinct from the earlier selected-candidate experiment's 99% recovery endpoints,
which used 5,869 and 21,474 distance evaluations. Preserve those effort distinctions.

## Full grid and interpretation limits

Every calibration and development result is retained at budgets 1,000, 2,000, 4,000,
8,000, 16,000, 32,000 and 64,000, alongside native ef=200 and ef=3200 outcomes. At larger
budgets some native searches exhaust their score streams, so actual distance counts can be
below the nominal budget. For example, final development means at the 64,000 ceiling are
99.60%, 99.56% and 99.54%, while actual mean distance counts are approximately 31,149,
31,067 and 31,029. These are not searches that necessarily execute 64,000 evaluations.

The budgeted rankings are **offline prefixes of native ef=3200 traces**. They include
upper-layer and entry distance evaluations. They do not measure a production native
capped-search API, prefix expansion counts, or capped-query latency. The native event format
does not expose expansions that discover no new neighbors. At an exactly equal distance
ceiling, full score-stream coverage also need not mean every final expansion has executed.

These results concern Epzero incremental construction. They do not provide a new matched-work
FAISS/IVF comparison, establish general index aging, or demonstrate a new algorithm. Queries
repeat across snapshots and seeds; the experiment contains 100 distinct queries, not 900
independent query samples. No confidence interval, significance, failure-rate estimate or
held-out claim is assigned to this development analysis.

## Verification

All three jobs passed 26 fixture/regression tests and reproduced the original snapshot states
and six baseline run identities per job. Local verification checked the 72-file submitted
supplement, baseline/source/binary/package/input identities, corpus audits, ID joins, insertion
orders, query separation, ranked outputs, recall values and calibration selections.

- **1,800 complete native traces** checked, totaling **30,126,016 events**, with no truncation.
- **1,800 native query rankings and 6,300 budgeted query rankings** checked across 162
  exported cohort run files.
- Prefix rankings independently reconstructed by stable sorting of all scored candidates at
  each checkpoint, rather than using the experiment's heap implementation. Native beam
  admission flags were independently checked against the retained-distance bound.
- **27,604,498 distinct scored query/document checks across jobs and snapshots**, recomputed
  against audited vectors using float64 accumulation. Repeated pairs across jobs/snapshots
  are counted separately; this is not a global unique-pair or independent-sample count.
- Maximum absolute distance discrepancy: approximately **1.06e-7**.
- **48 native score streams** match the earlier seed-11/29 diagnostic hashes at shared
  efforts and snapshots.
- Corrupted distance counts, admission flags and repeated layer-0 discoveries are rejected
  by verifier checks; early cutoffs and stable ties are also checked.

Explicit stopped-traversal parity checks ran on PACE for the four declared diagnostic queries
at each snapshot and seed, at all feasible budgets. Local checks verified their returned
records and independently reconstructed every query's budgeted ranking. Full graph adjacency
and invisible expansion events were not exported, so the entire native index and those
explicit traversals were not reconstructed locally. Post-observation graph equality is the
remote fingerprint assertion, supported by baseline reproduction and source identity checks.

Peak diagnostic-worker RSS was 17.2619–17.2631 GiB. Diagnostic process wall times were
55m52.94s, 54m44.00s and 55m26.95s for seeds 11, 29 and 47. These include graph reconstruction
and instrumentation; they are not isolated query latency. Preparation records place the large
data and work files in the compute nodes' `/tmp/longeval-JOB_ID.*` directories.

## Artifacts and next decision

`build/longeval-work-budget-comparison/` contains:

- `comparison.json`: evidence links, verification totals, complete cohort grids and summary.
- `cohort-grids.csv`: all 162 native/budget cohort conditions.
- `selected-budgets.csv`: frozen/current calibration selections and both cohorts' outcomes.
- `per-query.csv`: all 8,100 native/budget query conditions, including zero-recall cases.
- `resources.csv`, `verifier-checks.json`, and `analysis-manifest.json`.

Comparison SHA-256:
`c774317c180d7d605c1658a8dc61165534a2a38deec6a9f5d7a4fb3579556196`.
Per-job receipts are in
`build/longeval-pace-work-budget-evidence-JOB_ID/local-verification.json` and bind the
received archives, returned files and verification code. Scripts are
`build/verify-longeval-work-budgets.py`, `build/longeval_work_budget_verification.py`, and
`build/summarize-longeval-work-budgets.py`.

The full-cohort HNSW work diagnostic is complete. Next, define and implement the remaining
IVF controls: bounded training samples, frozen-centroid incremental updates, same-training
rebuild reproduction, and fresh-current-snapshot training, with explicit seeds and resource
accounting. These require their own validated experiment; they are not already completed by
this result. Representation freeze, practical-effect margins, inference, held-out evaluation,
and a defensible novelty assessment remain necessary for a paper. No further PACE job was
prepared or submitted during this analysis.
