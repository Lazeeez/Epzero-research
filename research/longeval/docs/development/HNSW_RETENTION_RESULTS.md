# Verified candidate-retention results

PACE jobs **7128488 (seed 11)** and **7128489 (seed 29)** completed, and both returned
bundles passed local verification. All 144 predeclared search conditions were checked.
This report supersedes the pending-results status in
[HNSW_RETENTION_EXPERIMENT.md](HNSW_RETENTION_EXPERIMENT.md); the submitted protocol and
source package remain unchanged.

**The selected deferred expansions recover the two targeted failures, but provide no
recall advantage over wider ordinary search at the measured distance-evaluation budgets.**
All 48 intervention/control pairs produce exactly the same ranked document IDs and scores,
not just the same recall. This is evidence about the mechanism in selected development
cases, not a demonstrated general search improvement or a publication-ready novelty claim.

## Targeted recoveries

Q1 is `161820e44e7022cf44fd99a59bce5792`; Q2 is
`21e8c323e168b0813eff14fdac972f1f`. Both rows below are seed 11, snapshot 3. Recall is
exact-top-100 overlap, not judgment-based relevance recall.

| Query | Base ef | Deferred document | Base recall | Rescue recall | Wider-search control recall | Distance evaluations: base / rescue / control |
|---|---:|---|---:|---:|---:|---|
| Q1 | 200 | `85864826` | 0% | 99% | 99% | 3,213 / 5,869 / 5,869 |
| Q2 | 1600 | `154424178` | 0% | 99% | 99% | 2,100 / 21,474 / 21,474 |

The intervention changes only how one admitted-but-evicted, unexpanded candidate is handled:
after ordinary search exhausts, expand that candidate once and continue normal processing.
It does not change graph edges, the upper-layer route, or force-admit newly encountered nodes.
The wide-beam control uses ef=3200 and stops at the intervention's distance-evaluation count.

These are not both fixes at ef=200. **Q2's selected candidate does not trigger at ef=200**,
where recall remains zero. Its recovery uses an ef=1600 baseline followed by the deferred
expansion. Increasing the ordinary beam by one slot, to 201 or 1601 respectively, also
leaves the two failures at zero recall.

Saving one candidate does not mean doing only one extra expansion. The new neighbors start
further search. Q1's expansion count rises from 207 to 420; Q2's from 1,626 to 3,231. The
paired wide searches use 421 and 3,433 expansions respectively. The interventions therefore
use fewer expansions and smaller configured beams in these comparisons, but they do not
reduce distance evaluations or improve ranked output at the matched endpoints. No isolated
native latency or total search-memory advantage was measured.

## All controls retained

The experiment contains four selected development queries, two seeds, six ordinary effort
settings and two intervention effort settings. Each intervention setting independently tests
both selected documents and one hash-selected unrelated evicted candidate. Every intervention
has its own distance-capped wide-beam control.

| Check across both jobs | Result |
|---|---:|
| Total search conditions | 144 |
| Intervention/control pairs | 48 |
| Exact ranked-output equality with paired control | 48 / 48 |
| Equal actual distance counts with paired control | 48 / 48 |
| Selected-document interventions triggered | 2 / 32 |
| Selected-document interventions inactive | 30 / 32 |
| Unrelated-candidate interventions triggered | 16 / 16 |
| Unrelated-candidate interventions improving recall | 0 / 16 |
| Interventions improving recall over their lower-effort baseline | 2 / 48 |

The selected documents do not trigger rescue in seed 29 or in the two other development
queries. Inactive interventions reproduce their baseline rankings and work counts exactly.
The unrelated-candidate interventions add work without changing recall in the tested cases.
These counts describe this deliberately selected diagnostic matrix; they are not estimates
of how often failures or useful candidates occur in a population.

## Independent verification

Both PACE runs passed 19 fixture/regression tests and reproduced the three original
incremental snapshot states and all six baseline ranking-file identities per job. Local
verification checked the submitted 64-file supplement, original source/input identities,
the Epzero binary and package versions, query separation, corpus audits, insertion-order
digests, baseline recall and scores.

A separate verifier uses a heap frontier and bounded result set, rather than the experiment's
cursor implementation. It replays every saved expansion, ordered neighbor list, score,
admission, eviction, deferred expansion and stopping decision, and reproduces the output and
work counts for all 144 conditions. It also checks that shared graph-neighbor lists agree
across conditions and that the unrelated-candidate selections follow the declared hash rule.

- **2,119,434 complete log events** verified, with no truncated traces.
- **264,236 distinct query/document distance checks across the two jobs**, independently
  recomputed from the audited vectors using float64 accumulation. Pairs shared across jobs
  are counted once per job, not as globally distinct samples.
- Maximum absolute score/distance discrepancy in the retention arms: approximately **8.64e-8**.
- **32 complete native score-event streams** match the earlier diagnostic stream hashes at
  shared effort settings, including streams whose earlier saved logs were truncated.
- All 144 top-100 rankings and recalls independently checked; graph-state equality after
  observation is recorded by the remote fingerprint check.

The verifier reconstructs the explored portions of the graph from the exported adjacency
lists, not the entire native index. Native ordinary-search parity was checked on PACE and
is additionally supported by the prior stream identities; the native graphs were not copied
and rebuilt locally. Equal distance counts do not imply equal expansions, memory or latency.

Observed peak process RSS was 17.2644 GiB for seed 11 and 17.2600 GiB for seed 29. Diagnostic
process wall times were 51m29.55s and 51m33.35s. These cover reconstruction and instrumentation,
not isolated query performance. The returned preparation records locate large data/work files
under each compute node's `/tmp/longeval-JOB_ID.*` directory.

## Evidence locations

- `build/longeval-pace-retention-evidence-7128488/local-verification.json`
- `build/longeval-pace-retention-evidence-7128489/local-verification.json`
- `build/longeval-retention-comparison/comparison.json`: evidence links, all pairs, findings
  and interpretation limits.
- `build/longeval-retention-comparison/all-arms.csv`: all 144 search conditions.
- `build/longeval-retention-comparison/paired-controls.csv`: all 48 matched comparisons.
- `build/longeval-retention-comparison/resources.csv`: process memory by job and seed.

Comparison SHA-256:
`61e42a9eff2200621853dbfab885cc1db3f76597e1be07aaad7daf5b1cbed79c`.
Received archive SHA-256 values:

- Seed 11: `4c132d1284a0f63ac7ff9c7b70f26399e2066597ffb3d0ab81fb290192870583`.
- Seed 29: `e0918f277ed4202d05a4e9fe9bc01d7d50eea37f7a06144e9e6acf79c2d078cf`.

Verification scripts: `build/verify-longeval-retentions.py` and
`build/longeval_retention_verification.py`. Aggregation:
`build/summarize-longeval-retentions.py`. Receipts bind their source hashes and every returned
file. Verification requires the existing audited local corpus and query/reference artifacts.

## Research decision

Preserve this as a mechanism result with a negative result on recall advantage at the tested
work budgets. A deferred expansion is sufficient to recover each selected failure at its
tested lower effort, but neither necessity nor a unique bottleneck is established. No native
algorithm patch or new primary operating point is justified by these results.

Do not extend hand-selected candidate sweeps simply to find a favorable outcome. The next
useful work is broader coverage under declared search-work budgets, together with the
remaining IVF and resource controls. Any general retention policy needs a candidate-selection
rule that does not use these post-hoc document identities, separate development validation,
and eventually frozen held-out evaluation. The wider-search comparator must remain in that
evaluation. Representation freeze, practical-effect thresholds, inference, and related-work
assessment remain necessary before a defensible paper.
