# Verified HNSW development matrix: seeds 11, 29 and 47

All 18 trajectories are complete and locally verified: two backends, three construction modes
and three seeds, each evaluated at three full snapshots. This gives 54 snapshot conditions and
108 exported development runs across frozen and calibration-selected search effort. This report
supersedes the pending-replication status in HNSW_SEED11_RESULTS.md.

The same 50 development queries and 50 separate calibration queries were used throughout.
Documents, query precision, exact references, source bundle and numeric configuration remained
fixed. No final-test queries or judgments were used. These are descriptive development results,
not confirmatory evidence or a claim of publication readiness.

## Main findings

Replay equals incremental construction for every backend, seed and snapshot: all 18 paired
snapshot comparisons have matching recorded index fingerprints, chronological order digests
and both ranking-file hashes. Epzero uses topology fingerprints; FAISS uses canonical serialized
index fingerprints with efSearch set to 200. Within these tested conditions, keeping an index
alive across snapshot boundaries has no additional observed effect beyond its construction
history, since fresh replay of that history reproduces the same result.

The two zero-recall queries occur only at snapshot 3 for Epzero seed 11 incremental and its
identical replay. They do not occur in the other seeds, shuffled Epzero, or any FAISS run.
Repeated replay of the same graph is a correctness control, not independent statistical
replication of this failure. All seeds, including the seed-11 result, remain in the analysis.

At fixed ef=200, shuffled construction improves mean final-snapshot recall only for Epzero
seed 11. It slightly reduces the mean for the other five backend/seed combinations. The current
evidence therefore does not support a general benefit from rebuilding with shuffled order or
a general additional degradation from retaining a live index. Subsequent diagnostics recover
both seed-11 failures on the unchanged graph by increasing search effort and identify candidate
evictions along successful paths. The construction-level cause remains unresolved; see
[HNSW_FAILURE_DIAGNOSTIC_RESULTS.md](HNSW_FAILURE_DIAGNOSTIC_RESULTS.md).

## Final snapshot: mean development ANN recall@100

All rows use ef=200. Replay is omitted as a separate numeric column because its full rankings
and fingerprints equal incremental construction in every row.

| Backend | Seed | Incremental / replay | Shuffled | Shuffled minus incremental |
|---|---:|---:|---:|---:|
| Epzero | 11 | 92.00% | 95.48% | +3.48 pp |
| Epzero | 29 | 95.82% | 95.70% | -0.12 pp |
| Epzero | 47 | 95.88% | 95.54% | -0.34 pp |
| FAISS HNSW | 11 | 95.60% | 95.30% | -0.30 pp |
| FAISS HNSW | 29 | 95.54% | 95.42% | -0.12 pp |
| FAISS HNSW | 47 | 95.66% | 95.38% | -0.28 pp |

Arithmetic means across all three seeds are 94.5667% versus 95.5733% for Epzero incremental
and shuffled, and 95.6000% versus 95.3667% for FAISS. The positive Epzero average order
difference is driven by seed 11; it should not be presented without the per-seed table.
Three runs of the same 50 queries are not 150 independent queries. No confirmatory p-value,
confidence interval or estimated failure probability is assigned from this development matrix.
In the shuffled condition, changing the seed changes both graph randomness and permutation;
the two sources of randomness have not been separately varied in a factorial design.

## Calibration and resources

All 54 snapshot conditions report meeting the 0.95 calibration target at the selected grid
point. This does not guarantee development recall of 0.95. The search grid is discrete, and
the selected effort can be 200 or 400 with different target overshoot. Selected-effort
comparisons must not be described as equal-cost or exactly equal-recall comparisons.

Observed peak worker RSS across all nine jobs for each backend was 17.2496–17.4003 GiB for
Epzero and 20.8396–21.2186 GiB for FAISS. The 64 GB allocation was sufficient in these runs.
Memory includes harness state and input metadata. Different native capacity/growth policies,
physical nodes and instrumentation limit performance comparisons. The unusually long total
wall time of an individual job cannot be assigned to index construction without its component
ledger; downloads and preparation are included in scheduler wall time.

## Verification and artifacts

The 12 newly returned bundles passed source/input identity checks, their nine native tests,
independent development recall recounts and direct score recomputation for all six runs each.
Verification also checked the pinned corpus audit receipts, ID membership, expected insertion
sequence (including seeded permutations), state record digests and incremental parent chains.
The same checks were completed earlier for the six seed-11 bundles.

These receipts do not imply independent reconstruction of the full native graph. Only its
remote fingerprint was retained. Calibration rankings were not exported, so calibration
selection is checked against the reported grid rather than independently recounted. The
harness records the Epzero binary hash and FAISS package version, but no FAISS binary hash.

Artifacts:

- `build/longeval-hnsw-three-seeds/comparison.json`: all 18 evidence chains, 54 results,
  replay comparisons, per-query order differences, zero-recall cases and seed summaries.
- `build/longeval-hnsw-three-seeds/snapshot-results.csv`: all 54 snapshot rows.
- `build/longeval-hnsw-three-seeds/seed-summary.csv`: descriptive mean, SD and range over
  the three seeds, preserving the outlier.
- `build/longeval-hnsw-three-seeds/resources.csv`: all 18 process-memory and construction
  measurements, bound to job IDs and conditions.
- `build/longeval-seeds29-47-received.json`: identities and hashes of the 12 received archives.
- `build/longeval-seeds29-47-verification-status.json`: completed verification status for all 12.
- `build/longeval-pace-trajectory-evidence-{JOB_ID}/local-verification.json`: detailed per-job
  checks and a hash ledger for every returned file.

Comparison SHA-256: `134a8b2c45e125614bb5a923a6466599001e8b900209d6dbe0e9ecb203d1f831`.
The aggregation script is `build/compare-longeval-hnsw-three-seeds.py`. The new verification
implementation is `build/verify-longeval-seed-replications.py`, orchestrated by
`build/run-longeval-replication-verification.py`. Each receipt binds its verifier hash.

## Implication for the research

The broad HNSW development matrix and the focused post-hoc diagnostic are complete.
PACE jobs 7126117 and 7126118 reproduced the original seed-11 and seed-29 incremental
graphs and rankings; both returned bundles passed independent local verification. At
snapshot 3, seed-11 Q1 recovers from zero to 100% recall at ef=400 and Q2 from zero to
99% at ef=3200, without changing the graph or upper route. Candidate-retention replay
identifies evictions on witnessed successful paths. This motivates a controlled retention
intervention with complete expansion records, rather than a graph-disconnection claim.
See [HNSW_FAILURE_DIAGNOSTIC_RESULTS.md](HNSW_FAILURE_DIAGNOSTIC_RESULTS.md) for evidence,
trace truncation limits and next steps. The original matrix and operating points remain intact.

The paper still needs a defensible contribution and mechanism, completed remaining IVF and
resource controls, a frozen primary representation and inferential plan with practical-effect
margins, held-out test evaluation and related-work/novelty assessment. The current controls
provide evidence against interpreting the seed-11 collapse as a general ANN aging result.
