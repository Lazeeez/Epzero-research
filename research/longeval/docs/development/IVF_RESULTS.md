# Verified IVF maintenance controls

PACE jobs **7133623 (seed 11)**, **7133624 (seed 29)** and **7133625 (seed 47)**
are complete and independently verified locally. Each ran frozen incremental, same-training
rebuild and refreshed rebuild over all three snapshots. All 27 snapshot conditions met
the calibration target. Of 297 calibration grid points, 291 were feasible and six one-probe
points returned incomplete rankings; they are retained as infeasible, not silently dropped. These are
development results, with no held-out evaluation or significance claims.

## Controls and calibration

All nine incremental/same-training rebuild snapshot pairs have identical serialized index
hashes, training identities, membership/order digests, list sizes and development rankings.
All three modes agree at snapshot 1. Rebuilding with the identical codebook and insertion
history provides no quality change in these conditions.

The smallest grid point reaching mean calibration neighbor Recall@100 of 0.95 is 128
probes except at snapshot 1 for seed 29: 128 reaches 0.9432, so 256 is selected. Seed 29
selects 128 at later snapshots. Every all-list query returns the exact top-100 neighbor
set; no numerical boundary substitutions were needed. All regimes are retained.

## Final snapshot at current calibration (128 probes)

| Seed | Frozen incremental recall | Refreshed rebuild recall | Difference (pp) | Incremental maintenance (s) | Retrain + rebuild maintenance (s) |
|---:|---:|---:|---:|---:|---:|
| 11 | 95.88% | 95.64% | -0.24 | 19.48 | 156.67 |
| 29 | 95.46% | 95.34% | -0.12 | 18.78 | 154.36 |
| 47 | 96.54% | 95.78% | -0.76 | 19.42 | 156.61 |

Retraining also lowers development mean neighbor recall at snapshot 2 under current
calibration: -0.72, -0.02 and -0.82 pp for seeds 11, 29 and 47. This does not establish
that retraining is generally harmful: it changes the sample and centroids, and outcomes
depend on the search regime.

**At seed 29's original 256-probe setting**, final refreshed recall is **98.30%** versus
frozen-training **97.96%**, an increase of 0.34 pp. Do not select only the less favorable
retraining comparison. Mean-target success also does not establish a per-query floor:
at current calibration final minima are 0.77/0.78/0.75 with frozen training and
0.79/0.76/0.75 after retraining. There are no zero-recall cases in these final regimes.

## Costs and relevance

Current-calibrated final mean scanned-vector counts are approximately 225,760–229,300
with frozen training and 226,851–234,171 after retraining. Every query also compares with
1,024 coarse centroids. Equal nprobe does not mean equal work.

Mean per-query median native latency is 185.58–188.47 ms for frozen training and
186.41–192.46 ms after retraining. The protocol is one query per call, one warmup and
three timed repeats including Python/native overhead. Earlier instrumented HNSW batch
timings are not comparable; the new native-cost package addresses that gap.

Maintenance sums training, allocation and build/delta insertion, including sampling,
gathering and adapter overhead, excluding audits/fingerprints and query evaluation.
Whole-worker peak RSS is 22.36–22.66 GiB, not isolated native index memory. Whole-process
times are 34m13s–39m34s including verification and all search grids. Fixed mode order
and separate allocations limit small latency-effect claims.

All 81 IVF development run conditions were scored against DCTR and raw training labels,
with independent evaluator agreement to 1e-12. **All labels belong to snapshot 1.** Later
scores are historical-label diagnostics, not contemporaneous temporal effectiveness.

| Seed | Final DCTR nDCG@10, frozen training | Refreshed training (current calibration) |
|---:|---:|---:|
| 11 | 0.259583 | 0.255010 |
| 29 | 0.259583 | 0.254094 |
| 47 | 0.257841 | 0.248451 |

Relevant-document Recall@100 increases for seeds 11 and 29 and decreases for 47. Exact
nDCG@10 against these historical labels is 0.266470. Both relevance metrics and both
probe regimes remain available; neighbor recall alone does not establish relevance quality.

## Verification

- Three successful jobs, each with 16 passing native tests. Returned manifests match the
  81-file submitted supplement; source/runtime/input identities match prior evidence.
- All local vector files rehashed; full audit chains match. Large PACE paths are
  compute-node `/tmp/longeval-JOB_ID.*` directories.
- All 378 exported run conditions (18,900 query conditions) recounted, including calibration
  selection, work counters and repeat metadata.
- Nine distinct seed/snapshot training samples independently selected by a full priority
  sort. The 589,824 selected row occurrences match training-vector byte hashes. Frozen
  modes' reused training samples and centroid identities agree.
- **416,691 scored query/document checks across arms/snapshots**, using independent
  float64 einsum scores. Maximum absolute discrepancy: **1.613e-7**. This count is not
  globally unique or a count of independent observations.
- **18,900 coarse-probe checks** against centroid geometry; all-list neighbor sets match
  exact references with zero boundary substitutions.
- Six corruption checks reject omitted better centroids, reversed probe order, duplicate
  and invalid probes, wrong scores and nonfinite scores.

Full native indexes were not exported/rebuilt locally, and centroids were not retrained
locally. State equality uses returned serialization hashes, corroborated by control
rankings and source identity. Local checks independently validate returned document
membership, training samples, centroid artifacts, geometry, scores and metrics.

Evidence: `build/longeval-pace-ivf-evidence-JOB_ID/local-verification.json`.
Analysis: `build/longeval-ivf-comparison/` with 81 cohort rows, 297 calibration rows,
4,050 development query rows, 27 maintenance rows, 168 relevance summaries (including
six exact references), and 8,100 relevance query rows.
Scripts: `build/receive-longeval-ivf.py`, `build/verify-longeval-ivf.py`,
`build/longeval_ivf_local_checks.py`, `build/check-longeval-ivf-verifier.py`,
`build/summarize-longeval-ivf.py`. The vector verifier uses the existing WSL environment.

Archive SHA-256:

- 7133623: `ed6a8a443b9606ed1da217ae531d2457b0333be2cc3493763b21dc44f50b3dd3`
- 7133624: `2f4028b551c985e2326da121e0c62ffb05125ecf7f4f03189423c37e3b5612cf`
- 7133625: `c8b86ee6ccf5b1039d39fb7873751891962aea83851698bd2358be6a6c35ad42`
