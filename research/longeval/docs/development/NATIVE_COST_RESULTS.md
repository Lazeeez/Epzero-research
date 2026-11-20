# Verified full-data native HNSW costs — 2026-11-18

PACE jobs **7139467 / 7139468 / 7139469** (seeds 11 / 29 / 47) are independently
verified. Each ran Epzero and FAISS HNSW across all three snapshots. All 18 recorded
index states and original frozen/calibrated rankings reproduce the earlier baselines.
The evidence comprises **54 cohort conditions, 2,700 query conditions and 8,100 timed
calls**. No held-out query retrieval was performed.

Timings below are the mean across 50 development queries of each query's median of
three native calls, in milliseconds. Each condition has one immediate warmup. Both
backends used single-thread settings on exclusive Gold6226 allocations; the launch
script requests core/NUMA binding. Binding overhead is included; tracing and validation
are outside timing. This is warm search latency, not service throughput.

## Final snapshot: original fixed ef=200

| Seed | Epzero Recall@100 | Epzero ms | FAISS Recall@100 | FAISS ms | Epzero / FAISS time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 92.00% | 2.670 | 95.60% | 2.463 | 1.084 |
| 29 | 95.82% | 2.726 | 95.54% | 2.468 | 1.105 |
| 47 | 95.88% | 2.686 | 95.66% | 2.520 | 1.066 |

Epzero is 6.6–10.5% slower in these observed means. Seeds29/47 exchange modest
recall differences for latency; this table is not an exactly matched-recall claim.
Seed11 reproduces the two zero-recall failures and misses the 0.95 development mean,
despite meeting its calibration target. Neither timing nor lower whole-worker RSS
removes that quality failure.

## Current calibration: retain the selected settings and achieved recall

| Seed | Epzero ef / recall / ms | FAISS ef / recall / ms |
| --- | --- | --- |
| 11 | 200 / 92.00% / 2.686 | 200 / 95.60% / 2.466 |
| 29 | 200 / 95.82% / 2.726 | 400 / 97.44% / 4.670 |
| 47 | 200 / 95.88% / 2.710 | 400 / 97.70% / 4.700 |

For FAISS seeds29/47, calibration recall at ef200 is 94.36%/94.52%, so the frozen
selection rule chooses ef400. The higher cost accompanies 1.62/1.82 percentage points
more development recall than Epzero. Calling Epzero approximately 1.7 times faster
at equal recall would misrepresent this comparison. The raw regime name
`matched_recall` means calibration-selected effort, not equality of achieved recall.
Duplicate ef200 conditions remain separate timing observations with identical ranks.

## Wide diagnostic search: ef=3200

| Seed | Epzero recall / ms | FAISS recall / ms | Epzero / FAISS time |
| --- | --- | --- | ---: |
| 11 | 99.60% / 33.243 | 99.62% / 31.980 | 1.039 |
| 29 | 99.56% / 33.185 | 99.58% / 32.107 | 1.034 |
| 47 | 99.54% / 33.132 | 99.58% / 32.054 | 1.034 |

FAISS has lower mean time with similar mean recall here. Epzero p95 latency is
40.55–40.84ms versus FAISS37.69–38.01ms. Query-tail recall is mixed: Epzero minima are
96%/95%/95%, FAISS94%/96%/94%. No backend dominates every query-quality measure.
This wide setting is a diagnostic, not a calibrated minimum-cost setting for 99%
recall. Full native recall–latency curves remain necessary for a method-paper comparison.

## Construction, memory and IVF context

Snapshot1 allocation plus construction is 1,426–1,435s for Epzero and 1,411–1,418s
for FAISS. Final delta insertion is 580–581s and 583–585s respectively. Small
differences are descriptive, not a general build-speed claim. Whole-worker peak RSS
is 17.25–17.26GiB for Epzero and 20.85–20.86GiB for FAISS. These include dataset and
harness memory; they are not isolated index-memory measurements. Entire workers take
approximately 52 and 56 minutes, including audits and all timed/untimed query stages.

The earlier IVF-Flat current-calibrated final means are 185.58–188.47ms with frozen
training and 186.41–192.46ms after retraining. The CPU model and native-call protocol
now align, although allocations and achieved recall differ. Both measured HNSW
implementations are substantially faster at their selected settings in this workload.
This does not establish an Epzero-specific contribution over FAISS HNSW. IVF's final
incremental insertion takes approximately19s, versus approximately581–585s for HNSW;
query and maintenance costs must both be retained. All cross-backend context rows
include both achieved recalls rather than presenting unmatched time ratios as speedups.

## Verification and limits

- Each job passed ten native/trajectory tests; seven additional local verifier tests
  reject malformed cohorts, duplicate documents, incorrect recalls, incomplete timing
  repeats, mixed counter conventions and nonfinite scores.
- Returned 99-file supplement manifests match the submitted v02 package. Original
  source identities, Epzero binary hash, package versions, query contracts, exact
  references and full-data audit chains match earlier verified evidence. FAISS's
  package version is pinned; a separate FAISS shared-object hash is not in this protocol.
- Local full vector files were rehashed. Independent float64 scoring checked
  **47,116 distinct query/document pairs within job/snapshot groups**, covering all
  **270,000 returned occurrences**. Maximum absolute score difference: **1.613e-7**.
  These are verification counts, not independent statistical observations.
- Epzero ranks and distance/expansion counts match the previously audited complete
  traces at both ef200 and ef3200. FAISS counters are kept under their own names.
- Native index files were not returned or rebuilt locally; recorded state equality
  is corroborated by ranks, source identity, and independent score checks. Latency
  observations were recounted, not independently remeasured.
- Three allocations, sequential backend order (reversed for seed29), fixed query
  order, repeated queries and only50 unique development queries limit inference.
  Actual affinity masks were not logged separately. No significance claim is made.

Evidence: `build/longeval-pace-native-cost-evidence-JOB/local-verification.json`.
Tables: `build/longeval-native-cost-comparison/` contains all54 cohort rows,
2,700 per-query rows,27 paired comparisons,18 maintenance rows and IVF context.
Scripts: `build/receive-longeval-native-cost.py`, `build/verify-longeval-native-cost.py`,
`build/summarize-longeval-native-cost.py`; reusable checks: `native_cost_verification.py`.

The next research requirement is still a distinct repair/search mechanism with native
gains against strong competitors, including NGFix/RFix. These results complete a
baseline measurement requirement; they do not justify freezing the final held-out
protocol before the new-method gates are met.
