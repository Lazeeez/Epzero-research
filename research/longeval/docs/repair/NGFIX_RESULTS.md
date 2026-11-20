# NGFix/RFix native integration results — job 7141656

The returned bundle is independently verified. This establishes a working direct
competitor on the exposed 8,192-vector sample. It does **not** establish a new
method, a full-data reproduction of the paper, or a competitive Epzero advantage.

NGFix improves development recall modestly at each tested search setting. Its
much larger fitted-query gain makes selecting search effort on those same queries
optimistic. The next performance protocol must separate fitting, tuning and
evaluation. Fifty fitting queries and one sample cannot establish a limitation of
properly trained NGFix.

## Verification and measurement boundary

Returned archive: `build/longeval-ngfix-7141656-evidence.tar.gz`, SHA-256
`18a12cc7f7f4eccddb350b0f0b7d173385ef31d978b318aa8b040bee0230d747`.
The 21 original returned files are preserved under
`build/longeval-pace-ngfix-evidence-7141656/`. The returned package manifest matches
the submitted v03 archive byte for byte; all 500 unpacked package files verify.
The upstream commit remains `9acd987cf1c420069f8fd602c0d4e4aa81053558`.

Independent local checks cover:

- 1,600 real-data conditions / 4,800 timed calls and 256 synthetic conditions /
  768 timed calls; every condition returns 100 ordered, unique, valid IDs.
- All 185,600 returned score occurrences recomputed against float64 dot products
  (not 185,600 distinct vector pairs). Maximum absolute score discrepancy is
  1.014e-7 for real data and 7.076e-8 for the fixture.
- Exact references, input hashes, snapshot ingestion identity, all reported
  recalls, repeated counters, timings and selected settings recomputed.
- Eleven PACE Python tests and three native malformed-input rejection checks pass.
  Synthetic ef 800 recovers every exact top-100 neighbor for every query/control.
- Fixture, sample and rejection checks record the same executable SHA-256:
  `59cf56197f6c1ae72ce6a822116d49c6a983eae09a43a408904d34500ed982de`.
  The executable and graph states stayed on node-local disk; their bytes were
  **not** returned. Query-time state equality was checked on PACE and is supported
  locally by the returned hashes/sizes, not independently rehashed graph files.

The synthetic fixture first regenerated on Windows failed its input-report hash
check because JSON CRLF line endings enter the artifact hashes. The binary
matrices are identical. Regeneration in the existing Linux environment matches
the returned input contract and passes the independent verification. No verifier
tolerance or expected hash was relaxed.

GCC 13.3.0, AVX-512 full-precision inner product, Xeon Gold 6226, and process affinity
`[0]` are recorded. Each query/control has one warmup and three timed native C++
calls; tables show the mean of the per-query median times. Variant order rotates
by query. These timings exclude Python bindings and cannot directly rank this
implementation against earlier Epzero/FAISS/hnswlib binding timings. This is one
allocation and one graph build, without uncertainty estimates across builds.

## Development results at fixed search effort

All controls use the same sample, query vectors, insertion order and original
bottom-layer graph. Fifty queries fit repairs; fifty disjoint, previously exposed
development queries measure transfer. M=16, efConstruction=100, MEX=48, k=100.
Equal ef is not equal actual distance work or equal recall.

| ef | Base recall | NGFix recall | Gain (percentage points) | Base time (ms) | NGFix time (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 93.42% | 93.88% | +0.46 | 1.099 | 1.115 |
| 200 | 97.54% | 97.80% | +0.26 | 1.908 | 1.922 |
| 400 | 99.42% | 99.48% | +0.06 | 3.130 | 3.163 |
| 800 | 99.82% | 99.86% | +0.04 | 4.731 | 4.761 |

At ef 100, 12 development queries improve and 38 tie; none loses recall. NGFix adds
23 recovered neighbor occurrences out of 5,000. It improves 8/3/2 queries at
ef 200/400/800 respectively, with no recall losses. Minimum development recall at
ef 100 remains 78%; queries below 95% recall decrease from 28 to 24. At ef 400, the
minimum increases from 97% to 98%.

NGFix uses slightly more work: mean upstream distance counts at ef 100 are
1,321.2 versus 1,311.8, and at ef 400 are 3,351 versus 3,334.22. These counters omit
initial entry scoring. The observed mean times are 0.6–1.4% higher at common ef;
these small differences are descriptive, not a claim of statistically reliable
overhead or of an inferior quality–latency curve. A denser, independently tuned
curve is needed for an equal-quality comparison.

RFix alone produces the same saved state hash as base and identical rankings,
scores and counters in all 400 query/ef conditions. NGFix+RFix likewise matches
NGFix rankings/scores/counters in all 400 conditions, but its saved state hash is
different. Thus there is no observed additional retrieval benefit from RFix in
this screen; the combined graph must not be described as byte-identical to NGFix.

## Why the selected settings are not a speedup result

NGFix reaches 100% recall on every fitted query even at ef 100, compared with 93.88%
for the base. The original screen uses those fitted queries as calibration.
Consequently both the 95% and 99% selectors choose ef 100 for NGFix, while its
development recall is only 93.88%. This is fit/tuning reuse; it is not development
label leakage into repair, but it prevents interpreting calibration success as
an independently validated operating point.

| Target | Base selected ef | Base development recall / ms | NGFix selected ef | NGFix development recall / ms |
| --- | ---: | --- | ---: | --- |
| 95% | 200 | 97.54% / 1.908 | 100 | 93.88% / 1.115 |
| 99% | 400 | 99.42% / 3.130 | 100 | 93.88% / 1.115 |

RFix follows the base selector; the combined control follows NGFix. Neither
comparison supports an equal-recall speedup. The calibration gain of 6.12 percentage
points at ef 100 versus 0.46 points on development motivates independent training and
tuning populations; it does not by itself prove a fundamental generalization limit.

## Build, repair and resource costs

Base construction takes 6.921s. Repair loops take 0.08045s for NGFix, 0.01084s for
RFix, and 0.09096s for both. These exclude graph loading/serialization, input
preparation and exact-reference generation; they are not total learning costs.

Base/RFix serialized states contain 84,571,512 bytes; NGFix/combined states contain
84,586,632 bytes, an increase of 15,120 bytes. This is serialized storage, not resident
memory or an independently counted edge increase. Whole sample-worker peak RSS is
420,588KiB with four graphs and verification work resident; it is not per-index
memory. No large input or index was copied into the returned evidence bundle.

## Slurm warnings

The complete log contains **five** `couldn't chdir` fallbacks to `/tmp`: during
tests, both native fixture steps, sample reconstruction, and sample execution.
The abbreviated user log showed only the final two. Hardware evidence records two
XFS mount entries for `/tmp`, including a job-specific local-device mount. This is
consistent with mount-visibility timing, but does not establish the site-level cause.

The requested temporary package directory was later usable by the Python/native
processes: source/input identities, relative test loading, scores, exit status and
recorded CPU affinity all pass. No result corruption is demonstrated. There is
also no controlled evidence that the warnings have zero timing effect. Retain them
with the results; for subsequent jobs, use a stable explicit step working directory,
absolute test/input paths, and record the runtime working directory and mount view.
Keep all large downloads and work files in compute-node local temporary storage.

## Next research gate

Before a further repair-performance claim, freeze disjoint fitting, parameter
selection and evaluation roles, including a declared training budget. Use the
same roles for our candidate and the direct competitor; record exact-reference
generation cost. The existing 100 queries remain exploratory, and all 381 final
held-out retrieval outcomes remain untouched. A broader independent fitting pool
and an additional dataset are still needed before a paper-level conclusion.

The next method must address transfer beyond fitted neighborhoods and be
algorithmically distinct from NGFix/RFix. A successful competitor integration is
not a third successful novel-method candidate. Do not launch a large sweep of
either previously rejected prototype on the strength of these results.

Reproduction: `build/receive-longeval-ngfix-7141656.py` performs bounded extraction;
`build/analyze-longeval-ngfix-7141656.py` verifies provenance and generates
`build/longeval-ngfix-comparison-7141656/`, including both verification reports,
1,600 per-query rows, 32 cohorts, 24 paired comparisons and a machine-readable summary.
The extraction script requires a fresh destination; the analysis script is rerunnable.
