# Research log

A record of decisions, PACE jobs and verifications, kept as the work happened. The
[README](../README.md) gives the current summary.

**Research objective updated 2026-11-18:** the author wants a new method paper with
competitive performance. [METHOD_RESEARCH_PLAN.md](repair/METHOD_RESEARCH_PLAN.md) now
governs development. An additive overlap-guided frontier-resumption prototype passes
10 tests and completed an 800-condition small real-vector pilot. Its 400 equal-work
controls show no persuasive benefit over wider search; this version is not promoted
to a large sweep. See [FRONTIER_PILOT_RESULTS.md](repair/FRONTIER_PILOT_RESULTS.md). Novelty,
full-data gains and native speedups remain unproven. Earlier empirical findings remain
supporting evidence. The 381 held-out query retrieval outcomes remain untouched.

**Paper scope decided (2026-11-19):** an empirical analysis paper (CLEF 2027 conference
track; LongEval 2027 working notes if that lab runs), then Epzero distance-cost work.
The author approved [FINAL_PROTOCOL_V1_CANDIDATE.md](protocol/FINAL_PROTOCOL_V1_CANDIDATE.md):
GPT-OSS-120B primary labels on the fixed all-snapshot panel, 0.01 nDCG@10 margin,
three-test Holm family. Not yet frozen. Step 1, test-query encoding (vectors only), is
done: job 7164678 (package `longeval-test-queries-03`, A40, CUDA FP32, TF32 off) encoded all
381 test queries after re-encoding the 100 training queries at min cosine 0.9999999999959
against job 7107422. Evidence `build/longeval-testq-7164678-evidence.tar.gz` (SHA-256
`4a4f9f576067b48bf129fce7fa19ba10d83624aede712ed9bb8a5d25e8ab2486`) re-verified locally:
encoding checksums, manifest matches the -03 receipt, `load_test_queries` returns 381 unit
vectors with unique IDs. No retrieval or judgment was touched. Earlier attempts: job 7160008
(wrong package path) and job 7163327 (second `set_num_interop_threads` call in one process,
fixed in `encoder.py`).

**Protocol v1 frozen (2026-11-19):** [FINAL_PROTOCOL_V1.md](../FINAL_PROTOCOL_V1.md) with receipt
`FINAL_PROTOCOL_V1_FREEZE.json` (55 hashed files; no held-out outcome existed). Pipeline:
`heldout.py` (exact references, held-out HNSW/IVF entry points; `trajectory_core` and `ivf_core`
gained an `evaluation_role`, byte-identical to the development path in tests), `heldout_analysis.py`
(evidence recount, panels, D_A with Holm and the ±0.01 margin, RQ1 equality),
`freeze_protocol.py`, `scripts/pace/longeval-heldout.sbatch`, `package-longeval-heldout.py`,
`LONGEVAL_HELDOUT.md`. Pipeline test on development evidence (`build/longeval-heldout-dryrun-01`):
21 conditions, 153 runs, 27/27 RQ1 equality comparisons identical; numbers are not results.
Held-out exact references computed locally after the freeze (`build/longeval-heldout-exact-01`,
69/118/140 s); the calibration re-scoring matched the development references exactly (50/50
identical rankings, score difference 0.0, all snapshots). PACE package
`build/longeval-heldout-01.tar.gz` (SHA-256
`9c6049fcbf1b1f7de9b16703f375cda9c8baf886fe33384cf3ee81eb4674bf3e`, 97 files) passed the
frozen-file check, preflight and an extracted-package test run. Next: 15 PACE jobs per
`scripts/pace/LONGEVAL_HELDOUT.md`; no judged score has been computed.

**Held-out results (2026-11-20):** jobs 7168978–7168992 all completed; evidence verified and the
frozen analysis run once. See [HELDOUT_RESULTS.md](../HELDOUT_RESULTS.md). Primary D (GPT-OSS-120B,
374 queries): FAISS HNSW −0.0007 and IVF frozen −0.0015 are negligible within ±0.01; Epzero
HNSW −0.0059 [−0.0129, −0.0006] is inconclusive (Holm p 0.117) because its calibration never
raised ef while test recall fell 94.5→93.5%. RQ1 held: 27/27 identical.

**After the held-out results (2026-11-20):** research statement and PDF updated with the results
(4 pages). Timing package `build/longeval-heldout-cost-01.tar.gz` (SHA-256
`ae5dd9455c6b1725b3fdf586f1e07cd52faaf90c5aaedbc777053ee8f9e302a3`; new `heldout_cost.py`, no
frozen file changed) is ready: 3 jobs per `scripts/pace/LONGEVAL_HELDOUT_COST.md`. Manuscript
draft `paper/manuscript-clef2027.md` (timing section is a placeholder). Then direction B.

**Statement of research updated (2026-11-21):** `reports/longeval-research-statement.md` and its
PDF were rewritten for submission as a short statement: motivation, questions, approach and
findings in outline, without method details or result tables. The target is CLEF 2027. The
detailed account is the manuscript.

**Exploratory (post-freeze, 2026-11-20):** [HELDOUT_EXPLORATORY.md](../HELDOUT_EXPLORATORY.md).
HNSW misses documents inserted after snapshot 1 at 1.54–1.63x the rank-matched rate
(IVF 1.14–1.17x); in development, shuffled-order rebuilds lower it in 4/4 comparisons. Dropped
top-10 neighbours are less relevant than kept ones; 76–82% of HNSW replacements are unjudged.
Condensed-list nDCG shrinks the gaps by roughly half to three-quarters with the same temporal
pattern. Figures in `paper/figures/` (`paper/figures.py`); manuscript updated. Fresh-content
queries (77 of 374; ≥50% of exact top 10 new) show Epzero gap −0.026 vs −0.008 for the rest.
Timing jobs 7178503–7178505 submitted. Exploratory extension package
`build/longeval-heldout-ext-01.tar.gz` (SHA-256
`510670544d7be40094e4e66307a91cf06a7a3fbe78c3673345d5e1934edc3b79`; held-out shuffled-order arm
+ ef sweep, 6 jobs per `scripts/pace/LONGEVAL_HELDOUT_EXT.md`) is ready.

**Timing and extension returned (2026-11-20):** all 9 jobs passed; evidence verified; rebuilt
incremental indexes reproduced the analysed runs. Timing: HNSW ef 200 ≈2.5 ms, ef 400 3.9 ms,
ef 3200 ≈31 ms; IVF calibrated 129–184 ms. Held-out shuffled order lowers the new-document miss
ratio (1.54–1.63 → 1.21–1.25) but does not improve nDCG; at ≈98% recall (ef 800) the gap is
−0.002 to −0.003 at every snapshot. Manuscript §4.4–4.5, abstract, discussion and Figure 3
updated. Remaining: optional LLM judging of replacements, LNCS conversion, then direction B.

**Third method direction rejected (2026-11-19):** job 7151090 encoded the pseudo-queries
(A40, encoder gate passed). On the sample they are closer to real queries than documents
are, and their bridge edges cut answer-set fragmentation (6.80 to 5.74 components), but
every encoder arm loses 0.04–0.39 pp recall at equal distance work, and a documents-as-
queries control does better. See [PSEUDO_QUERY_RESULTS.md](repair/PSEUDO_QUERY_RESULTS.md).
No novel method is validated.

**Third method direction prepared (2026-11-18):** encoder-synthesized title
pseudo-queries for query-aware graph repair, with no query log. On the local
8,192-document sample, real queries' exact top-100 sets split into 6.6 components
of the exact document 16-NN graph, against 3.8 for document neighborhoods. GPU encoding
package `build/longeval-pseudoquery-01.tar.gz` (SHA-256
`4f6b20d5f3729d5b76d1612ad845dabe013ca70b7abf4784552e267f280285de`) is queued as
job 7151090 on an A40 (A100s unavailable). Unsubmitted fallback v02 (`16925220…f760`)
lowers the GPU-memory minimum to 20 GiB and checks it before setup.
Feasibility gates are fixed in [PSEUDO_QUERY_PLAN.md](repair/PSEUDO_QUERY_PLAN.md); no retrieval
result exists yet. Commands: [LONGEVAL_PSEUDOQUERY.md](../../../scripts/pace/LONGEVAL_PSEUDOQUERY.md).

**Latest verified screen:** job 7148412 completes the three disjoint 30/20/50 splits.
All 24 tuning-selected control/target combinations meet development mean recall;
every control chooses ef 200 for 95% and ef 400 for 99%. NGFix adds 0.04–0.18 percentage
points at ef 200 with similar observed latency, without a speedup. All 506 packaged
files and 505,600 score occurrences verify. The working-directory warnings are absent.
Raw base-state hashes differ across builds although 1,200 baseline ranking/score/counter
conditions match; unused serialized capacity is a demonstrated possible cause, not
a proven explanation of the unreturned PACE bytes. Future adapters need explicit
delete-flag initialization and canonical state fingerprints. See
[NGFIX_DISJOINT_RESULTS.md](repair/NGFIX_DISJOINT_RESULTS.md). No novel method is validated.

**New candidate feasibility result:** a fit-only regional progress certificate
passes zero of 58 overlapping proposal/split cases. Most proposals do not even
improve on their source's score, which can still be compatible with finding more
top-100 neighbors. Seven tests pass; do not promote this rule to PACE. See
[REGION_MARGIN_RESULTS.md](repair/REGION_MARGIN_RESULTS.md). A distinct effective coverage
method remains the central open research problem.

**Earlier NGFix screen:** job 7141656's returned evidence verifies locally:
500 packaged files, 1,600 real plus 256 synthetic conditions, and 185,600 score
occurrences independently recomputed. NGFix improves sample development recall
from 93.42% to 93.88% at ef 100, versus 100% on its fitted queries. Selecting ef on those
same fitted queries misses both 95% and 99% development targets; this is not an
equal-quality speedup. Job 7148412 subsequently implements separate fitting and tuning.
The complete log contains five working-directory warnings; one-core affinity and
input/source checks pass, but their exact Slurm cause remains unresolved. See
[NGFIX_RESULTS.md](repair/NGFIX_RESULTS.md). This verifies a competitor, not a new method.

**Previous launch failure:** NGFix job 7141318 failed after 2s (exit 1:0); the returned log
confirms the missing-module error. A Python import-shadowing defect in the v02 launcher
was reproduced locally and fixed in isolated **v03**, with 22 packaged tests passing
and successful launch from the historical checkout. Native algorithm and inputs are
unchanged. See [failure and corrected commands](repair/NGFIX_LAUNCH_FAILURE_7141318.md).

**Full-data native comparison:** native-cost jobs **7139467 / 7139468 / 7139469** are fully
verified: 18 baseline states reproduced, 54 cohorts, 2,700 query conditions and 8,100
timed calls. At final ef 200, Epzero takes 2.67–2.73ms versus FAISS 2.46–2.52ms;
Epzero seed 11 still has two zero-recall failures. At ef 3200 both exceed 99.5% mean
recall, with FAISS lower observed mean latency. Calibration-selected FAISS uses
ef 400 for seeds 29/47 and achieves higher recall, so its higher latency there is
not an equal-quality Epzero win. See [NATIVE_COST_RESULTS.md](development/NATIVE_COST_RESULTS.md).
The baseline timing gap is closed; no new method has passed the promotion gate.

**Method-development continuation:** a full-data phase audit verified all 1,800 complete traces
and 30,126,016 scored-edge events. At the selected 4,000-call prefix, every final
development query encounters an exact neighbor, but 523 answer occurrences remain
missing across three seeds. The main residual target is neighborhood coverage;
see [TRACE_PHASE_RESULTS.md](development/TRACE_PHASE_RESULTS.md). Closely related published
NGFix/RFix now constrains the novelty claim. Its controlled native AVX-512 sample
screen is independently verified; it shows modest development transfer and strong
fit/tuning reuse effects. The isolated **PACE screen package v03** corrects v02's
launch-path defect; 22 packaged tests pass and 11 tests ran on PACE. Synthetic checks
preceded corpus download; large data stayed on compute-node temporary disk. See
[NGFIX_REPRODUCTION.md](repair/NGFIX_REPRODUCTION.md) and
[corrected PACE commands](repair/NGFIX_LAUNCH_FAILURE_7141318.md). Seven trace-audit tests
pass. No new method has passed the promotion gate.

**Previous method screen:** a second, calibration-only witness-repair candidate
completed 800 conditions but did not meaningfully transfer to development queries.
All 32 added-edge destinations were calibration exact neighbors with single-query
support. A full-corpus coverage audit finds only 7.62% shared development/calibration
answer occurrences at snapshot 3; this is a coverage diagnostic, not a bound on repair.
See [WITNESS_REPAIR_RESULTS.md](repair/WITNESS_REPAIR_RESULTS.md). Neither prototype is
promoted to a large sweep. New native adapters for Epzero, FAISS and hnswlib 0.8.0
passed six tests and a nine-graph, three-seed, 3,600-condition local sample comparison;
FAISS has lower times at its calibration-selected settings. See
[STANDARD_GRAPH_RESULTS.md](repair/STANDARD_GRAPH_RESULTS.md). Nine repair tests also pass.

Earlier verified PACE results: **7133623**, **7133624**, **7133625**, the three-seed IVF
maintenance controls. All nine trajectories and 27 snapshot conditions are independently
verified locally. See [IVF_RESULTS.md](development/IVF_RESULTS.md). Earlier completed work includes
[development relevance](development/DEVELOPMENT_RELEVANCE_RESULTS.md) and full-cohort HNSW work-budget
jobs **7130605**, **7130606**, **7130607**. All relevance labels used so far belong to
snapshot-1 training queries; later scores are historical-label diagnostics. Earlier source-bundled plans and
status notes are retained as historical records; this file identifies the latest evidence.

## Complete

- Full fixed-text and fixed-vector audits of all three snapshots; query-encoding diagnostics
  and full-corpus exact reference generation for the development/calibration training split.
- Eighteen HNSW trajectories: two backends, three construction modes, three seeds. All
  results verified. Rebuilding the same insertion history reproduces the persistent index
  in every tested condition. See [HNSW_THREE_SEED_RESULTS.md](development/HNSW_THREE_SEED_RESULTS.md).
- Focused failure diagnosis for Epzero seeds 11 and 29. Larger search effort recovers two
  seed-11 failures on the same graph and upper route. See
  [HNSW_FAILURE_DIAGNOSTIC_RESULTS.md](development/HNSW_FAILURE_DIAGNOSTIC_RESULTS.md).
- Controlled deferred-candidate experiment: all 144 search conditions verified through
  complete logs and independent score checks. Two targeted rescues recover 99% recall;
  all 48 paired wider searches produce identical rankings at equal distance counts.
- Full-cohort work-budget diagnostic: all 100 training queries, three snapshots and three
  seeds; 1,800 complete traces and 6,300 budgeted rankings verified. All nine calibration
  conditions select 4,000 evaluations. Final development means are 96.18%, 96.68%, 96.68%,
  with no zero-recall queries at that budget. The worst final per-query recalls remain
  64%, 66%, 69%. This is offline search-work analysis, not native capped-query latency.
- IVF: same-training rebuilding reproduces incremental states/rankings in all nine pairs.
  At final current-calibrated nprobe=128, retraining changes development neighbor recall
  by -0.24/-0.12/-0.76 pp across seeds 11/29/47. At seed 29's original nprobe=256,
  retraining instead improves recall by 0.34 pp. All regimes are retained. Local verification
  covers 416,691 scored-pair checks, 18,900 coarse-probe checks and nine distinct training
  samples. All-list searches recover exact top-100 sets. Native IVF costs and historical-
  label relevance are analyzed.

## Interpretation

The evidence supports history/seed-sensitive search failures in these development cases,
with candidate retention explaining a sufficient recovery intervention. It does not establish
general degradation from keeping an index alive, an improved general search algorithm, or
a novel publishable contribution. The two selected rescues are diagnostic rather than a
deployable selection rule. Q2's rescue requires ef=1600 and does not trigger at ef=200.
At the full-cohort selected work budget, the earlier failing queries reach 85% and 87%,
using the wide-search trajectory. Their zero-recall collapse is avoided, while individual
shortfalls remain. Frozen snapshot-1 and current calibration budgets coincide on this grid.

## Next work and paper requirements

The method-development and competitor gates in METHOD_RESEARCH_PLAN.md precede final
protocol freeze. The native timing work below remains useful baseline infrastructure;
its existing package does not evaluate the new prototype.

Before another learned-repair experiment, specify a structural signal that can help
outside calibration answer IDs or a larger independent training-query protocol.
Do not consume held-out queries for this purpose or retune the failed screens.

1. Execute and verify the prepared NGFix/RFix integration screen, then establish a
   distinct mechanism that improves neighborhood coverage outside fitted answer IDs.
2. Extend native quality–latency curves and competitor comparisons for that mechanism.
   The original HNSW/IVF cost comparison is complete; it does not cover a new method.
   Keep FAISS native counters separate from Epzero's audited counters.
3. Freeze the primary representation, hypotheses, practical-effect margins and inferential
   protocol before using held-out evaluation for final claims.
4. Establish a defensible contribution against related work, evaluate on untouched held-out
   queries, and prepare the reproducible manuscript and artifact.

The returned IVF and native HNSW cost jobs are fully analyzed. Historical submitted
native-cost package: `build/longeval-pace-native-cost-02.tar.gz`,
1,958,413 bytes, SHA-256 `3f1d9c560d463c1c2c327d1c71ee9b5329e94eb30c49cc8c469e1292250522fc`.
All 127 original source files and all 81 prior supplemental files are preserved; the new
supplement has 99 files. The pre-IVF-analysis status/manuscript and their sealed record
are archived in `build/longeval-pre-ivf-analysis-02.tar.gz`. The prior work-budget analysis, including its status document,
is preserved in `build/longeval-work-budget-analysis-01.tar.gz` (19 entries, SHA-256
`672e46c8ce3ec8f1c6b3285befe3da389e0c2995b9c417a9256cef996127e47a`).
The earlier retention analysis remains in `build/longeval-retention-analysis-01.tar.gz`.
See [completion plan](history/COMPLETION_PLAN.md) and the
[unfrozen final evaluation draft](protocol/FINAL_EVALUATION_PROTOCOL_DRAFT.md). Held-out retrieval
evaluation has not started. Publication readiness and novelty remain unestablished.
The historical [working manuscript](history/manuscript-development-draft.md) contains the earlier empirical
development results; new method/cost records above govern the current work. Query-clustered
paired bootstrap/randomization and Holm-adjustment utilities are implemented and tested,
but have not been used to assign significance to development findings. Seven analysis
tests pass; the final inference population/contrasts/margin are still unfrozen.
All future large PACE downloads and work files
must stay in the compute node's temporary directory; persist only bounded result evidence.
