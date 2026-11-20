# Final evaluation protocol v1 — FROZEN

This protocol was approved by the user on 2026-11-19 as FINAL_PROTOCOL_V1_CANDIDATE.md. That file
is kept unchanged. This version adds the execution mapping below and three clarifications that
do not touch the primary analysis.

Frozen by `freeze_protocol.py`. The receipt `FINAL_PROTOCOL_V1_FREEZE.json` holds hashes of this
file, both configurations, every research module, the job script and the packager. It also
records that no held-out retrieval outcome existed when the protocol was frozen. Any later
change produces v2, which must state which test outcomes were already visible.

## Scope

This is an empirical analysis paper for the CLEF 2027 conference track, and for LongEval 2027
working notes if that lab runs. It reports how approximate search, index history and
collection growth affect dense retrieval over three snapshots, together with four
pre-registered method screens that failed at equal work. It makes no new-method claim.

## Research questions

- **RQ1 (history).** Does keeping an index alive while the collection grows change its results
  compared with rebuilding from the same insertion history?
- **RQ2 (approximation over time).** How large is the signed ANN-minus-exact effectiveness gap at
  each snapshot, and does it change from snapshot 1 to snapshot 3?
- **RQ3 (search failures).** How do neighbour recall, failure tails and search work change with
  growth, at a fixed search setting and at a calibration-adjusted one?
- **RQ4 (repair, development only).** Do query-driven graph repairs beat wider search at equal
  work? This is reported from the existing development screens. There are no held-out runs.

## Data and representation

The original archived document vectors for the three cumulative snapshots are used unchanged.

The 381 test queries were encoded once with Qwen3-Embedding-4B (pinned revision and prompt) on
CUDA in FP32 with TF32 off and batch size 1, as for the training queries. This was job 7164678,
and its training-query re-encoding check had a minimum cosine of 0.9999999999959. The exact and
approximate arms use the same query vectors.

The 50 calibration training queries set every search parameter. The 50 development queries are
not used held-out.

## Arms

All arms run single-threaded on Gold 6226 nodes with construction seeds 11, 29 and 47.

| Arm | Settings | Execution |
| --- | --- | --- |
| Exact | Float64 exhaustive top-100 | `heldout.py exact` (wraps `oracle.exact_mapped`), run once per snapshot after the freeze |
| Epzero HNSW, incremental | M=16, efC=100. Two settings: ef 200 held fixed, and a calibration-selected ef (smallest in {100, 200, 400, 800} reaching 95% mean recall@100 on the calibration queries at that snapshot) | `heldout.py hnsw --backend epzero --mode incremental` (wraps `trajectory.trajectory_core`). Settings are named `frozen` and `matched_recall` |
| FAISS HNSW, incremental | Same as Epzero | `--backend faiss_hnsw --mode incremental` |
| Epzero / FAISS HNSW, identical-history rebuild | Same | `--mode replay` (for RQ1 only) |
| FAISS IVF-Flat | nlist 1024, 65,536 training rows. Three settings: the nprobe selected at snapshot 1 held fixed (`frozen_snapshot1`); the nprobe re-selected at each snapshot (`current_calibration`, smallest in {1, 2, …, 1024} reaching 95%); and all lists (`all_lists`) | `heldout.py ivf` with modes `frozen_incremental` (the primary IVF arm), `frozen_rebuild` (for RQ1) and `refreshed_rebuild` |

The calibration references are the verified development exact references. When the held-out
exact job re-scores the calibration queries, it must agree with them up to float rounding
(`reference_agreement`). All seeds are kept and averaged within each query; no seed is selected.

## Judgments and populations

- **Primary judgments.** The released GPT-OSS-120B judgments, specific to each snapshot. The
  panel is fixed: the test queries that have at least one positive judgment at all three
  snapshots, computed by `heldout_analysis.panel` (the audit found up to 374 such queries).
  These labels are automated, and the paper says so.
- **Sensitivity analyses.** These are reported separately and never pooled. Each family uses its
  own panel under the same rule: raw clicks, DCTR, and GPT-OSS-20B at all three snapshots; human
  labels at snapshot 3 only.
- **Unjudged documents** count as non-relevant. Judged@10 coverage is reported for every run.

## Outcomes

- **Primary:** nDCG@10, with linear gains and deterministic tie-breaking. It is checked against
  pytrec_eval to 1e-12.
- **Secondary:** relevant-document R@100, judged@10, and ANN neighbour recall@100 against exact
  search (mean, worst-decile mean, minimum, and counts of zero-recall and below-0.5 queries).
  For IVF, the vectors scanned per query.

## Primary contrasts (one Holm family of three tests)

The three ANN arms A are:

- `epzero_hnsw` = (epzero, incremental, matched_recall);
- `faiss_hnsw` = (faiss_hnsw, incremental, matched_recall);
- `ivf_frozen` = (faiss_ivf, frozen_incremental, current_calibration).

All use the primary panel.

    D_A = [nDCG(A, s3) - nDCG(Exact, s3)] - [nDCG(A, s1) - nDCG(Exact, s1)]

Each contrast is computed for every query and seed, averaged over seeds within each query, and
then averaged over queries (`paired_inference.paired_query_estimate`).

- **Test:** a two-sided paired sign-randomization test with 100,000 draws, Holm-adjusted at 0.05.
- **Intervals:** 95% paired query-bootstrap percentile intervals from 10,000 samples.
- **RNG seed:** 20261119.

**Practical margin: 0.01 absolute nDCG@10.** Each contrast gets one label:

- *negligible* if its whole 95% interval lies within ±0.01;
- *changed* if it is not negligible and its Holm-adjusted p is below 0.05;
- *inconclusive* otherwise.

A non-significant result alone is never called equivalence.

## Secondary analyses (no multiplicity control, labelled as secondary)

- The gap at each snapshot, ANN minus exact, for each primary arm and for the fixed-setting
  arms. The fixed-setting arms are ef 200 for HNSW, the snapshot-1 nprobe for frozen IVF, and the
  refreshed IVF arm.
- D_A for the fixed-setting and refreshed arms.
- Every quantity above for each sensitivity family. For human labels this is the snapshot-3 gap
  only.
- The exact-search trend from snapshot 1 to snapshot 3. This is a property of the
  representation and the labels, not of the index.

**RQ1** is an equality check, not a test. For every seed and snapshot, the following must match
exactly between the incremental index and its identical-history rebuild:

- the state fingerprints: everything except the parent link for HNSW, and the serialized index,
  live set and insertion order for IVF;
- all rankings and scores.

Any mismatch is reported as a finding.

## Clarifications relative to the approved candidate

1. **IVF RQ1 check.** The frozen-training IVF rebuild (`frozen_rebuild`) also runs, so the RQ1
   equality check covers IVF as it did in development. It does not enter the primary family.
2. **Cost outcomes.** HNSW distance counts and native latency at the calibrated settings come
   from separate timing jobs run after the primary results. Those jobs cannot change any primary
   input or output. IVF scan counts come from the primary jobs.
3. **Human labels.** These exist only at snapshot 3, so they support a snapshot-3 gap and no D_A.

## Reporting rules

- Report every arm, seed, snapshot and missed calibration target.
- Development and test findings are labelled separately. There are no significance claims on
  development data.
- Failed job attempts are listed (`excluded_attempts`) and never analysed.
- Invalid executions get a documented repair. Unfavourable valid outcomes stay in.

## Pipeline test before the freeze

`heldout_analysis.py dry-run` ran the whole analysis on the 50 development queries and the
verified development evidence: 21 conditions and 153 runs. It reused the snapshot-1 training
judgments at every snapshot. Its output, `build/longeval-heldout-dryrun-01`, tests the code and
supports no claim. Its receipt hash is in the freeze receipt.
