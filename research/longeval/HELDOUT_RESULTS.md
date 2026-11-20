# Held-out results under FINAL_PROTOCOL_V1 (2026-11-20)

These are the first and only held-out outcomes. The protocol was frozen on 2026-11-19 (receipt
`FINAL_PROTOCOL_V1_FREEZE.json`) before any test-query retrieval took place. The analysis is
the frozen `heldout_analysis.py heldout`, and nothing in it was changed after the judgments were
read.

## Execution and verification

| Item | Value |
| --- | --- |
| Test queries | 381, encoded by job 7164678 |
| Exact references | Computed locally after the freeze. Re-scoring the calibration queries matched the development references exactly (50 of 50 rankings, score difference 0.0, all 3 snapshots) |
| PACE jobs | 7168978–7168992: 12 HNSW and 3 IVF, all `COMPLETED`, exit 0, on Gold 6226 |
| Evidence | `build/longeval-heldout-01-evidence.tar.gz`, SHA-256 `886047e02cdabb1d73e914b019f6cd80f0d34acf0bcd0dd6d57326cd95fabf1f`, identical on PACE and locally |
| Integrity | The evidence checksums of all 15 jobs verify. All 15 ran package manifest `d040f87d…`, and no frozen file had changed. There is one Epzero binary, identical to the development one |
| Analysis | 21 conditions and 153 runs. Every reported ANN recall was recounted against the exact references. pytrec_eval parity held to 1e-12. No attempt was excluded |
| Output | `build/longeval-heldout-analysis-01/report.json` and `per-query.csv` (LongEval-derived; not committed) |

## Primary result (RQ2): GPT-OSS-120B labels, panel of 374 queries

The contrast is D = (ANN − exact nDCG@10 at snapshot 3) − (the same at snapshot 1). It is averaged
over three seeds within each query. The test is sign randomization with 100,000 draws; the
intervals come from 10,000 bootstrap samples; and the p-values are Holm-adjusted over three
tests. The margin is ±0.01.

| Arm | D | 95% interval | p | Holm p | Label |
| --- | --- | --- | --- | --- | --- |
| Epzero HNSW, incremental, calibrated ef | −0.0059 | [−0.0129, −0.0006] | 0.039 | 0.117 | **inconclusive** |
| FAISS HNSW, incremental, calibrated ef | −0.0007 | [−0.0038, +0.0027] | 0.687 | 0.687 | **negligible** |
| FAISS IVF-Flat, frozen training, calibrated nprobe | −0.0015 | [−0.0037, +0.0005] | 0.145 | 0.291 | **negligible** |

The Epzero interval crosses −0.01, so it is not negligible. Its Holm-adjusted p is above 0.05,
so it is not called changed.

### ANN-minus-exact gap by snapshot (secondary)

| Arm | Snapshot 1 | Snapshot 2 | Snapshot 3 |
| --- | --- | --- | --- |
| Epzero HNSW, calibrated (equal to fixed ef 200) | −0.0057 | −0.0102 | −0.0117 |
| FAISS HNSW, calibrated | −0.0048 | −0.0049 | −0.0055 |
| FAISS HNSW, fixed ef 200 | −0.0048 | −0.0064 | −0.0069 |
| IVF frozen, calibrated nprobe | −0.0065 | −0.0079 | −0.0080 |
| IVF frozen, snapshot-1 nprobe held fixed | −0.0065 | −0.0064 | −0.0065 |
| IVF refreshed training, calibrated | −0.0065 | −0.0066 | −0.0082 |

Every gap is below 0.012 nDCG@10 in absolute value.

The secondary D for FAISS HNSW at fixed ef is −0.0022 [−0.0053, +0.0013]. For IVF at fixed
nprobe it is −0.0000 [−0.0018, +0.0018], and for refreshed IVF −0.0017 [−0.0051, +0.0014].

### Why the two HNSW arms differ: calibration, not implementation

Calibration chooses the smallest ef in {100, 200, 400, 800} that reaches 95% mean recall@100 on
the 50 calibration queries.

- **Epzero:** calibration recall stayed at 95.1–95.6% at ef 200 for every seed and snapshot, so
  ef never changed. The calibrated Epzero runs are identical to the fixed ef-200 runs. On the
  test queries, Epzero's mean neighbour recall fell from 94.5% to 93.5%, and its worst-decile
  mean fell from 0.69 to 0.65.
- **FAISS HNSW:** for seeds 29 and 47, calibration recall at ef 200 fell to 94.4–94.6% at
  snapshots 2 and 3. Calibration therefore stepped up to ef 400, and test recall rose to 95.7%
  (snapshot 2) and 95.3% (snapshot 3).
- **At the same fixed ef 200:** the two implementations behave alike on the test queries. Recall
  is 94.5→93.5% for Epzero and 94.4→93.6% for FAISS, and the snapshot-3 gaps are −0.0117 and
  −0.0069.

In this data, the change in the effectiveness gap follows the change in neighbour recall. A
calibrated setting holds the gap only if a coarse, 50-query calibration actually triggers.

## RQ1: index history (held-out)

Each incremental index matched its identical-history rebuild in all 27 comparisons:

- 2 HNSW implementations and IVF;
- 3 seeds and 3 snapshots;
- state fingerprints, and every ranking and score at every search setting.

This confirms the development finding on the test queries.

## RQ3: search failures (held-out, mean over three seeds)

| Arm, setting | Recall@100 by snapshot (1 / 2 / 3) | Worst-decile mean at snapshot 3 | Zero-recall queries (snapshots 1–3) |
| --- | --- | --- | --- |
| Epzero, ef 200 | 0.945 / 0.939 / 0.935 | 0.646 | 0 / 0 / 1 |
| FAISS HNSW, ef 200 | 0.944 / 0.942 / 0.936 | 0.654 | 0 / 0 / 0 |
| FAISS HNSW, calibrated | 0.944 / 0.957 / 0.953 | 0.724 | 0 / 0 / 0 |
| IVF frozen, calibrated | 0.964 / 0.956 / 0.958 | 0.793 | 0 / 0 / 0 |
| IVF frozen, fixed nprobe | 0.964 / 0.965 / 0.966 | 0.829 | 0 / 0 / 0 |

The IVF all-lists setting returns the exact neighbours up to boundary ties (recall 0.9999–1.0).
The IVF calibrated nprobe was 128 everywhere, except seed 29 at snapshot 1, where it was 256.

## Exact search over time (not an index effect)

| Labels (panel) | Exact nDCG@10 at snapshots 1 / 2 / 3 | Snapshot 3 − snapshot 1 [95% interval] | Judged@10 |
| --- | --- | --- | --- |
| GPT-OSS-120B (374) | 0.804 / 0.810 / 0.811 | +0.0070 [−0.0000, +0.0139] | 0.995–0.996 |
| GPT-OSS-20B (377) | 0.806 / 0.810 / 0.812 | +0.0061 [−0.0011, +0.0135] | 0.995–0.996 |
| Raw clicks (80) | 0.159 / 0.141 / 0.092 | −0.0670 [−0.1154, −0.0210] | 0.11 → 0.06 |
| DCTR (10) | 0.147 / 0.079 / 0.069 | −0.0787 [−0.198, 0.0] | 0.38 → 0.08 |
| Human (29, snapshot 3 only) | 0.564 at snapshot 3 | not applicable | 0.97 |

The click-based labels are sparse: judged@10 is 0.06–0.38, and coverage falls as the collection
grows. Their apparent decline is confounded by that coverage.

## Sensitivity families (secondary, never pooled)

- **GPT-OSS-20B (377 queries)** reproduces the primary pattern:
  - Epzero D −0.0074 [−0.0142, −0.0019], unadjusted p 0.007;
  - FAISS calibrated D −0.0017 [−0.0049, +0.0017];
  - IVF frozen D −0.0014 [−0.0035, +0.0007].
- **Human labels (29 queries, snapshot 3):** every ANN-minus-exact gap lies within ±0.0005.
- **Raw clicks (80 queries) and DCTR (10 queries):** these give positive or zero D with wide
  intervals. Their judged@10 coverage is too low for an index comparison to be informative.

## What the paper can claim

1. **RQ1.** In append-only growth, an incrementally maintained HNSW or IVF index is bit-identical
   to a rebuild from the same insertion history. This holds on held-out queries, for both
   implementations and all seeds.
2. **RQ2.** At about 95% calibrated neighbour recall, the ANN-minus-exact nDCG@10 gap stays below
   0.012 at every snapshot. For FAISS HNSW and frozen IVF, its change from snapshot 1 to
   snapshot 3 is negligible within ±0.01. For Epzero, whose calibration never raised ef, the gap
   widened from −0.006 to −0.012. That change is not negligible and is not significant after
   Holm adjustment. GPT-OSS-20B shows the same direction.
3. **Mechanism.** The change in the gap follows the change in neighbour recall on the evaluated
   queries. A fixed search setting lets recall drift down as the collection grows, and a
   calibrated setting prevents this only if its calibration set detects the drift.
4. **Practical implication.** For longitudinal evaluations like LongEval, exact-search
   conclusions transfer to an ANN index to within about 0.01 nDCG@10. This requires the index to
   be held at about 95% neighbour recall on the evaluated queries, re-checked at each snapshot,
   rather than kept at a setting chosen once.

## Limitations

- One collection and one embedding model. The data is append-only. The primary labels are
  LLM-generated.
- Calibration uses 50 queries and a coarse grid that doubles ef or nprobe at each step. This is
  the protocol's design, and it is the reason the Epzero and FAISS arms diverge.
- Latency and HNSW distance-count outcomes come from separate timing jobs, which are still to
  run.
