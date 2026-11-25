# E4 on the BGE vectors: the held-out design

Recorded on 25 November 2026, before any job was submitted and before any held-out search on the BGE
vectors. The inputs are prepared by `research/longeval/bge_heldout.py` (amendment 4); the arms, the
configurations, the seeds and the analysis are the frozen ones (`LONGEVAL_HELDOUT.md`), unchanged.

## Design

As the frozen held-out run: Epzero HNSW and FAISS HNSW, each kept incrementally and rebuilt from the
identical history (`replay`); FAISS IVF-Flat with training frozen (incremental and rebuilt) and refreshed
(rebuilt); seeds 11, 29 and 47; the frozen `trajectory-development.json` and `ivf-development.json`; one
thread; the study's environment (released Epzero 0.5.0). The 381 test queries are searched at every
snapshot; the calibration cohort sets each arm's search width by the frozen rule. Exact references are
computed per snapshot from the BGE vectors by the frozen `oracle.exact_mapped`; at snapshot 3 they must
agree with `bge_sample`'s exact neighbours, or the job stops.

**Analysis:** `bge_heldout analyze`: the frozen `heldout_analysis.collect` and `analyze_core` with the
frozen labels and panel, giving D for each arm with its 95% paired query-bootstrap interval (10,000 draws)
and the decisions under the frozen ±0.01 margin; the three within-encoder arm tests form one Holm family
at 0.05 (EXTENSION_PROTOCOL_V1, section 4). The paired contrast of D between encoders (BGE minus Qwen, on
the same queries) is reported with the same margin.

## Predictions (E4's, fixed on 23 November 2026, before any BGE vector existed; quoted)

3. "IVF is much weaker. Frozen IVF's neighbour-level excess for snapshots 2+3 against 1 is below 1
   percentage point."
4. "D stays small. FAISS HNSW and IVF: D within ±0.01 (negligible under the frozen rule). Epzero: |D|
   below 0.006."
5. "Absolute miss rates differ from Qwen (BGE is a smaller model with 768 dimensions); no prediction is
   made about their level."

Predictions 1 and 2 are tested by the order experiment (`LONGEVAL_BGE_ORDER.md`).

## Jobs

One package (`longeval-bge-heldout-01`), run from the PACE LongEval checkout, where the BGE vectors are,
on exclusive Gold 6226 nodes with one computation thread: 12 HNSW jobs and 3 IVF jobs (each IVF job runs
its three modes in sequence), as the frozen held-out run.

```bash
sha256sum longeval-bge-heldout-01.tar.gz
tar -xzf longeval-bge-heldout-01.tar.gz
(cd longeval-bge-heldout-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  for backend in epzero faiss_hnsw; do
    for mode in incremental replay; do
      sbatch --job-name="le-bh-${backend}-${mode}-${seed}" \
        longeval-bge-heldout-01/scripts/pace/longeval-bge-heldout.sbatch hnsw "$backend" "$mode" "$seed"
    done
  done
  sbatch --job-name="le-bh-ivf-${seed}" longeval-bge-heldout-01/scripts/pace/longeval-bge-heldout.sbatch ivf "$seed"
done
```

## After completion

```bash
sacct -X -u "$USER" -S 2026-11-25 --format=JobID,JobName%32,State,ExitCode,Elapsed | grep -E 'JobID|le-bh'
grep -h '^PASS\|^exit_code' longeval-bge-heldout-*.log
tar -czf longeval-bge-heldout-01-evidence.tar.gz build/pace-longeval-bge-heldout/jobs longeval-bge-heldout-*.log
sha256sum longeval-bge-heldout-01-evidence.tar.gz
```

Locally: check that every job prepared identical references (`prepared.json`), then run
`python -m research.longeval.bge_heldout analyze PREPARED RELEASE EVIDENCE OUTPUT --hnsw-config
research/longeval/trajectory-development.json --ivf-config research/longeval/ivf-development.json`, with
PREPARED one job's copied `prepared` folder. The archive holds run files derived from LongEval data: never
commit it.

## Result (25 November 2026)

All 15 jobs (7298272-7298286) passed on Gold 6226 nodes; the evidence archive's SHA-256 matched the node's
(01416036…c02c3) and every job's evidence checksums verified. The jobs' `prepared.json` files differ (temporary
folders, gzip timestamps) while their content is identical, so the inputs were checked by content
(amendment 6, recorded before the analysis ran). Aggregates: `research/longeval/e4-bge-heldout.json`.

- **History:** all 27 incremental-against-replay comparisons identical; no excluded attempt.
- **Prediction 4 held.** D (GPT-OSS-120B, 374 queries, 10,000 draws): Epzero +0.0016 [−0.0040, +0.0084],
  FAISS HNSW −0.0027 [−0.0093, +0.0028], frozen IVF +0.0005 [−0.0012, +0.0021]; all three negligible under
  the frozen ±0.01 rule (Holm p 1.0), and Epzero's |D| is below 0.006.
- **Prediction 3 held.** Frozen IVF's neighbour-level excess, snapshots 2+3 against 1 at snapshot 3: −0.02
  [−0.32, +0.30] points (within-query ratio 0.995). HNSW at ef 200: Epzero 1.245 [1.163, 1.339], FAISS 1.193
  [1.122, 1.279] (`bge_heldout strata`, the computation of `insertion_strata.heldout` with the BGE loader).
- **Encoders (reported, BGE minus Qwen, same queries):** FAISS −0.0020 [−0.0097, +0.0048] and IVF +0.0020
  [−0.0008, +0.0050], within the margin; Epzero +0.0075 [−0.0017, +0.0205], not within it (Qwen's D was
  −0.0059, BGE's +0.0016).
- **Exact search** loses 0.037 nDCG@10 [0.026, 0.048] from snapshot 1 to 3 with BGE (Qwen: +0.007).
