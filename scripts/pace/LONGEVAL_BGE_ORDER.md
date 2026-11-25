# E4 on the BGE vectors: the arrival-order reversal and the engine arm

Recorded on 25 November 2026, before any job was submitted and before any neighbour search on the BGE
vectors. The vectors are those of job 7290861 (`LONGEVAL_BGE.md`), verified locally: file hashes, shapes,
unit norms, document order equal to the study's snapshot 3, and CPU re-encoding agreement.

## Part A: the arrival-order reversal (E4, frozen with the extension protocol)

As E2's full-scale order rerun (`LONGEVAL_INSERTION_ORDER_IDS.md`), with the BGE vectors: Epzero (no
compression) and FAISS HNSW, M 16, efConstruction 100, one thread, one vector at a time, in the orders as
arrived, reversed and shuffled; seeds 11, 29 and 47; searched at ef 100, 200, 400 and 800 with the 381
test queries and the 100 training (confirmation) queries. The sample is written by `bge_sample.py`
(amendment 3); builds and searches are `insertion_pilot run`, unchanged.

**Predictions** (E4's, fixed on 23 November 2026, before any BGE vector existed; quoted):

1. "For both HNSW implementations at M 16 and ef 200, snapshot-3 against snapshot-1 exact top-100
   neighbours: within-query ratio as arrived above 1.10, shuffled within [0.90, 1.10], reversed below
   0.95; the paired change in the newest cohort's absolute excess miss risk (as arrived minus reversed)
   above +1 percentage point with its interval excluding 0."
2. "Overall recall@100 differs by less than 1 point between the three orders, for each implementation."

Prediction 1 is the one the paper's claim rests on: if it fails, the arrival effect is reported as
specific to the Qwen representation. **Analysis:** `insertion_strata.orders` (within-query ratio, ef 200,
seeds averaged within query, 10,000 query-bootstrap draws) and `order_pairs.compare` (the paired change,
as arrived minus reversed) on the test queries; the same on the confirmation queries, reported as a
confirmation reading.

## Part B: the engine arm on BGE (decision 2's remaining confirmation reading)

As the third full-scale engine arm (`LONGEVAL_ENGINE_ARM.md`, package -03) on the BGE sample: Epzero's
product path (8-bit codes, `fast_build`, M 16, efConstruction 100); snapshot 1 bulk-built on all cores
and refined, snapshots 2 and 3 inserted one at a time in arrival order. Arms `grown`, `repaired8w`
(`extra_backlinks` 8, each later snapshot's insertions refined at width 100) and `rebuilt`; seeds 11, 29,
47; ef 100, 200, 400. `engine_arm.py` and `engine_arm_analysis.py --arm repaired8w` (and `--confirm`),
unchanged; all three arms in the same job.

**Decision (decision 3's margins, as before):** effect: the interval of repaired8w minus grown excludes 0,
and repaired8w's excess is at most 1.0 point on the test queries; no harm: upper 95% bounds of the oldest
cohort's rise at most 0.75 points and of recall@100's fall at most 0.5 points. The confirmation reading is
decided by the effect.

**Prediction:** both readings pass, as on the Qwen vectors and the public streaming workload.

## Jobs

One package (`longeval-bge-order-01`), run from the PACE LongEval checkout, where the BGE vectors are
(`build/pace-longeval-bge/jobs/7290861/vectors`), on exclusive Gold 6226 nodes only: Part A one job per
implementation and seed (three orders each), Part B one job per seed (three arms); nine jobs.

```bash
sha256sum longeval-bge-order-01.tar.gz
tar -xzf longeval-bge-order-01.tar.gz
(cd longeval-bge-order-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
for seed in 11 29 47; do
  for backend in epzero faiss_hnsw; do sbatch longeval-bge-order-01/scripts/pace/longeval-bge-order.sbatch order "$backend" "$seed"; done
  sbatch longeval-bge-order-01/scripts/pace/longeval-bge-order.sbatch engine "$seed"
done
```

## After completion

```bash
sacct -X --name=longeval-bge-order -S 2026-11-25 --format=JobID,State,ExitCode,Elapsed,Constraints
grep -H '^PASS\|^exit_code' longeval-bge-order-*.log
tar -czf longeval-bge-order-01-evidence.tar.gz build/pace-longeval-bge-order/jobs longeval-bge-order-*.log
sha256sum longeval-bge-order-01-evidence.tar.gz
```

The archive holds run files derived from LongEval data: copy it to the local `build` directory and never
commit it.

## Result (25 November 2026)

All nine jobs (7297889-7297897) passed on Gold 6226 nodes; the evidence archive's SHA-256 matched
the node's, every job's evidence checksums verified, and all nine wrote the same sample (cohorts, exact
targets, query and document IDs). Aggregates: `research/longeval/e4-bge-order.json` (10,000 draws).

**Part A, test queries, ef 200, snapshot 3 against 1.** Within-query ratio as arrived / shuffled /
reversed: Epzero 1.305 [1.207, 1.423] / 0.992 / 0.807; FAISS 1.240 [1.144, 1.354] / 0.980 / 0.839.
Change in the newest cohort's excess miss risk, as arrived minus reversed: Epzero +4.8 [+3.9, +5.7]
points, FAISS +4.0 [+3.1, +4.8]. **Prediction 1 held** for both implementations. Recall@100 spread
between orders: 0.60 and 0.77 points at ef 200 (0.31-0.89 across ef 100-800). **Prediction 2 held.**
Confirmation queries: the change is +4.1 [+2.0, +6.5] and +3.1 [+1.6, +4.9]; the recall spread
between orders there is 1.18 points for Epzero and 0.96 for FAISS at ef 200, above 1 point for
Epzero (a confirmation reading, not the prediction's test, reported as it is).

**Part B.** Test queries: newest excess grown 3.31, repaired8w 0.36, rebuilt 0.38 points; repaired8w
minus grown −2.95 [−3.64, −2.27]; oldest +0.08 [−0.29, +0.42]; recall@100 +0.90 [+0.56, +1.27]:
passed. Confirmation: grown 1.01, repaired8w −1.03, rebuilt −1.27; effect −2.04 [−3.33, −0.79]:
passed. **The prediction held.**
