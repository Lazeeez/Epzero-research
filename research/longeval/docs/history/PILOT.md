# Synthetic pilot (historical)

This documents the early synthetic pilot that validated the pipeline before real data were used. It
is kept for reproducibility; the study itself is summarized in the [README](../../README.md).


This is an executable **development pilot**, not a completed LongEval experiment or evidence of
novelty. It separates input validation, explicit index construction, exhaustive reference scoring,
calibration, evaluation, and provenance. See [STATUS.md](STATUS.md) for the remaining research gates
and [NATIVE_VALIDATION.md](../data/NATIVE_VALIDATION.md) for source-build execution evidence.
[RELEASE_AUDIT.md](../data/RELEASE_AUDIT.md) records the real query/qrel and dense identifier-map audits;
[VALIDATION.md](../data/VALIDATION.md) preserves the earlier Windows-wheel pilot record.
[RESOURCE_VALIDATION.md](../data/RESOURCE_VALIDATION.md) records the verified first full snapshot and
isolated real-vector resource probes. PACE staging and job templates are described in
[scripts/pace/LONGEVAL.md](../../../../scripts/pace/LONGEVAL.md).

## Reproduce the pilot

From the repository root, in Python 3.11 with the native build prerequisites from CONTRIBUTING.md:

```sh
python -m pip install -r research/longeval/requirements.txt
python -m pip install ./engine
python -m pytest -q research/longeval/tests engine/python/tests/test_ir.py
python -m research.longeval fixture build/longeval-fixture
python -m research.longeval run --dataset build/longeval-fixture/dataset.json --output build/longeval-pilot --native-origin checkout-build
python -m research.longeval verify build/longeval-pilot --dataset build/longeval-fixture/dataset.json
```

Use a **new output directory** for every fixture generation and run. Existing outputs are never
overwritten. Failed/cancelled runs retain a manifest and stage-cost ledger and are rejected by the
verifier. There is no cache or in-place resume. Restart by replaying in a fresh process/directory.
Declare `--native-origin released-wheel` if using a released wheel instead of a local source build.
The binary hash and package version are recorded; the origin flag is a declaration, not attestation.

Set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before starting Python for
the recorded pilot environment. Both ANN implementations explicitly use one construction/search
thread. Exact-scoring BLAS settings are recorded in provenance. On restricted Windows environments,
give pytest a fresh workspace-local `--basetemp=build/longeval-tests-N`.

The default fixture contains three nested snapshots of 192/288/384 vectors, 8 development queries,
12 calibration queries, and 24 evaluation queries. Its complete artificial judgments come from
cluster membership. Nothing is downloaded by the runner. Seeds 11/29/47 are development choices,
not final research parameter selections.

## Implemented controls

- Float32 canonical vectors; content/vector hashes; unique string IDs; cumulative membership
  checks; rejection of removals, changed shared documents, changed vectors, or role leakage.
- Labels assigned in snapshot-arrival order, lexically within each new batch. Source-file row
  order does not determine insertion. Numeric-looking IDs remain strings.
- Explicit Epzero HNSW with compression, projection, half storage, refinement, guard, and adaptive
  stopping disabled. Searches override implicit reranking. No call to `epzero.build()`.
- Exact exhaustive NumPy scoring; incremental, same-order replay, and shuffled Epzero and FAISS
  HNSW; IVF-Flat trained initially and frozen versus freshly trained on each current snapshot.
- FAISS HNSW inserts one row at a time to avoid its internal batch shuffling. IVF uses all current
  rows for training, sorted by ID, with fixed centroid count, seed, and iteration count. Training
  IDs are retained. These are explicitly different adapters, not equal-work comparisons.
- Frozen search effort and held-out matched-recall calibration. Calibration exhausts the declared
  grid, tolerates nonmonotone recall, validates the selected setting, and reports unattainable
  targets. Evaluation-query recall is reported separately; target attainment is not guaranteed.
- Synchronous mutation/measurement barriers; repeated read-only search stability; live-set checks;
  Epzero incremental/replay topology and result equality. Native checkpoint continuation is disabled.
- Deterministic exact ties: descending score, then descending document ID. Native boundary-tie
  membership may differ. Strict and boundary-tolerant ANN recall are separate diagnostics; canonical
  score tolerance is `1e-5`, and never changes relevance ranking. Float32 inputs are accumulated in
  float64 by the independent scorer; FAISS Flat and Epzero exact comparisons have explicit tolerances.
- Linear-gain nDCG, relevance recall, judged fraction, and explicit/positive/judged populations.
  Missing runs and zero-IDCG queries score zero. The pilot uses all evaluation queries, not an
  asserted official LongEval population. Run scores use 17 significant digits.

The fixture uses k=10; its relevance recall is **R@10**, not R@100. Metric names include the actual
depth. Change the configuration and validate calibration at the desired depth before claiming
recall@100. The development queries are reserved but currently unused: fixture settings are fixed.

## Inputs and artifacts

The `fetch-release`, `audit-release`, and `inspect-dense` commands audit pinned real artifacts
separately from the synthetic runner. They do not turn metadata or judgment counts into retrieval
results. See RELEASE_AUDIT.md for their commands and verified scope. Full real vector ingestion is
implemented separately from the pilot; the published matrices change row order between snapshots.
See [INGESTION.md](../data/INGESTION.md) for the resumable `prepare-dense` job, its live status artifacts,
memory-mapped validation, and the remaining full-experiment limitations.

`dataset.json` records a release identity, synthetic flag, three snapshot entries, query-role
entries, and a representation contract (model, revision, preprocessing, normalization, dtype,
dimension, metric). Every table uses a JSONL file containing `{id, text}` records and a corresponding
float32 NPY matrix. Each snapshot has a four-column TREC qrels file. See the generated fixture for
an executable example. Cosine inputs must already be unit normalized; the loader does not silently
change vectors. All files must live within the manifest directory.

The loader deliberately keeps all snapshots in memory and rejects qrels containing unknown or
nonlive IDs. Real release eligibility and judgment policies must be audited before adapting this
contract. This fixture loader must not be used to load the full release. `mapped.py` supplies the
separate full-corpus reader; it does not silently convert a release into a pilot dataset.

Each completed output has:

- `manifest.json`: input/config/code/binary hashes, package/platform identity, declared protocol,
  run status, and artifact checksums.
- `labels.json`: reversible external-ID mapping.
- `states.json`: live-set/order hashes, ancestry for persistent arms, training IDs, native parameters,
  Epzero topology hashes, and FAISS serialized-state hashes. These fingerprints are not checkpoints.
- `*.run`: full-precision rankings for every seed/snapshot/arm/regime.
- `results.json`: per-query metrics, explicit cohort counts, strict and tie-aware ANN recall,
  calibration grid/selection, and target-miss flags.
- `costs.json`: allocation, training, building, insertion, calibration/oracle work, warmup, and
  three search-batch repetitions. Instrumentation, provenance hashing, and output writing are not
  all timed: this is a component development ledger, not total lifecycle accounting.

The verifier checks file hashes and can independently re-score exported relevance metrics against
the original inputs. Source hashes include local modifications; a Git commit alone is insufficient.
Exact results are recomputed after every membership change. Capacity is preallocated using final
fixture size (a disclosed retrospective resource choice), but future vectors are never used to train
an earlier index. Shuffled rebuilds are retrospective controls, not observed arrival chronology.

## Limits before research use

This pilot does not implement matched-cost selection, isolated per-index peak memory, request-level
latency distributions, uncertainty analysis, intervention accounting, or automatic recovery. Search
timings include Python adapter and validation work. RSS is sampled for the whole process while
multiple indexes and snapshots are resident; it is a lower bound, not an index-size measurement.
The synthetic dataset and small grid cannot establish LongEval relevance effects or performance.

The independent evaluator parity tests include zero-positive and missing topics. They validate the
declared formulas; official task compatibility still requires pinning and auditing the organizer's
actual evaluation population and rules. Native serialization is unchanged; continued construction
after loading a checkpoint remains unsafe for this protocol until its dedicated gate is passed.
