# Training-query diagnostic preparation

PACE query encoding in job 7107422 has completed, and the returned artifacts have been
independently verified locally. Exact ranking comparisons have completed across all three real
snapshots and have been independently recounted from the saved run files. See
[QUERY_SENSITIVITY_RESULTS.md](QUERY_SENSITIVITY_RESULTS.md) for the results and limits.

- Pinned 100-query training input bundle: `build/longeval-training-query-inputs-01/`.
- Deterministic role split: 50 development and 50 calibration queries; no qrels or test-query
  contents are read by this stage.
- Offline tokenizer preflight: 21–35 tokens per prompted query, 2,484 tokens total; no truncation.
- Exact baseline query prefix is preserved, including no added space after `Query:`.
- Twenty-one focused query, exhaustive-oracle and representation tests passed; JUnit is
  `build/longeval-query-tests-02.xml`. The end-to-end fixture covers sample encoding receipts,
  both query precisions, bounded exhaustive search, ranking comparison, and changed-vector rejection.
- The PACE script passes Bash syntax validation. Original 127 source files and all 21 supplemental
  entries verify after clean extraction of the source, bootstrap and query bundles.

Transfer bundle: `build/longeval-pace-query-01.tar.gz`, 245,984 bytes, SHA-256
`04a17bf781d98ed855a837ec1bf48c70da106410459a90567484b18daf3a0c66`.
Submit `scripts/pace/longeval-query-sensitivity.sbatch` directly after extraction. Model/package
downloads remain entirely on compute-node temporary disk. The job writes only small diagnostic
outputs and receipts to `build/pace-longeval-queries/jobs/JOB_ID/`.

After retrieval and independent verification, use `research.longeval.query_sensitivity` against
the locally audited document matrices to measure exact top-100 ranking changes. No final query
representation, relevance result, ANN-history effect, or acceptance threshold is claimed by
preparation or synthetic tests. See [QUERY_DIAGNOSTIC.md](QUERY_DIAGNOSTIC.md) for the protocol.

## Reported PACE completion

The supplied tail reports the final training-query `PASS` message, verification of 127 original
and 21 supplemental source/input files, and completion in 3 minutes 58 seconds. Job 7107422 ran
on a single compute node in `coc-gpu`, account `coc`, QOS `coc-ice`, with four CPUs, one A100
and 40 GB host RAM. The epilog reports memory `16318628K`, a whole-job scheduler measurement.

The shown comparison fragment reports maximum BF16-versus-float32 cosine similarity
`0.9998158639659898`, zero bitwise-identical rows, and maximum coordinate difference
`0.003497665748000145`. The minimum, median and per-query values were not included in the
pasted fragment. Do not infer ranking agreement from these vector-level measurements.

Retrieve `build/pace-longeval-queries/jobs/7107422/` and `longeval-queries-7107422.log`.
These contain only small inputs, query vectors, logs and receipts. Large downloads remain in
compute-node temporary storage. Next: verify returned identities and query-pair contracts,
then run bounded exhaustive precision comparisons against the locally audited snapshots.

## Returned artifact verification and exact execution

Archive `build/longeval-queries-7107422-evidence.tar.gz` has SHA-256
`a237742eeb9e8b190e91cc647a3cc3766630ba251e3e16013af56c625ec8763c`.
It was safely extracted to `build/longeval-pace-query-evidence-7107422/`. The exit code is zero;
the manifest matches the submitted bundle; all input/source identities, query roles, model
identities and precision-pair contracts verify. Local recomputation reproduces all 100 per-query
vector comparisons within `1e-14` absolute tolerance. The recorded download stage is local XFS
at `/tmp/longeval-7107422.XfRKWl`.

BF16-versus-float32 query cosine similarity: minimum `0.9995510899401738`, median
`0.9997488872694831`, maximum `0.9998158639659898`. No query vector pair is bitwise identical.
The verification receipt is `build/longeval-pace-query-evidence-7107422/local-verification.json`.

The exact comparisons completed under `build/longeval-query-sensitivity-01/`, one fresh process per
snapshot, one BLAS thread, with full vector-file identity checks before scoring. `job.json`
records progress and report hashes; each snapshot has its own results and progress report.
`/usr/bin/time` resource measurements are retained, but these concurrent workstation runs are
not isolated latency measurements. No relevance judgments are loaded by this stage.
