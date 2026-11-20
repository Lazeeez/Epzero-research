# PACE NGFix/RFix integration screen

This is one job to validate a direct published competitor. It is not a new-method
result, a full-corpus benchmark, or a fair-training-budget comparison.

Copy `build/longeval-ngfix-screen-02.tar.gz` into the existing isolated PACE Epzero
checkout. Verify its SHA-256 against the local companion receipt
`build/longeval-ngfix-screen-02-receipt.json`, then run from that checkout:

```bash
sha256sum longeval-ngfix-screen-02.tar.gz
tar -xzf longeval-ngfix-screen-02.tar.gz
PYTHONPATH="$PWD/longeval-ngfix-screen-02" build/pace-longeval/venv/bin/python \
  -m research.longeval.ngfix_screen package-verify "$PWD/longeval-ngfix-screen-02"
sbatch longeval-ngfix-screen-02/scripts/pace/longeval-ngfix.sbatch
```

The supplement has its own directory and manifest; it does not overwrite the
original source, existing experiment supplements, or the shared Python environment.
The package contains pinned upstream source, a compiler-derived Boost header
closure, adapter/tests, and small query/reference files. It contains no dataset
vectors or native indexes. GCC13.3.0 is required, matching the local compile check.
Version01 failed the isolated compiler check because its header inventory omitted
Boost headers marked as system headers. Use only version02.

The job requests one exclusive Gold6226 node, one computation thread, 8GB RAM,
48GB node temporary storage and a two-hour ceiling. It verifies the package and
CPU features, compiles the original AVX-512 kernels, runs verifier tests and three
native rejection tests, then verifies a 512-vector synthetic run. The synthetic
run must recover every exact neighbor at exhaustive search width before proceeding.

Only after those checks does it download the pinned **snapshot-1 archive (~8.89GB)**
and extract it on the compute node's temporary disk. It reconstructs the existing
8,192-vector sample and requires the exact previously recorded binary hash.
The archive, extracted corpus, matrices, all graph states and compiler binary stay
on temporary disk. There is no shared/scratch fallback. No model download is needed.

Four controls run on the same base graph: base, NGFix, RFix, and both. Each has
100 queries, four ef settings, one warmup and three timings per condition. Fitting
uses only 50 calibration queries; no held-out test queries are present. Independent
verification checks 160,000 returned scores, all 4,800 timing records, calibration
selection, and before/after graph hashes. The returned bundle preserves graph hashes
and sizes, not the large graph binaries; local rechecking cannot rehash those binaries.

After completion, replace `JOB_ID` below with the submitted ID:

```bash
job=JOB_ID
sacct -X -j "$job" --format=JobID,JobName%32,State,ExitCode,Elapsed
tail -n 45 "longeval-ngfix-${job}.log"
tar -czf "longeval-ngfix-${job}-evidence.tar.gz" \
  -C build/pace-longeval-ngfix/jobs "$job" \
  -C "$PWD" "longeval-ngfix-${job}.log"
```

Copy that archive to the local `build` directory. Keep failed-job evidence too;
an exit failure should not be worked around by disabling correctness checks.
The final log must report both synthetic and sample verification. A scheduler
`COMPLETED` status alone is insufficient.

Local rechecking uses the saved original `build/longeval-ngfix-inputs-01` matrices:

```bash
python -m research.longeval.ngfix_screen verify \
  build/longeval-ngfix-inputs-01 PATH_TO_RETURNED_JOB/sample --returned
```

Before interpretation, also check the returned package manifest, CPU/compiler,
binary identities, fixture verification, native rejection tests, and job exit status.
The job's native C++ timing boundary differs from Python binding timings; do not
claim a cross-backend speedup from comparing these files with earlier WSL results.
