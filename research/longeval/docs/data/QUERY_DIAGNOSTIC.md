# Training-query precision sensitivity

This diagnostic follows the representative document result: GPU float32 produced higher cosine
agreement with released document vectors than GPU BF16 for all 32 sampled documents. It does
not establish query compatibility or approve a primary research representation. No acceptance
margin is set from those observed results.

`query_diagnostic.py` reads only the pinned training-query TSV: 100 queries. It checks the release
manifest and upstream MD5, and the cached baseline SHA-256 at commit
`a488f50f2132f38c07631f484eec0b67f3a18926`. It does not read qrels or test-query contents.
The baseline file is retained as evidence and never executed. Original query text is preserved;
the exact prefix is:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:
```

The query text follows the colon immediately, with no inserted space. The existing deterministic
SHA-256 split assigns 50 training queries to development and 50 to calibration. Those role IDs
are verified on every load. This engineering diagnostic uses both training roles and reports
ranking sensitivity separately for each; it does not optimize ANN settings or select test cohorts.

The query wrapper uses the existing verified text encoder on already-prompted records. Its nested
`encoder/encoding.json` is preserved unmodified; the outer `query-encoding.json` declares the
actual training-query semantics, prompt, roles, source/input hashes and diagnostic-only status.
The wrapper verifies output identities and norms. Comparing precisions also requires identical
model files, packages, device, GPU, token counts, pooling, normalization and nonprecision settings,
including disabled TF32. Failed jobs keep explicit failed status and cannot be loaded as completed.

Prepare locally:

```bash
python -m research.longeval.query_diagnostic prepare \
  build/longeval-release/files build/longeval-release/source/baseline.py \
  build/longeval-training-query-inputs-01
```

After transferring the versioned small query bundle, submit directly on PACE:

```bash
sbatch scripts/pace/longeval-query-sensitivity.sbatch
```

Large downloads, model files, packages, environments and caches use the compute-node temporary
directory through the tested local-storage helper. No login setup is needed. Only small inputs,
diagnostic vectors and reports are retained in shared storage. Results live at
`build/pace-longeval-queries/jobs/JOB_ID/`. The Slurm log is `longeval-queries-JOB_ID.log`.
The job produces 100 BF16 and 100 float32 query vectors; it does not download document vectors
to PACE or perform the exact corpus scan there.

After retrieving and verifying that result directory, run the bounded-memory exact comparison
locally against one already verified snapshot at a time:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m research.longeval.query_sensitivity \
  ~/epzero-longeval/dense/snapshot-1 \
  build/longeval-snapshot1-audit/snapshot-1/scan.json \
  RETURNED_JOB/inputs RETURNED_JOB/bfloat16 RETURNED_JOB/float32 \
  build/NEW-TRAINING-QUERY-SENSITIVITY --snapshot 1
```

The runner verifies the full document vector-file identity against the scan and ingestion receipts.
It scores both query precisions in one corpus pass with float64 accumulation over original float32
values, retaining top 100 with deterministic descending-score/descending-ID tie handling.
Outputs report top-1 agreement, ordered-list equality and top-10/top-100 set overlap per query and
training role. No relevance judgments are read and no ANN-history or relevance-effect claim follows
from this diagnostic. Corpus vectors remain unchanged. Device/runtime must remain fixed within
future research trajectories; query-vector similarity alone is not a ranking-stability guarantee.
