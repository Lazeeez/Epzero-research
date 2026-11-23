# E4: BGE-base-en-v1.5 vectors for the second-encoder replication

Part of the extension protocol (`research/longeval/docs/protocol/EXTENSION_PROTOCOL_V1.md`, decision 6).
The design, the pinned model revision and the predictions are in
`research/longeval/docs/protocol/E4_SECOND_ENCODER.md`, frozen on 23 November 2026, before any BGE
vector existed. This job only produces vectors; the held-out and arrival-order analyses on them are
separate jobs.

## What the job does

On one GPU node (at least 15 GB of GPU memory): checks the package; runs the encoder's unit tests;
builds a CUDA environment (PyTorch 2.9.0, the pinned `encoder-requirements.txt`); downloads the pinned
document archives and the pinned model (every file checked against its size and hash); encodes the
1,661,900 snapshot-3 documents (title, then abstract; CLS pooling, float32, L2-normalised, 512 tokens,
right truncation, the share truncated recorded) and the 381 test and 100 training queries (with BGE's
retrieval instruction); re-encodes a deterministic sample of 1,000 documents on the CPU one at a time,
which must agree to a cosine of at least 0.9999; copies the vectors and the report to the scratch build
folder.

## Submit

Build the package locally (`python scripts/pace/package-longeval-bge.py BUILD_DIR`), copy
`longeval-bge-01.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with
`longeval-bge-01-receipt.json`, then:

```bash
sha256sum longeval-bge-01.tar.gz
tar -xzf longeval-bge-01.tar.gz
(cd longeval-bge-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch longeval-bge-01/scripts/pace/longeval-bge.sbatch
```

## After completion

```bash
sacct -X --name=longeval-bge -S 2026-11-24 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-bge-*.log
tar -czf longeval-bge-01-evidence.tar.gz build/pace-longeval-bge/jobs longeval-bge-*.log
sha256sum longeval-bge-01-evidence.tar.gz
```

The archive holds document and query vectors derived from LongEval data (about 5 GB): copy it to the
local `build` directory and never commit it.
