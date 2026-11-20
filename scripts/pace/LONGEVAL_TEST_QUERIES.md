# Held-out test-query encoding job

Step 1 of `research/longeval/docs/protocol/FINAL_PROTOCOL_V1_CANDIDATE.md`. One GPU job encodes the 381
released test queries with the pinned Qwen3-Embedding-4B model, using the same batch-1
CUDA FP32, TF32-off path as the training queries (job 7107422). It produces query vectors
only; it runs no search and reads no judgments, so the protocol can still be frozen afterwards.

Before encoding test queries, the job re-encodes the 100 training queries. Every cosine
against the verified job-7107422 vectors must be at least 0.99999, or the job stops.

## Submit

Copy `build/longeval-test-queries-03.tar.gz` to the PACE LongEval checkout, compare its
SHA-256 with `build/longeval-test-queries-03-receipt.json`, then:

```bash
sha256sum longeval-test-queries-03.tar.gz
tar -xzf longeval-test-queries-03.tar.gz
(cd longeval-test-queries-03 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch longeval-test-queries-03/scripts/pace/longeval-test-queries.sbatch
# Another GPU type with >= 20 GiB: sbatch --gres=gpu:a100:1 longeval-test-queries-03/...
```

Resources: one NVIDIA GPU with at least 20 GiB (A40 by default), 4 CPUs, 40 GB RAM, 80 GB
node-local temporary disk, 2-hour ceiling; expect about 30 minutes, mostly setup. The torch
install and the 8 GB model download stay on the node's temporary disk.

## After completion

```bash
job=JOB_ID
sacct -X -j "$job" --format=JobID,JobName%32,State,ExitCode,Elapsed
tail -n 20 "longeval-testq-${job}.log"
tar -czf "longeval-testq-${job}-evidence.tar.gz" \
  -C build/pace-longeval-test-queries/jobs "$job" -C "$PWD" "longeval-testq-${job}.log"
```

The archive is about 6 MB. Copy it to the local `build` directory. Test-query vectors are
derived from LongEval data and must never be committed to the public repository.
