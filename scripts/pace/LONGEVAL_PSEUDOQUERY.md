# Pseudo-query encoding job (stage 1 of PSEUDO_QUERY_PLAN.md)

One A100 GPU job that encodes title pseudo-queries with the pinned Qwen3-Embedding-4B
model in FP32 with TF32 disabled, using the real-query instruction prompt.

- Set A: 16,378 texts for the 8,192 sample documents (`title` and `keywords4` forms).
- Set B: 131,040 `keywords4` texts for hash-selected snapshot-3 documents.

Before any pseudo-query is encoded, the 100 training queries are re-encoded through the
batched path. Every cosine against the verified job-7107422 FP32 vectors must be at
least 0.99999, or the job stops. This is an input-generation job; it measures no
retrieval and touches no held-out query outcome.

## Submit

Copy `build/longeval-pseudoquery-02.tar.gz` to the isolated PACE Epzero checkout (the
one containing `build/pace-longeval/venv` and `build/pace-longeval/tools/bin/uv`).
Compare the SHA-256 with `build/longeval-pseudoquery-02-receipt.json`, then:

```bash
sha256sum longeval-pseudoquery-02.tar.gz
tar -xzf longeval-pseudoquery-02.tar.gz
(cd longeval-pseudoquery-02 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch longeval-pseudoquery-02/scripts/pace/longeval-pseudoquery.sbatch
# Or choose a free GPU type: sbatch --gres=gpu:v100:1 longeval-pseudoquery-02/scripts/pace/longeval-pseudoquery.sbatch
```

Resources: one NVIDIA GPU with at least 20 GiB (A100, A40, H100, L40S, RTX 6000 or V100-32GB), 4 CPUs, 40 GB RAM, 80 GB node-local temporary
disk, 4-hour ceiling. The torch install and the 8 GB model download go to the
compute node's temporary directory only; there is no shared-storage fallback.
Expected time: about 20–30 minutes of setup, then roughly 1 hour of encoding
(job 7107422 measured about 32 ms per query at batch size 1 in FP32).

## After completion

```bash
job=JOB_ID
sacct -X -j "$job" --format=JobID,JobName%32,State,ExitCode,Elapsed
tail -n 30 "longeval-pseudoq-${job}.log"
tar -czf "longeval-pseudoq-${job}-evidence.tar.gz" \
  -C build/pace-longeval-pseudoquery/jobs "$job" -C "$PWD" "longeval-pseudoq-${job}.log"
```

The archive is about 1.5 GB, mostly the Set B vectors. Copy it into the local `build`
directory. If transfer size is a problem, Set A alone is enough for the stage-2 sample
gates: leave out `$job/encoding/vectors-B.npy` and keep it on scratch for the later
full-data screen. Keep failure evidence. These vectors are derived from LongEval data
and must never be committed to the public repository.
