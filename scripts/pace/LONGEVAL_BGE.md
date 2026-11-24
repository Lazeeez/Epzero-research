# E4: BGE-base-en-v1.5 vectors for the second-encoder replication

Part of the extension protocol (`research/longeval/docs/protocol/EXTENSION_PROTOCOL_V1.md`, decision 6).
The design, the pinned model revision and the predictions are in
`research/longeval/docs/protocol/E4_SECOND_ENCODER.md`, frozen on 23 November 2026, before any BGE
vector existed. This job only produces vectors; the held-out and arrival-order analyses on them are
separate jobs.

## What the job does

On one NVIDIA H200 node (packages -02 and -03; package -01 asked for any GPU, and job 7254797 landed on a node without NVIDIA tools): checks the package; runs the encoder's unit tests;
builds a CUDA environment (PyTorch 2.9.0, the pinned `encoder-requirements.txt`); downloads the pinned
document archives and the pinned model (every file checked against its size and hash); encodes the
1,661,900 snapshot-3 documents (title, then abstract; CLS pooling, float32, L2-normalised, 512 tokens,
right truncation, the share truncated recorded) and the 381 test and 100 training queries (with BGE's
retrieval instruction); re-encodes a deterministic sample of 1,000 documents on the CPU one at a time,
which must agree to a cosine of at least 0.9999; copies the vectors and the report to the scratch build
folder.

## Job 7254809 (package -02): time limit in the encode phase

The job passed every phase up to `encode` (package, GPU, environment, tests, CUDA environment, the
three document archives, the model), then reached its 4-hour limit in `encode` without printing a
single progress line. The model-loading warning in the log shows that the texts were read and the
model loaded. The saved encoder report has no `tokens` field, so the job never got past the token
count, which takes about 3 minutes for 1.66M texts with this tokenizer (measured locally), or the
first 50,000 documents. The job used 32 CPU-hours in 4 hours, all 8 cores busy, as a CPU-bound model
would be. On an H200 the encoding should take minutes. The encode step was the only GPU work started
with `srun`, which can leave a step without the GPU when the step does not ask for it.

Package -03 (24 November 2026; the encoder's outputs and settings are unchanged):

- the encoder runs in the batch step, without `srun`, where the job's GPU check ran;
- it saves its report after reading the texts, after loading the model (with the model's device), and
  after counting tokens;
- it prints its rate every minute and stops if a GPU run is slower than 200 documents per second after
  5,000 documents;
- GPU use is logged every minute (`gpu-usage.csv`), and the time limit is 6 hours.

## Submit

Build the package locally (`python scripts/pace/package-longeval-bge.py BUILD_DIR`), copy
`longeval-bge-03.tar.gz` to the PACE LongEval checkout, compare its SHA-256 with
`longeval-bge-03-receipt.json`, then:

```bash
sha256sum longeval-bge-03.tar.gz
tar -xzf longeval-bge-03.tar.gz
(cd longeval-bge-03 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch longeval-bge-03/scripts/pace/longeval-bge.sbatch
```

## After completion

```bash
sacct -X --name=longeval-bge -S 2026-11-24 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' longeval-bge-*.log
tar -czf longeval-bge-03-evidence.tar.gz build/pace-longeval-bge/jobs longeval-bge-*.log
sha256sum longeval-bge-03-evidence.tar.gz
```

The archive holds document and query vectors derived from LongEval data (about 5 GB): copy it to the
local `build` directory and never commit it.
