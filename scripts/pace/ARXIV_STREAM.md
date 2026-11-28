# Screen A1 on PACE: BGE vectors of arXiv abstracts in submission order

Design and predictions: `docs/screen-a1-arxiv-stream.md` (recorded before any vector existed). This job
only produces vectors; the builds and the analysis are separate.

## Prepare (locally)

```bash
python -I research/arxiv_stream/harvest.py HARVEST_DIR --delay 3       # resumable; about 7 hours
python -m research.arxiv_stream.prepare HARVEST_DIR PREPARED          # refuses an incomplete harvest
python scripts/pace/package-arxiv-stream.py PREPARED BUILD_DIR
```

## Submit (from the PACE LongEval checkout, which holds the CPU environment and uv)

Copy `arxiv-stream-01.tar.gz` there, compare its SHA-256 with `arxiv-stream-01-receipt.json`, then:

```bash
sha256sum arxiv-stream-01.tar.gz
tar -xzf arxiv-stream-01.tar.gz
(cd arxiv-stream-01 && sha256sum --check --quiet --strict SHA256SUMS && echo package OK)
sbatch arxiv-stream-01/scripts/pace/arxiv-bge.sbatch
```

One H200 node, 6-hour limit. The encoder is the LongEval BGE arm's, unchanged; it prints its rate every
minute and stops if a GPU run is slower than 200 abstracts a second. A sample of 1,000 abstracts is encoded
again on the CPU, one at a time, and must agree to a cosine of 0.9999 (the design asks for at least 100).

## After completion

```bash
sacct -X --name=arxiv-bge -S 2026-11-28 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' arxiv-bge-*.log
tar -czf arxiv-bge-01-evidence.tar.gz build/pace-arxiv-bge/jobs arxiv-bge-*.log
sha256sum arxiv-bge-01-evidence.tar.gz
```

The vectors (about 8.6 GB) are derived from CC0 metadata but stay outside the repository like every other
input; only aggregates are committed.

## Builds (after the encoding passed)

`arxiv-stream.sbatch` makes its own environment on the node (Epzero 0.6.0, FAISS 1.16.0 from PyPI) and runs the
package's tests first. One job writes the sample, with the exact top 100 of both query sets; then one job per
build, each on an exclusive Gold 6226 node (16-hour limit), reading a copy of the sample on node disk:

```bash
S=$(sbatch --parsable arxiv-stream-01/scripts/pace/arxiv-stream.sbatch sample ENCODING_JOB_ID)
for order in appended reversed shuffled; do for backend in epzero faiss_hnsw; do for seed in 11 29 47; do
  sbatch --dependency=afterok:$S arxiv-stream-01/scripts/pace/arxiv-stream.sbatch build $order $backend $seed
done; done; done
for seed in 11 29 47; do sbatch --dependency=afterok:$S arxiv-stream-01/scripts/pace/arxiv-stream.sbatch repaired $seed; done
```

That is 22 jobs: the sample, 18 order builds (the frozen order runner, `insertion_pilot.py run`, unchanged) and
3 repaired builds. Keep every job of the screen on Gold 6226 and do not overlap it with jobs on other CPU types.

```bash
sacct -X --name=arxiv-stream -S 2026-11-28 --format=JobID,State,ExitCode,Elapsed
grep -H '^PASS\|^exit_code' arxiv-stream-*.log
tar -czf arxiv-stream-01-evidence.tar.gz build/pace-arxiv-stream/jobs arxiv-stream-*.log
sha256sum arxiv-stream-01-evidence.tar.gz
```

The evidence holds reports, per-query tables and returned labels (no vectors).

## Package built (28 November 2026)

`arxiv-stream-01.tar.gz` in the local `build` directory: 1,224,781,297 bytes, 110 files, SHA-256
`c674fab87456dc38924bb20331f8b3a03f4fd262ec94f39c76cf03d9cd803f6b` (receipt `arxiv-stream-01-receipt.json`); corpus
3,259,971 abstracts, 5,000 + 5,000 queries. Checked locally: every package checksum verifies after unpacking; the
encoder's and preparation's tests pass in the encoder environment; the stream's tests pass in a clean environment
holding exactly the build job's packages (Epzero 0.6.0, FAISS 1.16.0, numpy 2.4.6, pytest 8.3.5).
