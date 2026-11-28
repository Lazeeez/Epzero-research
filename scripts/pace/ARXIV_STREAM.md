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
