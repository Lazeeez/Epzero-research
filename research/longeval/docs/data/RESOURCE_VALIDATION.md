# Real-vector preparation and resource checks — 2026-11-16

Snapshot 1 now has a verified complete archive, verified extracted payloads, and an all-row vector
audit. All 869,902 stored vectors have 2,560 components, are finite and nonzero, and have L2 norm
within `1e-5` of one. The observed norm range is 0.9999998679999677–1.0000001500543534.
No normalization was applied to the published vectors.

The vector payload SHA-256 is
`575dbdbb8c07a2e9aa14d87070a913d7e7b3f360e7a29dea5e9ab464e71db62b`.
The audit report is `build/longeval-snapshot1-audit/audit.json`, SHA-256
`aeef5ed4aadc2135c44847e27a2bd598502128c0b6cb598bf0f467898f247a64`.
The full process took 31.05 seconds and reached 280,892 KiB peak RSS, measured by `/usr/bin/time -v`.
Its raw resource record is `build/longeval-snapshot1-audit-resources.txt`.
This verifies one snapshot's numeric representation; the three-snapshot fixed-vector gate is pending.

## Isolated resource probes

Each backend ran in a fresh process on a deterministic subset of snapshot 1. Indexed IDs were
selected by SHA-256, then inserted lexically. Sixteen additional document vectors were held out
from the index and used as probes. They are not evaluation queries. No qrels were read and no
relevance results were produced. HNSW used M=16, efConstruction=100, one thread, and uncompressed
float32 vectors. These are engineering settings, not locked paper parameters.

| Indexed documents | Backend | Insertion seconds | Maximum sampled worker RSS, MiB |
|---:|---|---:|---:|
| 8,000 | Epzero HNSW | 4.29 | 316.29 |
| 8,000 | FAISS HNSW | 5.16 | 309.54 |
| 8,000 | FAISS IVF-Flat | 0.15 | 357.76 |
| 64,000 | Epzero HNSW | 51.55 | 1,438.79 |
| 64,000 | FAISS HNSW | 53.70 | 1,454.05 |

The IVF insertion time excludes its separately recorded 0.76-second training stage. That small
probe used 64 centroids and ten iterations. At nprobe=1 at least one query had fewer than 100
candidates; the operating point is marked infeasible, not padded. Scanning all 64 lists matched
the independent exact top-100 IDs for these probes.

RSS is sampled across the single worker, including its inputs, Python, and oracle scratch arrays.
It is not an index-only allocation measurement, nor a guaranteed capture of every brief peak.
Other downloads ran concurrently, so timings are engineering observations, not publication-quality
performance comparisons. The subset worker copies its selected corpus; it must not be extrapolated
unchanged into the 23.47 GiB WSL allocation for the final full corpus. PACE provides a prospective
way to measure larger isolated runs, subject to actual account and node availability.

All selected IDs, state fingerprints, exact/ANN runs, per-probe recall and component costs are in
`build/longeval-feasibility-{8000,64000}-{backend}/`. The consolidated record, including report
checksums, is `build/longeval-feasibility-summary.json`.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/longeval-linux-venv/bin/python -m research.longeval feasibility \
  ~/epzero-longeval/dense/snapshot-1 \
  --audit build/longeval-snapshot1-audit/snapshot-1/scan.json \
  --output build/NEW-RESOURCE-RUN --backend epzero --count 64000 \
  --native-origin checkout-build
```

## Document and encoder preparation

`documents.py` downloads the three original text ZIPs using identity-bound ranges and upstream
MD5 checksums from the SHA-256-pinned release manifest. Their combined size is 2,754,088,166 bytes.
The TU Wien transfer is substantially slower than the dense-index transfer. Its per-archive
`.download.json` records completed ranges. Windows metadata may lag while a WSL writer has a file
open; Linux-side sizes showed ongoing progress. An initial Windows transfer was stopped, and its
known 7 MiB prefix was explicitly bound to the same manifest identity before resuming in Linux.

The audit preserves the exact upstream `title + (' ' + abstract if abstract else '')` rule,
counts excluded empty texts, flags conflicting duplicate IDs, compares shared text hashes, and
can compare text membership to the separately audited dense ID maps. It streams ZIP members
without extracting paths; only ID/hash maps and a small sample remain in memory. Full text audits
have not completed. Commands:

```sh
python -m research.longeval fetch-documents build/longeval-release/source/downloads.json build/longeval-release/documents
python -m research.longeval audit-documents build/longeval-release/source/downloads.json build/longeval-release/documents --output build/NEW-TEXT-AUDIT --dense-maps build/longeval-release/dense
```

A **provisional** 32-document diagnostic was extracted from a downloaded compressed prefix:
`build/longeval-encoder-sample-provisional.jsonl`, with explicit unverified-source evidence in its
companion `.evidence.json`. It is the first 32 distinct nonempty documents in the first JSONL
member, selected without qrels. It must not be promoted into verified research data until the
complete source archive is checked. Maximum text length is 3,636 characters.

`encoder.py` pins Qwen3-Embedding-4B revision `5cf2132abc99cad020ac570b19d031efec650f2b`, Git blob
identities for ordinary configuration files, and full LFS SHA-256 identities for weights **and
the tokenizer JSON**. The tokenizer's Git pointer hash is not its payload hash; the initial
download correctly failed closed until that distinction was handled. The resumed model download
is logged in `build/longeval-encoder-download-resumed.log`; its process/location record is
`build/longeval-encoder-download-launch.json`. Model validation is incomplete until all files pass.

The separate CPU encoder environment is `~/epzero-longeval/encoder-venv`. Its initial
Windows-mounted counterpart was retained after an interrupted slow installation; do not use that
incomplete environment. Encoder diagnostics declare BF16 or float32, CPU or CUDA, last-token
pooling, float32 L2 normalization, and reject over-length inputs instead of silently truncating.
They use local files only, safetensors, and no remote model Python. GPU execution refuses an
unavailable device rather than falling back. The actual model forward pass and agreement with
archived document vectors have not yet been validated. No research query vectors have been made.

## Tests and remaining gates

The Linux research and IR suite passed **94 tests**. JUnit:
`build/longeval-continuation-linux-tests-01.xml`, SHA-256
`a3244de62d03fc9e22d986b0d2db340ba785a628c03d426f7ec4c5da774ae073`.
New tests cover source-text semantics, duplicates, row-order-independent diagnostic sampling,
identity-bound text download resume, model Git/LFS identities, diagnostic text changes, and
disjoint document probes. PACE scripts pass `bash -n`; they have not run on the cluster.

Still pending: snapshot-2/3 full vector comparisons, original text consistency, measured encoder
agreement, a locked query representation, full-data trajectory experiments, statistical analysis,
and a completed novelty review. An additional adjacent paper was identified:
[Wu and Yang, LongEval-Sci at CLEF 2026](https://arxiv.org/html/2607.04088v1).
It studies retrieval variants, temporal/citation overlays and qualitative refresh monitoring.
This motivates explicit differentiation of the proposed fixed-representation ANN-history study;
it is not evidence that the proposed study is novel.
