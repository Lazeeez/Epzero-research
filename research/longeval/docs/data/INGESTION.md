# Full dense preparation follow-up — 2026-11-16

The increased WSL allocation removes the earlier RAM-size obstacle to preparing individual
snapshots. WSL reports 23.47 GiB total RAM and about 22.79 GiB available during downloading.
The full preparation job is **running**, not yet evidence that the real vector gate passes.
There are still no real-query retrieval results or publication claims from this stage.

## Active job and artifacts

The job downloads all three revision-pinned dense archives, verifies their full SHA-256 hashes,
stream-extracts the permitted members, and then audits every stored vector. Downloaded archives
and extracted matrices live on the Linux filesystem at `~/epzero-longeval/dense`.
The three archives total 39,385,762,192 bytes; vector payloads total 39,470,305,280 bytes.
Allow roughly 80 GB for archives plus extracted contents, with additional disk safety margins.

Workspace artifacts:

- `build/longeval-preparation-launch.json`: command, Linux process ID, output and log locations.
- `build/longeval-preparation-01.log`: download byte counts, extraction stages, and scan results.
- `build/longeval-preparation-01/job.json`: current stage, source hashes, and completion/failure state.
- `build/longeval-preparation-01/vectors/audit.json`: created once vector auditing starts; inspect
  `status` and `fixed_vector_gate_passed` separately. A completed audit can fail the scientific gate.
- `build/longeval-preparation-resource-check.json`: timestamped RAM, disk, process, and partial-file
  observations. These are observations, not isolated index-memory measurements.

At the recorded resource check the first archive had reached 5,333,057,536 bytes and the downloader
used 38.68 MiB RSS. Downloading was deliberately interrupted once to attach the automatic audit;
the replacement job resumed from the retained partial file. Live progress supersedes this number.

Reproduce from the repository root using the Linux environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  build/longeval-linux-venv/bin/python -m research.longeval prepare-dense \
  ~/epzero-longeval/dense --output build/longeval-preparation-NEW
```

Do not start a concurrent writer against the active data directory. A retry after the job exits
uses the same download directory and a fresh output directory. Complete cached archives and
extractions are rechecked. Interrupted extraction retains a `.staging` directory for diagnosis;
extraction itself does not resume in place. Inspect and preserve that directory under a distinct
name before retrying extraction. Download interruption is resumable without such a step.

Individual stages are also available:

```sh
python -m research.longeval ingest-dense DATA_DIR --snapshots 1 2 3
python -m research.longeval audit-vectors DATA_DIR --output NEW_AUDIT_DIR
```

## Implemented correctness and memory controls

`storage.py` binds partial downloads to the pinned URL, size, and SHA-256. Range responses must
match the exact requested byte interval. A full checksum is mandatory before promoting an
archive. Extraction permits only the expected regular-file members, checks sizes, records each
member's hash, and promotes the staging directory only after completing the entire archive.

`mapped.py` reads little-endian float32 vectors without normalization or full-matrix copies.
It checks metadata/identifier hashes, shape and unique IDs, hashes every vector byte and every
row, measures finiteness and float64 norms, and compares consecutive snapshots through external
document IDs. Reordered rows are allowed; changed shared vectors and removed IDs fail the fixed
representation gate. Zero/nonfinite vectors also fail the current numeric gate. The scan reports
the unit-norm count separately; it does not infer encoder compatibility from norms. Source-text
consistency is not established by vector equality. Scratch arrays scale with 2,048 rows rather
than corpus size; Linux mmap pages are advised discardable after each chunk. Actual full-scan
memory measurements remain pending. A one-snapshot audit cannot pass the full three-snapshot gate.

`oracle.py` provides an exhaustive depth-100 reference using document blocks and query tiles.
It accumulates in float64, retains only top-k candidates, and resolves exact ties by descending
external ID. L2 uses direct subtraction; cosine requires unit vectors. Matrix multiplication can
differ by rounding from the scalar-query scorer, so parity tests include numerical tolerances.
The scalar pilot oracle remains available as an independent implementation.

`backends.py` now inserts bounded chunks without changing insertion order and streams native
graph/FAISS serialized fingerprints. This removes large temporary vector and serialized-index
copies. It does **not** make the fixture runner suitable for the complete corpus: it still holds
all snapshots and several live arms, and IVF training still materializes the full training matrix.
An isolated full-data runner and an explicit training-sample contract remain necessary.

## Validation evidence

- Linux research plus IR suite: **85 passed**. JUnit: `build/longeval-linux-ingestion-tests-02.xml`,
  SHA-256 `e36a567577408ac2f4cb71eba9f1ae1544fcc68f58480922944ce39c63ee7400`.
- New Windows component tests: 13 ingestion/mapped tests, 6 oracle tests, and 4 backend-memory
  tests passed. Windows uses the previously documented released native wheel.
- Source-built synthetic regression: `build/longeval-streamed-pilot` verified all 157 artifacts.
  All 153 run files and `states.json` match `build/longeval-native-pilot` byte for byte.
  Manifest SHA-256 `0d7cb3cc024de3eeee55d0bf0196f77f9f245c34d5805effe6a7aa16c1b9d0c1`.
- Download tests exercise exact-offset resume, corrupt data, unowned partial files and rejected
  range responses. Extraction tests exercise path rejection, corruption and retained failure state.
  Vector tests exercise nonfinite rows, checksum mismatches, reordered IDs, changed shared vectors,
  removals, and incomplete gate coverage. Oracle tests include FAISS Flat parity and boundary ties.
- Adapter tests show insertion chunk size preserves ranking and index state for Epzero HNSW,
  FAISS HNSW and FAISS IVF-Flat; streaming fingerprints match the earlier serialized bytes.

The synthetic regression ran while data downloaded. Its timing is not a benchmark. The C++ engine
was unchanged; the earlier 115-test source-build evidence is retained in NATIVE_VALIDATION.md.

## Encoder provenance follow-up

The published snapshot-2 prefix contains `index-ir-metadata.yml`. Its source record names commit
`06649a2e03f66f30a52e7f01c353171239a3d438`, with both uncommitted changes and unchecked files.
The complete public Git tree at that commit has no dense baseline source. Its recorded archive
path also refers to `snapshot-3`, despite being carried in snapshot 2. This metadata therefore
does not establish the exact source or server configuration that produced snapshot 2.
The metadata and Git-tree response are cached under `build/longeval-release/source/`.

The [Qwen model metadata](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-4B) identifies
candidate revision `5cf2132abc99cad020ac570b19d031efec650f2b`. Its two BF16 safetensors shards total
8,043,592,088 bytes, exceeding the local GPU's 6,141 MiB VRAM before runtime allocations. CPU or
GPU/CPU placement needs measurement; quantization must not be silently substituted.
Small configuration files are cached; model weights have not been downloaded in this step.

At that revision, the [query prompt configuration](https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/5cf2132abc99cad020ac570b19d031efec650f2b/config_sentence_transformers.json)
matches the pinned baseline instruction exactly, including no space after `Query:`. The
[module configuration](https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/5cf2132abc99cad020ac570b19d031efec650f2b/modules.json)
and [pooling configuration](https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/5cf2132abc99cad020ac570b19d031efec650f2b/1_Pooling/config.json)
specify last-token pooling followed by normalization for Sentence Transformers. These are useful
compatibility constraints, not proof of the historical vLLM server's behavior. Tokenization,
truncation, numerical precision, runtime revision, document preprocessing, and measured agreement
with archived document vectors still need to be pinned before generating research query vectors.
