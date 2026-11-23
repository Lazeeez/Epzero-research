# E4: second encoder — design and predictions

Recorded on 23 November 2026, before any BGE vector, run or result exists. Part of the extension
protocol (`EXTENSION_PROTOCOL_V1.md`, decision 6). Decided by the author; the advisor was
unavailable.

## Model, pinned

| Item | Value |
| --- | --- |
| Model | `BAAI/bge-base-en-v1.5` (BERT-base, 768 dimensions) |
| Revision | `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` (Hugging Face, last modified 2024-02-21) |
| Weights | `model.safetensors`, 437,955,512 bytes, LFS SHA-256 `c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7` |
| `config.json` | 777 bytes, Git blob `09a8200d8a549a3b9dba53b67156ca26810cc720` |
| `tokenizer.json` | 711,396 bytes, Git blob `688882a79f44442ddc1f60d70334a7ff5df0fb47` |
| `tokenizer_config.json` | 366 bytes, Git blob `37fca74771bc76a8e01178ce3a6055a0995f8093` |
| `vocab.txt` | 231,508 bytes, Git blob `fb140275c155a9c7c5a3b3e0e77a9e839594a938` |
| `special_tokens_map.json` | 125 bytes, Git blob `a8b3208c2884c4efb86e49300fdd3dc877220cdf` |
| `1_Pooling/config.json` | 190 bytes, Git blob `da5bfd57e34ca45582e4bdbaa3e6deb9efffa08d` (CLS pooling) |
| `sentence_bert_config.json` | 52 bytes, Git blob `ea85692bff64b0d1917833c31ddbca8ab10f5455` (max_seq_length 512, lower case) |
| `modules.json` | 349 bytes, Git blob `952a9b81c0bfd99800fabf352f69c7ccd46c5e43` |

The encoder module verifies every size and hash before loading, as `encoder.py` does for Qwen, and
loads no remote Python code.

## Encoding

- **Documents.** Text rule as in `documents.py`: `title + (' ' + abstract if abstract else '')`,
  per snapshot, from the pinned document archives. No instruction. A document whose text is
  identical across snapshots (same SHA-256) gets one vector.
- **Queries.** The 381 test queries and the 50 calibration queries, each prefixed with BGE's
  published retrieval instruction `Represent this sentence for searching relevant passages: `.
- **Model call.** The BERT encoder in float32, CLS pooling, L2 normalization, inner product.
  Maximum 512 tokens, truncating on the right. The number and share of documents truncated, and
  their token counts, are recorded per snapshot.
- **Verification.** A deterministic sample of 1,000 documents is encoded a second time on a
  different device or thread count; every pair of vectors must have cosine similarity ≥ 0.9999.
  The hardware, library versions and the vectors' SHA-256 are recorded.

## Design

The held-out design of `FINAL_PROTOCOL_V1`, unchanged apart from the vectors: the same queries,
374-query panel, arms (Epzero HNSW, FAISS HNSW, IVF with frozen training), construction seeds,
calibration rule, labels and estimators, with exact references recomputed on the BGE vectors. Then
the arrival-order experiment of E2 (as arrived, shuffled, reversed blocks; three seeds; M 16,
efConstruction 100; ef 100–800) on the full snapshot 3 encoded with BGE.

## Predictions, from the Qwen results, fixed now

1. **Arrival reversal replicates.** For both HNSW implementations at M 16 and ef 200, snapshot-3
   against snapshot-1 exact top-100 neighbours: within-query ratio as arrived above 1.10, shuffled
   within [0.90, 1.10], reversed below 0.95; the paired change in the newest cohort's absolute
   excess miss risk (as arrived minus reversed) above +1 percentage point with its interval
   excluding 0.
2. **Mean recall barely moves with order.** Overall recall@100 differs by less than 1 point
   between the three orders, for each implementation.
3. **IVF is much weaker.** Frozen IVF's neighbour-level excess for snapshots 2+3 against 1 is below
   1 percentage point.
4. **D stays small.** FAISS HNSW and IVF: D within ±0.01 (negligible under the frozen rule).
   Epzero: |D| below 0.006.
5. **Absolute miss rates differ from Qwen** (BGE is a smaller model with 768 dimensions); no
   prediction is made about their level.

A result against any prediction is reported as such. Prediction 1 is the one the paper's claim rests
on: if it fails, the arrival effect is reported as specific to the Qwen representation.
