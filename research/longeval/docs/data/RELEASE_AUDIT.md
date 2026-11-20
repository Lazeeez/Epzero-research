# LongEval-Sci release audit — 2026-11-16

This audit uses real released queries, judgments, and dense-index identifier maps. It does **not**
contain retrieval results or verify the complete vector matrices.

## Pinned inputs

The query/qrel manifest is from the [integration revision
63e73bdf](https://github.com/clef-longeval/ir-datasets-longeval/tree/63e73bdf54138ffed4246ce679627d7b50bd623a).
The actual path is `ir_datasets_longeval/etc/downloads.json`. Its SHA-256 is
`25d669a013a9db9dfc084a4c0468a12567e1c0d09077b150ff58447019eaa85e`.
All 12 selected release files passed the upstream MD5 checks; local SHA-256 values are also retained.
Only exact `umbrella_zeroshot_no_desc_no_narrative` LLM members were selected from the archive.

The [organizer-linked dense collection](https://huggingface.co/collections/jueri/longeval-2026-baseline-dense)
contains repositories ending in **`-index-dense`**. The baseline README's shorter `-index` example
points to a different, sparse artifact. Even the dense artifact cards contain sparse/Terrier tags;
the internal `pt_meta.json` identifies the inspected artifacts as dense FlexIndex. Select artifacts
by pinned identity and validated internal structure, not card tags or the abbreviated example.

| Snapshot | Dense repository revision | Stored rows | Dimensions | Float32 vector GiB |
|---|---|---:|---:|---:|
| 1 | `c6a54e3969887357ed2828714281923260bb3864` | 869,902 | 2,560 | 8.296 |
| 2 | `f75720b44e4bea83c5795ffbe615184e68d6e64b` | 1,322,720 | 2,560 | 12.614 |
| 3 | `b742f1709ad59361454432b2db9d86726ed1b1f4` | 1,661,900 | 2,560 | 15.849 |

These counts were checked by decoding the complete `docnos.npids` members, not just by trusting
metadata. The sidecar's declared `vecs.f4` size agrees with `4 * rows * dimensions` in each case.

Only 42,074,249 compressed prefix bytes were retrieved across the three archives. HTTP responses
had to honor exact byte ranges; the implementation never falls back to a full download. Sidecar
SHA-256 values are verified. Prefixes have recorded local hashes, but **the full archive SHA-256
values have not been verified**. Full compressed archives total approximately 39.39 GB decimal.

## Membership findings

| Snapshot | Distinct stored IDs | New IDs | Removed IDs | Duplicate rows |
|---|---:|---:|---:|---:|
| 1 | 869,902 | 869,902 | 0 | 0 |
| 2 | 1,322,720 | 452,818 | 0 | 0 |
| 3 | 1,661,900 | 339,180 | 0 | 0 |

The stored live sets are cumulative. However, neither later matrix preserves the earlier matrix
as a row prefix. Import and shared-vector comparison must join on document ID. Copying the first
N rows or assuming stable native row labels across archives would be incorrect.

Every document ID referenced by the audited raw, DCTR, selected LLM, and available human judgments
is present in its corresponding dense index. The snapshot-1 training qrels also have complete ID
coverage. This establishes judgment coverage in the published indexes, not complete eligibility or
text consistency of the original document releases.

## Queries and judgment populations

The query files contain 100 training and 381 test IDs with no overlap. A provisional deterministic
50/50 development/calibration split is generated from training IDs using SHA-256 ordering, without
consulting judgments. Test queries remain evaluation-only. Normalized query text has no cross-role
overlap under that split. The complete IDs and split rule are retained in the JSON audit.

| Judgment source | Positive queries S1 / S2 / S3 | Common-positive queries |
|---|---|---:|
| DCTR | 147 / 96 / 133 | 10 |
| Raw clicks | 200 / 166 / 179 | 80 |
| GPT-OSS-120B | 375 / 376 / 377 | 374 |
| GPT-OSS-20B | 378 / 378 / 378 | 377 |
| Llama-3.1-8B | 381 / 381 / 381 | 381 |
| Qwen3-32B | 381 / 381 / 381 | 381 |

DCTR has any judgments for 149 / 97 / 135 queries. The selected LLM sets each cover all 381 IDs,
with 19,675 / 20,489 / 20,961 judgment rows. Human judgments cover 29 test queries at snapshot 3.
Training DCTR and raw qrels each have positive judgments for all 100 training queries.

The 374-query panel is retrospective and conditional on automated judgments. It is not an online
query-selection policy, nor evidence that all 381 queries have stable click-based relevance labels.
Official-task evaluator population semantics remain a separate unresolved contract.

## Representation and resource gates

The [pinned baseline source](https://github.com/clef-longeval/longeval-code/blob/a488f50f2132f38c07631f484eec0b67f3a18926/clef26/scientific-retrieval/baseline-pyterrier-dense/baseline.py)
names Qwen3-Embedding-4B, encodes title-plus-abstract document text, and adds an instruction to
queries. It does not pin the model revision or establish the historical server's full preprocessing,
normalization, truncation, and pooling configuration. The model card's sample query encoder is not
proof of compatibility with the archived vectors. Query vectors are not among the listed archive
members. These points must be resolved before producing comparable real-data retrieval runs.

Still unverified: vector finiteness and normalization, unchanged vectors for shared IDs, original
document text/eligibility, full archive integrity, and query-encoder compatibility.

Observed local capacity was 31.71 GiB host RAM and 15.47 GiB available-to-WSL total RAM. The final
snapshot's vectors alone exceed the WSL RAM limit before graph overhead. All three canonical arrays
would occupy roughly 36.76 GiB, and the development loader currently copies/retains each snapshot.
Do not launch the full pilot unchanged. Next engineering work is memory-mapped/streaming ingestion,
one isolated arm at a time, blocked exact scoring, and a measured resource plan. Raising the WSL
limit alone would not solve the current multi-copy design. No system memory limits were changed.

## Reproduce and locate evidence

```sh
python -m research.longeval fetch-release build/longeval-release/files
python -m research.longeval audit-release build/longeval-release/files --output build/longeval-release/new-release-audit.json
python -m research.longeval inspect-dense build/longeval-release/dense --release-files build/longeval-release/files --output build/longeval-release/new-dense-audit.json
```

Add `--offline` to `inspect-dense` after prefixes are cached. Existing complete downloads are reused
only after checksum validation; existing output reports are never overwritten. Access query strings
are not copied into report URLs. Downloads do not execute upstream loader code.

The actual execution records are:

- `build/longeval-release/release-final.json`, SHA-256
  `336468c72a49fe2531b96e8ae971b533dc4b90a733585b451421b84ef62ee749`.
- `build/longeval-release/dense-final.json`, SHA-256
  `4b3c61170d92d9d7c1a29950345681c4f1d9f1998d4e6bef150c9e584737da5b`.

They record per-file/member hashes, query cohorts, row-order/live-set hashes, membership changes,
judgment coverage, reader versions, and source identities. Cached release files and prefixes stay
under ignored `build/`; the commands regenerate the audit without hidden notebook state.
