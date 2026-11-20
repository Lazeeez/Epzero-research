# First document encoder agreement diagnostic

The pinned Qwen3-Embedding-4B model completed CPU BF16 encoding of 32 documents using the
declared last-token pooling and float32 L2 normalization. Encoded vectors and their ID order
were checked against the completed encoding receipt. Comparison joined archived vectors by
document ID and reverified the full snapshot-1 vector-file SHA-256.

Cosine similarity between reconstructed and archived document vectors:

| Statistic | Value |
|---|---:|
| Minimum | 0.9997143308851861 |
| Median | 0.9998046445874654 |
| Maximum | 0.9998471741421128 |
| Bitwise-identical rows | 0 / 32 |
| Maximum absolute coordinate difference | 0.002508861944079399 |

Evidence: `build/longeval-encoder-agreement-01.json`, with per-document measurements,
source and output hashes, and explicit limitations. The one-off analysis script is
`build/compare-longeval-encoder-diagnostic-01.py`; its SHA-256 is recorded in the report.

The initial sample came from an incomplete archive prefix. It has now been checked against
the complete archive: upstream MD5 passed, its complete JSONL member was read through CRC
verification, and all 32 sample texts matched. The original provisional evidence is retained;
the new report documents the subsequent verification without rewriting that history.

The independent full snapshot-1 text audit also completed: 869,902 unique eligible documents,
zero duplicate/conflicting/empty-text rows, and exact coverage of the 869,902 dense-index IDs.
Evidence is `build/longeval-text-snapshot1-audit/audit.json`. A deterministic hash-selected
sample from the complete collection is available in that audit directory for a broader check.
All three text ZIPs are now present, and the all-snapshot audit has been launched at
`build/longeval-text-full-audit/`; its result remains pending.

This establishes close document-vector agreement on a first-member sample, not a representative
collection-level guarantee or a passed query-representation gate. No acceptance threshold was
assigned after inspecting these results. Historical runtime, precision and truncation details
remain unresolved, and the cause of the numerical differences has not been isolated. Query
encoding, broader representative diagnostics and sensitivity analysis remain necessary before
using the reconstructed representation for the primary real-data experiments.

## Representative follow-up

The reusable `research.longeval.representation` module exports the deterministic hash-selected
sample from a completed text audit, verifies the full reference vector-file identity, joins by
document ID, and creates a compact portable reference bundle. Comparison checks both receipts,
input-text identity, vector/ID hashes, membership, dimension and unit norms. It never silently
normalizes, substitutes missing IDs, or assigns an acceptance threshold.

Fifteen focused representation, encoder and document tests passed in the Linux environment;
JUnit is `build/longeval-representation-test-02.xml`. The portable 32-document reference is
`build/longeval-representative-sample-01/`. A CPU BF16 follow-up is running under
`build/longeval-representative-local-02/` and will automatically compare outputs when encoding
finishes. The first launcher failed before encoding due to a module search-path error; its logs
are preserved under `build/longeval-representative-local-01/`. The corrected launcher uses Python
module invocation and the second worker has successfully loaded the model.

The PACE supplement `build/longeval-pace-agreement-01.tar.gz` contains ten checksummed files,
including the reference sample and a separate CUDA setup/job. SHA-256:
`04759ef8d2baf8e48d9f13982c1a0d2affd1262bfa67936f87715242341ee44a`.
Its extracted contents were verified together with the original source and bootstrap bundles;
both shell scripts pass syntax checks. GPU execution has not yet happened. The job will compare
GPU BF16 and float32 on the identical sample, keeping TF32 disabled and recording both results.
See `scripts/pace/LONGEVAL_AGREEMENT.md` for the workflow and the checked baseline query prompt.

## Storage correction requested by the account owner

All subsequent large PACE downloads must use compute-node temporary storage. The replacement
bundle is `build/longeval-pace-agreement-02.tar.gz`, SHA-256
`74da98dc9a02c8b5ab13b01cdb6b21e1c1de55a92cebb7c072d0c17d0e2df823`.
It supersedes agreement-01. Submit the Slurm job directly; setup now runs inside that allocation,
with the model, packages, environment and caches under the node's job temporary directory.
The small source/reference bundle and result evidence remain in the shared checkout. Existing
bootstrap files are only read; no existing remote data was deleted or moved by this change.

The replacement requests 80 GB local disk and four hours including downloads. Four routing
tests passed: cache confinement, rejection of shared/memory filesystems, rejection of a stage
outside the node temporary directory, and refusal to run setup on login. Three scripts passed
Bash syntax validation, and all 127 original plus 12 supplemental files verified together.
Actual compute-node download access and GPU execution remain untested until submission.

The complete three-snapshot text audit has now finished with `fixed_text_gate_passed=true` in
`build/longeval-text-full-audit/audit.json`. Shared document text is unchanged across both
transitions. This closes the full text-consistency check, not the query-encoding gate.

## Representative CPU result

The corrected local worker completed encoding and automatic comparison for all 32 hash-selected
documents. CPU BF16 cosine agreement with the released vectors ranges from
`0.9996131269767212` to `0.9998640770392512`, with median `0.9997883869526796`.
Zero rows are bitwise-identical; the maximum absolute coordinate error is
`0.0031345579773187637`. These observations do not assign an acceptance threshold.
Evidence: `build/longeval-representative-local-02/agreement.json`; the input reference manifest
SHA-256 is `aaa62e811fede1dee0c92f7658c533f5263e32730ac774349f3636ebd679eea6`.

The account owner has reported that the PACE GPU job finished. Its job ID, final status and
artifacts have not yet been supplied, so GPU success and precision comparisons remain unverified.

Subsequently supplied log: job 7104607 stopped in the temporary-filesystem preflight after one
second, before downloading or encoding. `findmnt` returned `xfs` on two lines; the guard incorrectly
compared the entire multiline result with one filesystem name. The exact failure was reproduced
locally. The correction validates every record independently, accepting repeated local filesystems
and rejecting any mixed shared/memory/unknown result. All eight storage tests and Bash syntax
validation passed. No GPU precision results were produced by this failed job.

Replacement bundle `build/longeval-pace-agreement-03.tar.gz` has SHA-256
`00a634f6186e35624a12859bd7a272432c502af581f8b72e68bd66889511e80e`.
The original 127 source files and all 12 replacement-bundle files verified together after extraction.
The replacement was subsequently submitted as job 7106389. The supplied log reports both
GPU precision diagnostics completed successfully in 5 minutes 42 seconds, with float32 median
cosine agreement `0.9999491416916138`. See [PACE_ENCODER_VALIDATION.md](PACE_ENCODER_VALIDATION.md)
for the result and its limitations. The GPU artifacts have now been retrieved and independently
verified, including recomputation of all per-document comparisons. GPU float32 gives higher
cosine agreement than GPU BF16 on all 32 sample documents. The compute-node-only large-download
requirement remains in force. Training-query ranking sensitivity remains the next validation step.
