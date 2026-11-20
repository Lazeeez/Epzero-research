# Complete three-snapshot data validation

All three dense archives and all three original text archives have completed download and
identity verification. Every stored vector has been scanned, and eligible source text has been
audited across the complete archives. The historical prefix-only limitations in RELEASE_AUDIT.md
describe the earlier stage; the evidence below supersedes those outstanding data checks.

| Snapshot | Indexed documents | Added IDs | Removed IDs | Changed shared vectors | Changed shared texts |
|---|---:|---:|---:|---:|---:|
| 1 | 869,902 | 869,902 | — | — | — |
| 2 | 1,322,720 | 452,818 | 0 | 0 | 0 |
| 3 | 1,661,900 | 339,180 | 0 | 0 | 0 |

Each dense vector is finite, nonzero and unit-normalized within the declared absolute tolerance
`1e-5`. Shared-vector identity is bitwise, checked by stable document ID. Row positions are not
stable: all 869,902 shared IDs move between snapshots 1 and 2, and 1,322,094 of 1,322,720 shared
IDs move between snapshots 2 and 3.

For each text snapshot, eligible unique IDs match the dense index exactly. There are zero missing
texts, unindexed eligible texts, duplicate eligible rows, conflicting duplicates or empty-text
rows. Text comparison uses the baseline title-plus-abstract construction without added stripping
or normalization. Original ZIP identities and complete member CRCs are checked by the audit.

Evidence:

- Dense audit: `build/longeval-preparation-01/vectors/audit.json`, SHA-256
  `61d6fd13306982cd6bbf647e4240c4a437cd6d144070d51537b3218ff99e01bd`.
- Text audit: `build/longeval-text-full-audit/audit.json`, SHA-256
  `a5f3236f7156bae6f0677fa929c0ebd0d36fa1ccdd2c4282b473558793ae77cb`.

These findings support treating these benchmark snapshots as insert-only with fixed text and
stored vectors. They do not establish historical encoder/runtime identity, compatible query
reconstruction, an ANN-history relevance effect, or novelty. Those require the separate encoder,
query-sensitivity and controlled retrieval experiments.
