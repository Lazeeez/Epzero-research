# Full vector comparison: snapshots 1 and 2

The completed audit at `build/longeval-snapshots12-audit/audit.json` scans all 869,902 vectors
in snapshot 1 and all 1,322,720 vectors in snapshot 2. Both snapshots pass finite, nonzero and
unit-norm validation under the declared `1e-5` absolute tolerance. The audit uses verified
archive identities and records full vector-file hashes and per-row fingerprints.
The audit report SHA-256 is
`3083f50c1a538290c009f367282f115b6ee23fc9f313c8da197753005db32b3a`.

Joining by stable document ID gives:

- 869,902 shared IDs; zero removed IDs; 452,818 added IDs.
- Zero bitwise changes to vectors for shared documents.
- All 869,902 shared IDs occupy different row positions in snapshot 2.

This supports a fixed-vector, insert-only comparison for this transition. Row-position joins
would be invalid. It does not establish source-text consistency, historical encoder identity,
query compatibility, or the corresponding properties of snapshot 3. The overall three-snapshot
gate remains open; the partial audit intentionally does not mark that gate passed.

Reproduction command from the repository root in WSL:

```bash
build/longeval-linux-venv/bin/python -m research.longeval audit-vectors \
  ~/epzero-longeval/dense --snapshots 1 2 --output build/NEW-SNAPSHOTS12-AUDIT
```

The independently running full preparation job will audit all three snapshots after ingestion.
