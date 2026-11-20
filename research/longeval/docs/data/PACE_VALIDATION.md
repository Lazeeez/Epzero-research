# PACE bootstrap evidence — job 7098485

The account owner supplied the final Slurm log in the conversation on 2026-11-16.
The account owner subsequently copied the evidence archive back to the local workspace.
Local inspection of the JUnit reports and independent pilot verification confirm a successful
cluster build/test/pilot run:

- 115/115 C++ tests passed; CTest elapsed time 95.02 seconds.
- 138 Python tests passed in 14.18 seconds.
- Synthetic pilot completed with 153 result rows and 157 verified artifacts.
- All 127 source-manifest entries verified at base commit
  `ce951f6b5176527597f172e18003ffc87713952f` plus the transferred overlay.
- The script reached its final `PASS` message.

Job 7098485 used account `coc`, QOS `coc-ice`, partition `coc-cpu`, and node
a single compute node. Its allocation was four CPUs and 16 GB RAM; the epilog reports
wall time 00:03:37 and memory `1874728K`. These are scheduler measurements for the
bootstrap, not per-index measurements or full-data research resource estimates.
The epilog timestamp is `Nov-15-2026 20:06:05`; its timezone is not supplied.

Remote evidence directory:

```text
<PACE checkout>/build/pace-longeval/jobs/7098485
```

Preserve the log, hardware description, package freeze, source/staging manifests,
JUnit results, exit-code file, fixture and pilot outputs. Compiled build directories
need not be included in a small evidence transfer. The native binary identity is
recorded by the pilot provenance manifest.

This supersedes the earlier statements that PACE inventory/submission were pending
in STATUS.md, RESOURCE_VALIDATION.md and scripts/pace/LONGEVAL.md. Those transferred
source files are retained byte-for-byte for the existing source-manifest validation.
The bootstrap supplement was SHA-256
`0e7008ab50afe47749d97d2a4aa21cc6ec413baf2774765560bdd4658c2a9b4f`.

Remaining gates include full snapshot/text consistency, measured encoder agreement,
query encoding, full-corpus execution and resource controls, statistical analysis,
and novelty assessment. The cluster pilot is synthetic and does not establish a
retrieval relevance effect.

## Returned evidence verification

Archive `build/longeval-7098485-evidence.tar.gz`, SHA-256
`8c4623064b43cfa5fdb5622b298c8655d21094a77cb76f1dc815dfbaf1a904a5`, was safely extracted into
`build/longeval-pace-evidence-7098485/`. All 157 pilot artifact hashes and 127 source file hashes
verified locally. The exit-code file contains zero; JUnit confirms 115 native and 138 Python tests
with zero failures, errors or skips. The CLI verifier re-scored the transferred run files against
the transferred fixture successfully. Package/hardware records and source manifests are retained.

Comparison with `build/longeval-streamed-pilot` found identical document IDs and ranks for all 153
run files (36,720 scored rows). State fingerprints are byte-identical. Forty-five run files are
byte-identical; the other 108 HNSW run files contain numerical score differences. Across all rows,
4,325 scores differ, with maximum absolute difference `3.814697265625e-06`. The mechanism causing
those differences was not isolated; cross-machine bitwise score reproducibility is not claimed.
The machine-readable comparison is `build/longeval-pace-evidence-7098485/local-verification.json`.

All per-query relevance metrics, strict and tie-aware ANN recall, and selected search efforts
also match. The result records differ only within calibration records, as detailed in that JSON.
