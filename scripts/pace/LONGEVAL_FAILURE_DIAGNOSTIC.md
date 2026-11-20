# Focused Epzero failure diagnostic

The supplement `longeval-pace-diagnostic-01.tar.gz` extends the existing additive manifest.
It preserves all 127 original source files, all archived inputs and the engine binary. The
existing trajectory module gains an optional observer called after its original graph,
calibration and run outputs have been recorded. A new diagnostic module implements the
post-hoc protocol; the original 18-run results remain separately archived.

Copy and extract the checksummed supplement in the existing isolated PACE checkout, then:

```bash
sbatch --job-name=longeval-diag11 scripts/pace/longeval-failure-diagnostic.sbatch 11
sbatch --job-name=longeval-diag29 scripts/pace/longeval-failure-diagnostic.sbatch 29
```

Both jobs retain an exclusive Gold 6226 node, one computation thread, 64 GB RAM, eight hours
and 160 GB node-local temporary disk. They download and audit all three dense snapshots in
compute-node temporary storage. Tests and baseline source/runtime/input checks run before
downloads. No shared-storage fallback or model download is used.

For every snapshot, the graph/state and both original ranked-output hashes must reproduce
the verified baseline before diagnostics start. Seed 11 binds job 7108471; seed 29 binds job
7122361. The source manifest checks the unmodified engine; the CLI also checks its binary hash
and the previous package versions. Only the observer hook is exempted from comparison against
the old source list, and its new hash is bound by the supplemental manifest and run provenance.

The diagnostic saves development and calibration rankings for ef=[100,200,400,800,1600,3200].
It traces the two previously failing development queries and two deterministic ID-selected
controls, checks ordinary-search/trace ranking parity, and records entry routes and search work.
Trace prefixes have a declared event limit and explicit truncation counts; the entire native
event stream is hashed and summarized. If a target is still below 0.95 recall at the largest
effort at snapshot 3, directed layer-0 reachability is measured from its actual search start.
The graph is fingerprinted again to check that observation did not change it.

Full trace work files stay node-local. Bounded compressed prefixes, metrics, source/input
receipts and run files persist in `build/pace-longeval-diagnostic/jobs/JOB_ID/`. These are
post-hoc diagnostic measurements, not a new primary configuration or performance benchmark.
Higher-effort recovery alone does not establish the mechanism or justify a publication claim.

After completion, replace JOB_ID and return the evidence archive and log tail:

```bash
tail -n 60 longeval-diagnostic-JOB_ID.log
tar -czf longeval-diagnostic-JOB_ID-evidence.tar.gz \
  -C build/pace-longeval-diagnostic/jobs JOB_ID
```

Copy the archive into the local project's `build` directory. A reproduction mismatch fails
the job and must be investigated before interpreting any partial diagnostics. Native index
checkpoint continuation is not implemented; no graph or corpus is copied into the evidence archive.
