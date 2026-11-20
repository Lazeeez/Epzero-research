# PACE IVF maintenance controls

The additive `build/longeval-pace-ivf-02.tar.gz` preserves the original 127 source files
and all 72 files in the prior work-budget supplement. It adds bounded full-data IVF
controls, their protocol, tests and independent evidence checks. No engine rebuild or
package installation is required.

Copy the archive into the existing isolated PACE checkout. Compare its SHA-256 with
`build/longeval-pace-ivf-02-receipt.json`, then run from that checkout:

```bash
sha256sum longeval-pace-ivf-02.tar.gz
tar -xzf longeval-pace-ivf-02.tar.gz
build/pace-longeval/venv/bin/python scripts/pace/longeval-gpu-verify.py
for seed in 11 29 47; do
  sbatch --job-name="le-ivf-s${seed}" scripts/pace/longeval-ivf.sbatch "$seed"
done
```

Version 02 supersedes the version 01 preparation. It preserves queries with
empty output in infeasible-point metadata while correctly exporting their TREC run
(which has no row for an empty query). A regression test verifies this path through the
independent output checker. The original version-01 archive remains unchanged.

Each of the **three jobs** runs frozen incremental, same-training rebuild, and refreshed
training/rebuild in separate sequential processes. All three snapshots are included.
Each job uses an exclusive Gold 6226 CPU node, one computation thread, 64 GB RAM,
160 GB temporary disk and a 12-hour limit. Full input downloads occur once per job,
entirely on compute-node temporary disk. Only bounded experiment evidence persists.

Tests, source verification and all three input preflights run before corpus downloads.
The final verifier recounts query recall and distance counts and requires the identical
training/replay controls to match. A calibration target miss is recorded, not a job error.

Output: `build/pace-longeval-ivf/jobs/JOB_ID/`.
Log: `longeval-ivf-JOB_ID.log`.

After completion, substitute the printed job IDs:

```bash
sacct -X -j JOB_ID_11,JOB_ID_29,JOB_ID_47 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed
for job in JOB_ID_11 JOB_ID_29 JOB_ID_47; do
  tail -n 30 "longeval-ivf-${job}.log"
  tar -czf "longeval-ivf-${job}-evidence.tar.gz" \
    -C build/pace-longeval-ivf/jobs "$job"
done
```

Copy those three evidence archives to the local project's `build` directory. Preserve
training IDs, centroids, full reports and all run files. The archives contain no corpus
vectors or full native indexes. Local verification against audited vectors follows
retrieval; a remote PASS does not itself establish a research finding.
