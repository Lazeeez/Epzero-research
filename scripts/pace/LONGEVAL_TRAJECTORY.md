# First full-data development trajectory

Apply the checksummed `longeval-pace-trajectory-01.tar.gz` supplement to the existing isolated
checkout. It preserves the original 127 validated source files and extends the existing additive
manifest (the verifier retains its historical `longeval-gpu-verify.py` filename).

```bash
sbatch scripts/pace/longeval-trajectory.sbatch epzero incremental 11
```

Submit only this first job initially. It uses the existing built CPU environment, checks the source
manifests, reruns nine native trajectory/memory tests, downloads all three dense archives into
compute-node temporary storage, rechecks their hashes and full vector audit, then executes the
three-snapshot trajectory. It uses returned CUDA float32 training-query embeddings and the
previously verified exact references. No model download, query encoding or judgment loading is
part of this job. Node-local disk must have at least 120 GiB free; there is no shared fallback.

The exclusive allocation requests Gold 6226, 64 GB RAM, 160 GB temporary disk and eight hours.
Results are under `build/pace-longeval-trajectory/jobs/JOB_ID/`. A zero exit code and PASS line
mean completion, not that the calibration target was met. Inspect every snapshot's recall and
`calibration.target_met`, resource reports and provenance before scheduling the other arms.
Construction time and full-scale memory have not yet been established. A timeout requires a
fresh run; native-index checkpoint continuation is not implemented.

After completion, return the small evidence bundle (replace JOB_ID):

```bash
tar -czf longeval-trajectory-JOB_ID-evidence.tar.gz \
  -C build/pace-longeval-trajectory/jobs JOB_ID
```

Also retain `longeval-trajectory-JOB_ID.log` and `sacct -j JOB_ID` output. On failure, return the
same outputs so the next step is based on the actual failure. Archives and extracted vectors
remain in job-local temporary storage and are not copied to the evidence bundle.
