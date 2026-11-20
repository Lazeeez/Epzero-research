# Controls following verified PACE job 7108471

Update: replay 7111579 and shuffled 7111580 have now completed and passed local verification.
See EPZERO_CONTROL_RESULTS_SEED11.md. The commands in the next section are retained as the
record of those completed submissions; do not resubmit them. The next submissions are below.

The first incremental trajectory completed with a verified peak worker RSS of 17.2644 GiB
and trajectory wall time of 50:16.05. Retain the existing 64 GB RAM, eight-hour exclusive
Gold 6226 allocation for the first controls; full rebuilds require more construction work.
No new bundle or source changes are needed.

From the same isolated PACE checkout:

```bash
cd "$HOME/scratch/longeval-20261116/Epzero"
sbatch --job-name=longeval-replay11 scripts/pace/longeval-trajectory.sbatch epzero replay 11
sbatch --job-name=longeval-shuffle11 scripts/pace/longeval-trajectory.sbatch epzero shuffled 11
```

Each submission requests its own exclusive allocation. The existing script stages every new
large download and extraction under that compute node's temporary directory. It retains the
same queries, precision, exact references, calibration grid, capacity, construction parameters
and seed. Only construction mode changes. The Slurm log filename remains
`longeval-trajectory-JOB_ID.log` for both jobs, and output folders are keyed by job ID.

Replay rebuilds each snapshot from the same chronological ID sequence used by incremental
construction. Compare topology fingerprints and ranked-output hashes; fixture tests predict
equality, but full-corpus equality remains to be tested. Shuffled construction rebuilds each
snapshot using the already-declared seeded permutation. Compare all 50 development queries,
including the two zero-recall cases, at fixed effort and the calibration-selected effort.
Keep target misses explicit. Do not retune the protocol to remove the observed failures.

After each job, replace JOB_ID with its numeric ID and return:

```bash
tail -n 60 longeval-trajectory-JOB_ID.log
tar -czf longeval-trajectory-JOB_ID-evidence.tar.gz \
  -C build/pace-longeval-trajectory/jobs JOB_ID
```

Copy each archive to the local project's `build` directory. Inspect these controls before
expanding to FAISS and multiple seeds. These development comparisons are diagnostic; they
do not by themselves support a novel published result or final-test inference.

## Next submissions: FAISS HNSW, seed 11

Use the same checkout and existing source bundle; no upload or edit is needed. The native
fixture checks already exercise FAISS in all three modes, and the job reruns them before
downloads. Full-data FAISS resource use remains to be measured; retain 64 GB RAM and eight
hours, with a separate exclusive node allocation per job.

```bash
cd "$HOME/scratch/longeval-20261116/Epzero"
for mode in incremental replay shuffled; do
  sbatch --job-name="longeval-faiss-${mode}" \
    scripts/pace/longeval-trajectory.sbatch faiss_hnsw "$mode" 11
done
```

The source runner explicitly canonicalizes FAISS efSearch before fingerprinting to prevent
prior calibration from changing serialization-only comparisons. Interpret backend differences
within each backend's controls: native memory-growth policy and seeded graph construction
are not identical to Epzero. Return the same per-job evidence archives and log tails as above.
All large files continue to use compute-node temporary storage, with no shared-storage fallback.
