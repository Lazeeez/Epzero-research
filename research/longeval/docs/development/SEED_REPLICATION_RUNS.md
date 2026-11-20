# Development replication at seeds 29 and 47

Completion update: jobs 7122361–7122372 all reported COMPLETED with exit code 0:0. All 12
returned bundles have now passed local verification. The exact condition mapping is saved
in SEED_REPLICATION_JOBS.tsv; results and limits are in HNSW_THREE_SEED_RESULTS.md. The
submission commands below are retained for provenance and should not be rerun.

All six seed-11 trajectories are verified; see HNSW_SEED11_RESULTS.md. Submit the two remaining
seeds from the existing pilot seed list for both backends and all construction modes. No new
source bundle, model download, query encoding or judgment loading is needed. Each job uses
the existing exclusive CPU allocation, 64 GB RAM, eight hours and 160 GB node-local temporary
storage. Large files are downloaded separately to each compute node's temporary directory.

Run once from the existing PACE checkout. This submits 12 jobs and records the returned IDs:

```bash
cd "$HOME/scratch/longeval-20261116/Epzero"
for seed in 29 47; do
  for backend in epzero faiss_hnsw; do
    for mode in incremental replay shuffled; do
      if job=$(sbatch --parsable --job-name="le-${backend}-${mode}-s${seed}" \
          scripts/pace/longeval-trajectory.sbatch "$backend" "$mode" "$seed"); then
        printf '%s\t%s\t%s\t%s\n' "$job" "$backend" "$mode" "$seed" \
          | tee -a longeval-seeds29-47-submissions.tsv
      else
        printf 'Submission failed: %s %s seed %s\n' "$backend" "$mode" "$seed" >&2
      fi
    done
  done
done
```

If any submission fails, resubmit only its missing combination; do not rerun the entire loop
and duplicate successful submissions. Preserve the TSV, and paste its contents to record the
job-to-condition mapping. A completed scheduler state is not sufficient: return each job's
usual evidence archive after checking its log. Source/input checks and native tests run before
every download. The query cohorts, numeric settings and primary reported metrics stay fixed.

Assess all 12 outcomes before interpreting seed robustness. Repeated queries across seeds are
not independent new queries; no confirmatory statistical method is implied by this submission
batch. Final-test queries and judgments remain outside this development work.
