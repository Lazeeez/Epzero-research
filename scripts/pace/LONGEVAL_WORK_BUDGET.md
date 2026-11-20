# PACE full-cohort search-work diagnostic

The additive `build/longeval-pace-work-budget-01.tar.gz` supplement preserves all original
127 source files and all 64 files in the retention supplement. It adds the full-cohort work
diagnostic, tests, protocol, job script, and verified seed-47 baseline inputs. No engine rebuild
or package installation is required.

Copy the supplement to the existing isolated PACE checkout. Verify its SHA-256 against
`build/longeval-pace-work-budget-01-receipt.json`, extract it, and submit:

```bash
tar -xzf longeval-pace-work-budget-01.tar.gz
build/pace-longeval/venv/bin/python scripts/pace/longeval-gpu-verify.py
for seed in 11 29 47; do
  sbatch --job-name="le-work-s${seed}" scripts/pace/longeval-work-budget.sbatch "$seed"
done
```

Each job uses an exclusive Gold 6226 CPU node, one computation thread, 64 GB RAM, 160 GB
node-local temporary disk and an eight-hour limit. Tests and input/native checks run before
corpus downloads. All large downloads and work files stay in the compute node's temporary
directory. There is no shared-storage fallback; bounded output evidence is retained separately.

The job reproduces the original incremental trajectory, traces ef=200 and ef=3200 for all
50 calibration and 50 development queries at all three snapshots, and evaluates seven declared
distance-budget prefixes. Selection uses calibration queries only. Prefix results are offline
work diagnostics, not native capped-search latency measurements.

Results are in `build/pace-longeval-work-budget/jobs/JOB_ID`; the log is
`longeval-work-budget-JOB_ID.log`. After completion:

```bash
tail -n 60 longeval-work-budget-JOB_ID.log
tar -czf longeval-work-budget-JOB_ID-evidence.tar.gz \
  -C build/pace-longeval-work-budget/jobs JOB_ID
```

Return the three archives to the local project's `build` directory. Include the complete
bounded trace evidence; do not include corpus files or native indexes. Target misses are
recorded outcomes, while reproduction/parity failures invalidate the affected execution.
