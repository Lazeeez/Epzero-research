# PACE candidate-retention experiment

Use `build/longeval-pace-retention-01.tar.gz`, an additive supplement to the existing research
checkout. It preserves all 127 original source files and all 53 files in the previous diagnostic
supplement. It adds a separate search replay, intervention observer, tests and this job protocol.
No native rebuild or package installation is needed.

After copying the supplement to the PACE checkout, verify its SHA-256 against
`build/longeval-pace-retention-01-receipt.json`, then:

```bash
tar -xzf longeval-pace-retention-01.tar.gz
build/pace-longeval/venv/bin/python scripts/pace/longeval-gpu-verify.py
sbatch --job-name=le-retention-s11 scripts/pace/longeval-retention.sbatch 11
sbatch --job-name=le-retention-s29 scripts/pace/longeval-retention.sbatch 29
```

Each job requests an exclusive Gold 6226 CPU node, one computational thread, 64 GB RAM,
160 GB local temporary space and eight hours. Tests, native-kernel ABI parity and baseline
source/binary/input checks run before downloads. All corpus archives, extracted vectors,
audit arrays and full work logs stay on the compute node's temporary disk, with no shared
storage fallback. The existing shared Python environment is read-only.

The job reproduces all three baseline snapshots and runs 72 search arms at snapshot 3.
It checks native event/ranking/count parity before interventions, records complete expansions
and evictions, and compares targeted and sham deferred expansions with wider searches under
the same distance-evaluation cap. See `research/longeval/docs/development/HNSW_RETENTION_EXPERIMENT.md`.

Results persist under `build/pace-longeval-retention/jobs/JOB_ID`. Logs are
`longeval-retention-JOB_ID.log`. A successful exit establishes execution of the checks,
not that an intervention improved recall. Return both result archives for local verification:

```bash
tail -n 60 longeval-retention-JOB_ID.log
tar -czf longeval-retention-JOB_ID-evidence.tar.gz \
  -C build/pace-longeval-retention/jobs JOB_ID
```

Copy each archive into the local project's `build` directory. No corpus or native index
belongs in these evidence archives. Bounded compressed full-event logs and their summaries do.
