# PACE native HNSW cost measurements

Copy `build/longeval-pace-native-cost-02.tar.gz` to the existing PACE checkout and verify
its SHA-256 against `build/longeval-pace-native-cost-02-receipt.json`. Then run:

```bash
tar -xzf longeval-pace-native-cost-02.tar.gz
build/pace-longeval/venv/bin/python scripts/pace/longeval-gpu-verify.py
for seed in 11 29 47; do
  sbatch --job-name="le-native-cost-s${seed}" scripts/pace/longeval-native-cost.sbatch "$seed"
done
```

Three jobs; each runs Epzero and FAISS HNSW sequentially using the original incremental
construction. One exclusive Gold 6226 node, one computation thread, 64 GB RAM, 160 GB
temporary disk, 12-hour ceiling. Original graphs and ranking files must reproduce before
timings are accepted. All large data stays in compute-node temporary storage. No package
installation or engine rebuild is required. The existing IVF timings are retained.

After completion:

```bash
sacct -X -j JOB_ID_11,JOB_ID_29,JOB_ID_47 \
  --format=JobID,JobName%32,State,ExitCode,Elapsed
for job in JOB_ID_11 JOB_ID_29 JOB_ID_47; do
  tail -n 35 "longeval-native-cost-${job}.log"
  tar -czf "longeval-native-cost-${job}-evidence.tar.gz" \
    -C build/pace-longeval-native-cost/jobs "$job" \
    -C "$PWD" "longeval-native-cost-${job}.log"
done
```

Copy the three archives to the local `build` directory. They contain bounded run files,
timing/counter records, tests and audit receipts, not full native indexes or corpus files.
Execution completion alone does not establish a valid speed comparison. Local score,
baseline, protocol and hardware verification must precede interpretation.
