#!/usr/bin/env bash
# Native HNSW timings; matched single-query protocol to IVF. Node-local large files.
#SBATCH --job-name=longeval-native-cost
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --tmp=160G
#SBATCH --time=12:00:00
#SBATCH --output=longeval-native-cost-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
INPUTS="$ROOT/build/longeval-trajectory-inputs"
OUTPUT="$ROOT/build/pace-longeval-native-cost/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
SEED="${1:-11}"
case "$SEED" in 11|29|47) ;; *) echo 'Seed must be 11, 29 or 47' >&2; exit 2 ;; esac
BACKENDS=(epzero faiss_hnsw)
if [ "$SEED" = 29 ]; then BACKENDS=(faiss_hnsw epzero); fi
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE/preparation" ]; then
        mkdir -p "$OUTPUT/preparation"
        for path in "$STAGE/preparation/job.json" "$STAGE/preparation/vectors/audit.json"; do
            if [ -f "$path" ]; then cp "$path" "$OUTPUT/preparation/" || true; fi
        done
        for step in 1 2 3; do
            path="$STAGE/preparation/vectors/snapshot-$step/scan.json"
            if [ -f "$path" ]; then cp "$path" "$OUTPUT/preparation/scan-$step.json" || true; fi
        done
    fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    exit "$code"
}
trap finish EXIT
exec 9>"$ROOT/build/pace-longeval/setup.lock"
flock -sn 9 || { echo 'Research environment is being rebuilt' >&2; exit 1; }
module purge
module load gcc/13.3.0
source scripts/pace/longeval-node-env.sh
STAGE="$LONGEVAL_NODE_STAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" scripts/pace/longeval-gpu-verify.py
"$PYTHON" - "$STAGE" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 120 * 1024**3:
    raise SystemExit('Need 120 GiB free on compute-node temporary disk before dense downloads')
PY
cp longeval-gpu-manifest.json "$OUTPUT/"
{
    date -u
    hostname
    lscpu
    printf 'node_stage=%s\nseed=%s\nbackend_order=%s,%s\n' "$STAGE" "$SEED" "${BACKENDS[0]}" "${BACKENDS[1]}"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
    "$PYTHON" --version
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"
srun --cpu-bind=cores --mem-bind=local "$PYTHON" -m pytest -q \
    research/longeval/tests/test_native_cost.py research/longeval/tests/test_trajectory.py \
    -o "cache_dir=$STAGE/pytest-cache" --basetemp="$STAGE/tests" --junitxml="$OUTPUT/native-cost-tests.xml"
for backend in "${BACKENDS[@]}"; do
    baseline="$ROOT/build/longeval-native-cost-inputs/seed-$SEED/$backend.report.json"
    "$PYTHON" -m research.longeval.native_cost "$STAGE/dense" "$STAGE/preparation/vectors/audit.json" \
        "$INPUTS/text-audit.json" "$INPUTS/queries" "$INPUTS/float32" "$INPUTS/exact" \
        "$baseline" "$OUTPUT/experiment/$backend" --check-only
done
srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$PYTHON" -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation"
for backend in "${BACKENDS[@]}"; do
    baseline="$ROOT/build/longeval-native-cost-inputs/seed-$SEED/$backend.report.json"
    srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/$backend-resources.txt" \
        "$PYTHON" -m research.longeval.native_cost "$STAGE/dense" "$STAGE/preparation/vectors/audit.json" \
        "$INPUTS/text-audit.json" "$INPUTS/queries" "$INPUTS/float32" "$INPUTS/exact" \
        "$baseline" "$OUTPUT/experiment/$backend"
done
"$PYTHON" scripts/pace/longeval-gpu-verify.py
printf 'PASS: both HNSW native-cost trajectories completed; local verification and cost comparison still required: %s\n' "$OUTPUT"
