#!/usr/bin/env bash
# First full-corpus development trajectory; source and small evidence stay shared.
# All archive downloads, extracted matrices and audit work files stay node-local.
#SBATCH --job-name=longeval-trajectory
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
#SBATCH --time=08:00:00
#SBATCH --output=longeval-trajectory-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
INPUTS="$ROOT/build/longeval-trajectory-inputs"
OUTPUT="$ROOT/build/pace-longeval-trajectory/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
BACKEND="${1:-epzero}"
MODE="${2:-incremental}"
SEED="${3:-11}"
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE/preparation" ]; then
        # Only JSON summaries survive; never copy archives, vectors or fingerprint arrays.
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
"$PYTHON" - "$STAGE" "$BACKEND" "$MODE" "$SEED" <<'PY'
import json, shutil, sys
from pathlib import Path
from research.longeval.trajectory import validate_settings
validate_settings(json.loads(Path('research/longeval/trajectory-development.json').read_text()),
                  sys.argv[2], sys.argv[3], int(sys.argv[4]))
if shutil.disk_usage(sys.argv[1]).free < 120 * 1024**3:
    raise SystemExit('Need at least 120 GiB free on compute-node temporary disk before dense downloads')
PY
cp longeval-gpu-manifest.json "$OUTPUT/"
{
    date -u
    hostname
    lscpu
    printf 'node_stage=%s\nbackend=%s\nmode=%s\nseed=%s\n' "$STAGE" "$BACKEND" "$MODE" "$SEED"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
    "$PYTHON" --version
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"
# Run the new native trajectory checks on the actual CPU environment before downloads.
srun --cpu-bind=cores --mem-bind=local "$PYTHON" -m pytest -q \
    research/longeval/tests/test_trajectory.py research/longeval/tests/test_backend_memory.py \
    -o "cache_dir=$STAGE/pytest-cache" --basetemp="$STAGE/tests" --junitxml="$OUTPUT/trajectory-tests.xml"
srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$PYTHON" -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation"
srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/trajectory-resources.txt" \
    "$PYTHON" -m research.longeval.trajectory "$STAGE/dense" "$STAGE/preparation/vectors/audit.json" \
    "$INPUTS/text-audit.json" "$INPUTS/queries" "$INPUTS/float32" "$INPUTS/exact" "$OUTPUT/trajectory" \
    --backend "$BACKEND" --mode "$MODE" --seed "$SEED" --native-origin checkout-build
"$PYTHON" scripts/pace/longeval-gpu-verify.py
printf 'PASS: full-data development trajectory completed; inspect recall targets and resource limits: %s\n' "$OUTPUT"
