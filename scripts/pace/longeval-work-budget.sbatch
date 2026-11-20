#!/usr/bin/env bash
# Full-cohort distance-budget diagnostic; original graph/binary must reproduce.
# Large downloads and full work logs stay on compute-node temporary disk.
#SBATCH --job-name=longeval-work-budget
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
#SBATCH --output=longeval-work-budget-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
INPUTS="$ROOT/build/longeval-trajectory-inputs"
OUTPUT="$ROOT/build/pace-longeval-work-budget/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
SEED="${1:-11}"
case "$SEED" in 11|29|47) ;; *) echo 'Work-budget seed must be 11, 29 or 47' >&2; exit 2 ;; esac
BASELINE="$ROOT/build/longeval-diagnostic-inputs/seed-$SEED/report.json"
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
    printf 'node_stage=%s\nseed=%s\n' "$STAGE" "$SEED"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
    "$PYTHON" --version
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"
srun --cpu-bind=cores --mem-bind=local "$PYTHON" -m pytest -q \
    research/longeval/tests/test_work_budget.py \
    research/longeval/tests/test_retention_search.py research/longeval/tests/test_retention_diagnostic.py \
    research/longeval/tests/test_failure_diagnostic.py research/longeval/tests/test_trajectory.py \
    -o "cache_dir=$STAGE/pytest-cache" --basetemp="$STAGE/tests" --junitxml="$OUTPUT/work-budget-tests.xml"
diagnostic=("$PYTHON" -m research.longeval.work_budget "$STAGE/dense" "$STAGE/preparation/vectors/audit.json"
    "$INPUTS/text-audit.json" "$INPUTS/queries" "$INPUTS/float32" "$INPUTS/exact" "$BASELINE"
    "$OUTPUT/experiment" "$STAGE/work-budget-work" --seed "$SEED")
"${diagnostic[@]}" --check-only
srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$PYTHON" -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation"
srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/work-budget-resources.txt" "${diagnostic[@]}"
"$PYTHON" scripts/pace/longeval-gpu-verify.py
printf 'PASS: full-cohort work-budget diagnostic completed; inspect calibration targets, tails and work counts: %s\n' "$OUTPUT"
