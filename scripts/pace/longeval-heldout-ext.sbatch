#!/usr/bin/env bash
# EXPLORATORY held-out extension (post-freeze): shuffled-order HNSW arm and ef sweep. No judgments present.
#   sbatch longeval-heldout-ext-01/scripts/pace/longeval-heldout-ext.sbatch BACKEND SEED
# The incremental arm must reproduce the analysed held-out run; the shuffled arm rebuilds in random order.
# All archive downloads, extracted matrices and audit work files stay on compute-node disk.
#SBATCH --job-name=longeval-heldout-ext
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
#SBATCH --output=longeval-heldout-ext-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-heldout-ext-01"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
OUTPUT="$ROOT/build/pace-longeval-heldout-ext/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
BACKEND="${1:?backend}"
SEED="${2:?seed}"
case "$BACKEND" in epzero|faiss_hnsw) ;; *) echo 'backend must be epzero or faiss_hnsw' >&2; exit 2 ;; esac
case "$SEED" in 11|29|47) ;; *) echo 'seed must be 11, 29 or 47' >&2; exit 2 ;; esac
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE/preparation" ]; then
        # Only JSON summaries survive; never copy archives, vectors or native indexes.
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
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'backend=%s seed=%s\n' "$BACKEND" "$SEED" | tee "$OUTPUT/arm.txt"

PHASE=package
echo "phase=$PHASE"
(cd "$PACKAGE" && sha256sum --check --quiet --strict SHA256SUMS)
cp "$PACKAGE/SHA256SUMS" "$PACKAGE/manifest.json" "$OUTPUT/"

PHASE=environment
echo "phase=$PHASE"
exec 9>"$ROOT/build/pace-longeval/setup.lock"
flock -sn 9 || { echo 'Research environment is being rebuilt' >&2; exit 1; }
module purge
module load gcc/13.3.0
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" - "$STAGE" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 120 * 1024**3:
    raise SystemExit('Need at least 120 GiB free on compute-node temporary disk before dense downloads')
PY
{
    date -u
    hostname
    lscpu
    printf 'node_stage=%s\n' "$STAGE"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
    "$PYTHON" --version
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"

PHASE=tests
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local "$PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_heldout_extension.py" "$PACKAGE/research/longeval/tests/test_heldout.py" \
    --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")
(cd "$STAGE" && "$PYTHON" -P -m research.longeval.heldout_cost check "$PACKAGE/inputs" \
    "$PACKAGE/inputs/baselines/$BACKEND-$SEED.json" "$PACKAGE/research/longeval/trajectory-development.json") \
    | tee "$OUTPUT/preflight.json"

PHASE=prepare
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$PYTHON" -P -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation")

PHASE=run
echo "phase=$PHASE"
for mode in incremental shuffled; do
    # Each mode runs in a fresh process; only the node-local corpus is shared.
    (cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/$mode-resources.txt" \
        "$PYTHON" -P -m research.longeval.heldout_extension "$STAGE/dense" "$STAGE/preparation/vectors/audit.json" \
        "$PACKAGE/inputs" "$OUTPUT/$mode" "$PACKAGE/research/longeval/trajectory-development.json" \
        "$PACKAGE/inputs/baselines/$BACKEND-$SEED.json" \
        --backend "$BACKEND" --mode "$mode" --seed "$SEED" --package-root "$PACKAGE")
done
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: held-out extension completed for %s seed %s: %s\n' "$BACKEND" "$SEED" "$OUTPUT"
