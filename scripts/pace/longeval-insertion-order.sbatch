#!/usr/bin/env bash
# EXPLORATORY follow-up: insertion order at full scale. See LONGEVAL_INSERTION_ORDER.md.
#   sbatch longeval-insertion-order-01/scripts/pace/longeval-insertion-order.sbatch DATASET BACKEND SEED
# DATASET is `full` (all of snapshot 3) or `draw1`, `draw2`, `draw3` (stratified draws of snapshot-1 size).
# One job builds the index in three insertion orders: as arrived, reversed and shuffled.
# All archive downloads, extracted matrices and indexes stay on compute-node disk. No judgments are present.
#SBATCH --job-name=longeval-order
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
#SBATCH --output=longeval-order-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-insertion-order-01"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
DATASET="${1:?dataset}"
BACKEND="${2:?backend}"
SEED="${3:?seed}"
case "$DATASET" in full) DRAW=0 ;; draw1) DRAW=1 ;; draw2) DRAW=2 ;; draw3) DRAW=3 ;;
    *) echo 'dataset must be full, draw1, draw2 or draw3' >&2; exit 2 ;; esac
case "$BACKEND" in epzero|faiss_hnsw) ;; *) echo 'backend must be epzero or faiss_hnsw' >&2; exit 2 ;; esac
case "$SEED" in 11|29|47) ;; *) echo 'seed must be 11, 29 or 47' >&2; exit 2 ;; esac
OUTPUT="$ROOT/build/pace-longeval-insertion-order/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
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
    fi
    if [ -n "${STAGE:-}" ] && [ -f "$STAGE/sample/sample.json" ]; then cp "$STAGE/sample/sample.json" "$OUTPUT/" || true; fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'dataset=%s backend=%s seed=%s\n' "$DATASET" "$BACKEND" "$SEED" | tee "$OUTPUT/arm.txt"

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
if shutil.disk_usage(sys.argv[1]).free < 125 * 1024**3:
    raise SystemExit('Need at least 125 GiB free on compute-node temporary disk before dense downloads')
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
    "$PACKAGE/research/longeval/tests/test_post_freeze_analyses.py" \
    --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")

PHASE=prepare
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$PYTHON" -P -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation")

PHASE=sample
echo "phase=$PHASE"
# The sample holds the vectors in insertion-ready form, the arrival group of every document and the
# exact neighbours of the 381 test queries. For the full snapshot these are checked against the
# study's held-out exact references before anything is built.
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/sample-resources.txt" \
    "$PYTHON" -P -m research.longeval.insertion_pilot prepare-full "$STAGE/dense" "$PACKAGE/inputs" "$STAGE/sample" \
    --draw "$DRAW") > "$OUTPUT/sample-summary.json"
cat "$OUTPUT/sample-summary.json"

PHASE=run
echo "phase=$PHASE"
for order in appended reversed shuffled; do
    # Each order is built in a fresh process; only the node-local sample is shared.
    (cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/$order-resources.txt" \
        "$PYTHON" -P -m research.longeval.insertion_pilot run "$STAGE/sample" \
        "$OUTPUT/$DATASET/$BACKEND/$order-$SEED" --order "$order" --seed "$SEED" --backend "$BACKEND" \
        --M 16 --ef-construction 100 --efs 100 200 400 800) > "$OUTPUT/$order-summary.json"
    printf 'built %s\n' "$order"
done

PHASE=compare
echo "phase=$PHASE"
if [ "$DATASET" = full ]; then
    # Recorded, not required: does the chronological build match the held-out incremental run per query?
    (cd "$STAGE" && "$PYTHON" -P -m research.longeval.insertion_pilot compare-heldout "$STAGE/sample" \
        "$OUTPUT/$DATASET/$BACKEND/appended-$SEED" "$PACKAGE/inputs/baselines/$BACKEND-$SEED.json") \
        | tee "$OUTPUT/heldout-comparison.json"
fi
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: insertion-order builds completed for %s %s seed %s: %s\n' "$DATASET" "$BACKEND" "$SEED" "$OUTPUT"
