#!/usr/bin/env bash
# Further construction seeds for the held-out primary arms (exploratory follow-up). No judgments present.
#   sbatch longeval-heldout-seeds-01/scripts/pace/longeval-heldout-seeds.sbatch SEED
# One job per seed runs the three primary arms, incrementally maintained, through the frozen code:
# Epzero HNSW and FAISS HNSW (each with the calibration observer), then IVF with frozen training.
# All archive downloads, extracted matrices and audit work files stay on compute-node disk.
#SBATCH --job-name=longeval-seeds
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
#SBATCH --output=longeval-seeds-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-heldout-seeds-01"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
OUTPUT="$ROOT/build/pace-longeval-heldout-seeds/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
SEED="${1:?seed}"
case "$SEED" in 101|211|307|401|503|601|701) ;; *) echo 'seed must be one of 101 211 307 401 503 601 701' >&2; exit 2 ;; esac
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
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'seed=%s arms=epzero,faiss_hnsw,faiss_ivf\n' "$SEED" | tee "$OUTPUT/arm.txt"

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
    df -h "$STAGE"
    "$PYTHON" --version
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"

PHASE=tests
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local "$PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_heldout_seeds.py" "$PACKAGE/research/longeval/tests/test_heldout.py" \
    "$PACKAGE/research/longeval/tests/test_trajectory.py" "$PACKAGE/research/longeval/tests/test_ivf_trajectory.py" \
    --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")
(cd "$STAGE" && "$PYTHON" -P -m research.longeval.heldout check "$PACKAGE/inputs" \
    "$PACKAGE/research/longeval/trajectory-development.json" "$PACKAGE/research/longeval/ivf-development.json") \
    | tee "$OUTPUT/preflight.json"

PHASE=prepare
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$PYTHON" -P -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation")

PHASE=run
echo "phase=$PHASE"
for backend in epzero faiss_hnsw; do
    # Each arm runs in a fresh process; only the node-local corpus is shared.
    (cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/$backend-resources.txt" \
        "$PYTHON" -P -m research.longeval.heldout_seeds hnsw "$STAGE/dense" "$STAGE/preparation/vectors/audit.json" \
        "$PACKAGE/inputs" "$OUTPUT/$backend" "$PACKAGE/research/longeval/trajectory-development.json" \
        --backend "$backend" --seed "$SEED" --package-root "$PACKAGE" --native-origin checkout-build)
done
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/faiss_ivf-resources.txt" \
    "$PYTHON" -P -m research.longeval.heldout_seeds ivf "$STAGE/dense" "$STAGE/preparation/vectors/audit.json" \
    "$PACKAGE/inputs" "$OUTPUT/faiss_ivf" "$PACKAGE/research/longeval/ivf-development.json" \
    --seed "$SEED" --package-root "$PACKAGE" --native-origin checkout-build)
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: held-out primary arms completed for seed %s; verify locally before analysis: %s\n' "$SEED" "$OUTPUT"
