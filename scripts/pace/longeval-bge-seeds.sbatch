#!/usr/bin/env bash
# E4 on the BGE vectors, seven further construction seeds (amendment 7). See LONGEVAL_BGE_SEEDS.md.
#   sbatch longeval-bge-seeds-01/scripts/pace/longeval-bge-seeds.sbatch SEED
# Runs the three primary arms in turn: Epzero and FAISS HNSW incremental, IVF frozen_incremental.
# Inputs are written on compute-node disk from the verified BGE vectors (job 7290861); the arms are the
# frozen cores with the frozen configurations, one thread. No judgments are present.
#SBATCH --job-name=longeval-bge-seeds
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --tmp=60G
#SBATCH --time=12:00:00
#SBATCH --output=longeval-bge-seeds-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-bge-seeds-01"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
BGE="$ROOT/build/pace-longeval-bge/jobs/7290861/vectors"
SEED="${1:?seed}"
case "$SEED" in 101|211|307|401|503|601|701) ;; *) echo 'seed must be one of 101 211 307 401 503 601 701' >&2; exit 2 ;; esac
OUTPUT="$ROOT/build/pace-longeval-bge-seeds/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    # The prepared references, cohorts and identity are small and needed for the local analysis.
    if [ -n "${STAGE:-}" ] && [ -d "$STAGE/prepared" ]; then
        mkdir -p "$OUTPUT/prepared"
        for path in "$STAGE"/prepared/*.json "$STAGE"/prepared/*.json.gz "$STAGE"/prepared/*-queries.npy; do
            if [ -f "$path" ]; then cp "$path" "$OUTPUT/prepared/" || true; fi
        done
    fi
    if [ -n "${STAGE:-}" ] && [ -f "$STAGE/sample/sample.json" ]; then cp "$STAGE/sample/sample.json" "$OUTPUT/" || true; fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'arms=epzero,faiss_hnsw,faiss_ivf seed=%s\n' "$SEED" | tee "$OUTPUT/arm.txt"

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
if shutil.disk_usage(sys.argv[1]).free < 30 * 1024**3:
    raise SystemExit('Need at least 30 GiB free on compute-node temporary disk')
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
    "$PACKAGE/research/longeval/tests/test_bge_heldout.py" "$PACKAGE/research/longeval/tests/test_bge_sample.py" \
    "$PACKAGE/research/longeval/tests/test_heldout.py" "$PACKAGE/research/longeval/tests/test_trajectory.py" \
    "$PACKAGE/research/longeval/tests/test_ivf_trajectory.py" \
    --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")

PHASE=sample
echo "phase=$PHASE"
# The BGE sample, for its exact neighbours: the snapshot-3 references must agree with them.
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/sample-resources.txt" \
    "$PYTHON" -P -m research.longeval.bge_sample "$BGE" "$PACKAGE/inputs/qwen-sample" "$STAGE/sample") \
    | tee "$OUTPUT/sample-summary.json"

PHASE=prepare
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/prepare-resources.txt" \
    "$PYTHON" -P -m research.longeval.bge_heldout prepare "$BGE" "$PACKAGE/inputs/qwen-docnos" \
    "$PACKAGE/inputs/training/queries" "$STAGE/sample" "$STAGE/prepared") | tee "$OUTPUT/prepare-summary.json"
rm -f "$STAGE/sample/vectors.npy"            # the prepared snapshots hold the vectors now

PHASE=run
echo "phase=$PHASE"
for backend in epzero faiss_hnsw; do
    (cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/$backend-resources.txt" \
        "$PYTHON" -P -m research.longeval.bge_heldout hnsw "$STAGE/prepared" "$OUTPUT/$backend/trajectory" \
        "$PACKAGE/research/longeval/trajectory-development.json" \
        --backend "$backend" --mode incremental --seed "$SEED" --package-root "$PACKAGE" --native-origin checkout-build)
done
(cd "$STAGE" && srun --cpu-bind=cores --mem-bind=local /usr/bin/time -v -o "$OUTPUT/frozen_incremental-resources.txt" \
    "$PYTHON" -P -m research.longeval.bge_heldout ivf "$STAGE/prepared" "$OUTPUT/ivf/frozen_incremental" \
    "$PACKAGE/research/longeval/ivf-development.json" \
    --mode frozen_incremental --seed "$SEED" --package-root "$PACKAGE" --native-origin checkout-build)
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: BGE seed %s: three primary arms completed; verify locally before analysis: %s\n' "$SEED" "$OUTPUT"
