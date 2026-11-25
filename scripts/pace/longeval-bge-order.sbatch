#!/usr/bin/env bash
# E4 on the BGE vectors: the arrival-order reversal (Part A) and the engine arm (Part B). See LONGEVAL_BGE_ORDER.md.
#   sbatch longeval-bge-order-01/scripts/pace/longeval-bge-order.sbatch order BACKEND SEED   (Part A)
#   sbatch longeval-bge-order-01/scripts/pace/longeval-bge-order.sbatch engine SEED          (Part B)
# The sample is written on compute-node disk from the verified BGE vectors (job 7290861) by bge_sample.py;
# only aggregates, returned IDs and the sample's small metadata files are kept. No judgments are present.
#SBATCH --job-name=longeval-bge-order
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --tmp=100G
#SBATCH --time=16:00:00
#SBATCH --output=longeval-bge-order-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-bge-order-01"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
BGE="$ROOT/build/pace-longeval-bge/jobs/7290861/vectors"
PART="${1:?part: order or engine}"
case "$PART" in
    order) BACKEND="${2:?backend}"; SEED="${3:?seed}"
           case "$BACKEND" in epzero|faiss_hnsw) ;; *) echo 'backend must be epzero or faiss_hnsw' >&2; exit 2 ;; esac ;;
    engine) BACKEND=-; SEED="${2:?seed}" ;;
    *) echo 'part must be order or engine' >&2; exit 2 ;;
esac
case "$SEED" in 11|29|47) ;; *) echo 'seed must be 11, 29 or 47' >&2; exit 2 ;; esac
OUTPUT="$ROOT/build/pace-longeval-bge-order/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    for name in sample.json doc-ids.txt.gz query-ids.json confirm-query-ids.json cohort.npy exact-top100.npy confirm-exact-top100.npy; do
        if [ -n "${STAGE:-}" ] && [ -f "$STAGE/sample/$name" ]; then cp "$STAGE/sample/$name" "$OUTPUT/" || true; fi
    done
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'part=%s backend=%s seed=%s\n' "$PART" "$BACKEND" "$SEED" | tee "$OUTPUT/arm.txt"

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
export PYTHONPATH="$PACKAGE" PIP_DISABLE_PIP_VERSION_CHECK=1
if [ "$PART" = order ]; then export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; fi
"$CPU_PYTHON" - "$STAGE" <<'PY'
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
    gcc --version | head -n 1
    "$CPU_PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"

PHASE=tests
echo "phase=$PHASE"
(cd "$STAGE" && "$CPU_PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_bge_sample.py" "$PACKAGE/research/longeval/tests/test_post_freeze_analyses.py" \
    "$PACKAGE/research/longeval/tests/test_order_pairs.py" "$PACKAGE/research/longeval/tests/test_engine_arm_analysis.py" \
    --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")

PHASE=sample
echo "phase=$PHASE"
# The verified BGE vectors (their SHA-256 are pinned in bge_sample.PINNED), in the Qwen sample's document
# and query order (inputs/qwen-sample, from the order rerun's job 7253605).
(cd "$STAGE" && /usr/bin/time -v -o "$OUTPUT/sample-resources.txt" \
    "$CPU_PYTHON" -P -m research.longeval.bge_sample "$BGE" "$PACKAGE/inputs/qwen-sample" "$STAGE/sample") \
    | tee "$OUTPUT/sample-summary.json"

PHASE=run
echo "phase=$PHASE"
if [ "$PART" = order ]; then
    for order in appended reversed shuffled; do
        # Each order is built in a fresh process, as in the order rerun.
        (cd "$STAGE" && /usr/bin/time -v -o "$OUTPUT/$order-resources.txt" \
            "$CPU_PYTHON" -P -m research.longeval.insertion_pilot run "$STAGE/sample" \
            "$OUTPUT/builds/$BACKEND/$order-$SEED" --order "$order" --seed "$SEED" --backend "$BACKEND" \
            --M 16 --ef-construction 100 --efs 100 200 400 800) > "$OUTPUT/$order-summary.json"
        printf 'built %s\n' "$order"
    done
else
    # The development Epzero, built from the packaged source into a node-local environment.
    "$UV" venv --python "$CPU_PYTHON" --seed "$STAGE/dev"
    DEV_PYTHON="$STAGE/dev/bin/python"
    "$DEV_PYTHON" -m pip install --only-binary=:all: numpy==2.4.6 scikit-build-core==1.1.2 pybind11==3.1.0 cmake==3.31.6 ninja==1.11.1.4
    export PATH="$STAGE/dev/bin:$PATH"
    (cd "$PACKAGE/engine" && "$DEV_PYTHON" -m pip install --no-build-isolation --no-deps . )
    "$DEV_PYTHON" -m pip freeze > "$OUTPUT/dev-packages.txt"
    "$DEV_PYTHON" - <<'PY' | tee "$OUTPUT/engine-check.json"
import json, numpy as np, epzero
index = epzero.Index(dim=4, max_elements=10, extra_backlinks=8)
index.add(np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32))
index.refine_links(threads=1, ef=0, first=5)
print(json.dumps({'epzero': epzero.__file__, 'extra_backlinks': True, 'refine_first': True}))
PY
    for arm in grown repaired8w rebuilt; do
        (cd "$STAGE" && /usr/bin/time -v -o "$OUTPUT/$arm-resources.txt" \
            "$DEV_PYTHON" -P -m research.longeval.engine_arm "$STAGE/sample" "$OUTPUT/builds/engine" --arm "$arm" --seed "$SEED") \
            > "$OUTPUT/$arm-summary.json"
        printf 'built %s\n' "$arm"
    done
fi
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: BGE %s %s seed %s: %s\n' "$PART" "$BACKEND" "$SEED" "$OUTPUT"
