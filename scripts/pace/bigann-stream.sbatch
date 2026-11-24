#!/usr/bin/env bash
# A public streaming workload: Big-ANN's clustered MSTuring-10M runbook (exploratory; see BIGANN_STREAM.md).
#   sbatch bigann-stream-02/scripts/pace/bigann-stream.sbatch order BACKEND SEED   (Part A, single-threaded builds)
#   sbatch bigann-stream-02/scripts/pace/bigann-stream.sbatch engine SEED          (Part B, all five arms)
#   ENGINE_ARMS=repaired8w sbatch bigann-stream-02/scripts/pace/bigann-stream.sbatch engine SEED   (only the arms named)
# Part A builds the stream in three insertion orders with one implementation (the study's environment);
# Part B builds Epzero's product path in each engine arm with the development engine, built on the node.
# The public stream is downloaded to compute-node disk and prepared there; nothing from LongEval is used.
#SBATCH --job-name=bigann-stream
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
#SBATCH --output=bigann-stream-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/bigann-stream-02"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
PART="${1:?part: order or engine}"
case "$PART" in
    order) BACKEND="${2:?backend}"; SEED="${3:?seed}"
           case "$BACKEND" in epzero|faiss_hnsw) ;; *) echo 'backend must be epzero or faiss_hnsw' >&2; exit 2 ;; esac ;;
    engine) BACKEND=-; SEED="${2:?seed}"
            ENGINE_ARMS="${ENGINE_ARMS:-grown repaired repaired8 repaired8w rebuilt}"
            for arm in $ENGINE_ARMS; do
                case "$arm" in grown|repaired|repaired8|repaired8w|rebuilt) ;; *) echo "unknown engine arm $arm" >&2; exit 2 ;; esac
            done ;;
    *) echo 'part must be order or engine' >&2; exit 2 ;;
esac
case "$SEED" in 11|29|47) ;; *) echo 'seed must be 11, 29 or 47' >&2; exit 2 ;; esac
# The track's public files (data and queries unchanged since October 2023) and the runbook at a pinned commit.
STORE=https://comp21storage.z5.web.core.windows.net/comp23/clustered_data/msturing-10M-clustered
RUNBOOK_URL=https://raw.githubusercontent.com/harsha-simhadri/big-ann-benchmarks/ec58203170e3636ba7da82154ce6322de9cb8b33/neurips23/runbooks/clustered_runbook.yaml
RUNBOOK_SHA256=b65641b28646cdd779b1e9451e71920560d76072f285f3488ef2fea963d3e4e4
DATA_BYTES=4000000008
QUERIES_MD5=YDoYNpLnbqlEiO67RROu3Q==
GT_MD5=caXPVfho+buTAunrUGNpkA==
OUTPUT="$ROOT/build/pace-bigann-stream/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ] && [ -f "$STAGE/sample/sample.json" ]; then cp "$STAGE/sample/sample.json" "$OUTPUT/" || true; fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'part=%s backend=%s seed=%s arms=%s\n' "$PART" "$BACKEND" "$SEED" "${ENGINE_ARMS:--}" | tee "$OUTPUT/arm.txt"

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

PHASE=download
echo "phase=$PHASE"
mkdir "$STAGE/stream"
for name in msturing-10M-clustered.fbin testQuery10K.fbin clu_msturing10M_gt100; do
    curl -fsSL --retry 3 -o "$STAGE/stream/$name" "$STORE/$name"
done
curl -fsSL --retry 3 -o "$STAGE/stream/clustered_runbook.yaml" "$RUNBOOK_URL"
"$CPU_PYTHON" - "$STAGE/stream" "$DATA_BYTES" "$QUERIES_MD5" "$GT_MD5" "$RUNBOOK_SHA256" <<'PY' | tee "$OUTPUT/download-check.json"
import base64, hashlib, json, sys
from pathlib import Path
stream, data_bytes, queries_md5, gt_md5, runbook_sha = Path(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:]
md5 = lambda p: base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()
checks = {'data_bytes': (stream / 'msturing-10M-clustered.fbin').stat().st_size == data_bytes,
          'queries_md5': md5(stream / 'testQuery10K.fbin') == queries_md5,
          'gt_md5': md5(stream / 'clu_msturing10M_gt100') == gt_md5,
          'runbook_sha256': hashlib.sha256((stream / 'clustered_runbook.yaml').read_bytes()).hexdigest() == runbook_sha}
print(json.dumps(checks))
if not all(checks.values()):
    raise SystemExit('a downloaded file differs from the one recorded in the design')
PY

PHASE=tests
echo "phase=$PHASE"
(cd "$STAGE" && "$CPU_PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_bigann_stream.py" "$PACKAGE/research/longeval/tests/test_order_pairs.py" \
    "$PACKAGE/research/longeval/tests/test_engine_arm_analysis.py" --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")

PHASE=prepare
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$CPU_PYTHON" -P -m research.longeval.bigann_stream prepare "$STAGE/stream/msturing-10M-clustered.fbin" \
    "$STAGE/stream/testQuery10K.fbin" "$STAGE/stream/clu_msturing10M_gt100" "$STAGE/stream/clustered_runbook.yaml" \
    "$STAGE/sample") | tee "$OUTPUT/sample-summary.json"
rm -f "$STAGE/stream/msturing-10M-clustered.fbin"            # the sample holds the vectors now

PHASE=run
echo "phase=$PHASE"
if [ "$PART" = order ]; then
    for order in appended reversed shuffled; do
        # Each order is built in a fresh process, as in the LongEval order study.
        (cd "$STAGE" && srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/$order-resources.txt" \
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
    for arm in $ENGINE_ARMS; do
        (cd "$STAGE" && srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/$arm-resources.txt" \
            "$DEV_PYTHON" -P -m research.longeval.engine_arm "$STAGE/sample" "$OUTPUT/builds/engine" --arm "$arm" --seed "$SEED") \
            > "$OUTPUT/$arm-summary.json"
        printf 'built %s\n' "$arm"
    done
fi
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: public stream %s %s seed %s: %s\n' "$PART" "$BACKEND" "$SEED" "$OUTPUT"
