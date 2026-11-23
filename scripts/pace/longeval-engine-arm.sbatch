#!/usr/bin/env bash
# The repaired-Epzero arm at full scale (exploratory follow-up). See LONGEVAL_ENGINE_ARM.md.
#   sbatch longeval-engine-arm-01/scripts/pace/longeval-engine-arm.sbatch SEED
# One job builds all of snapshot 3 in Epzero's product path three ways (grown, repaired, rebuilt) and
# searches the test and confirmation queries. Epzero is built on the node from the packaged source.
# All archive downloads, extracted matrices and indexes stay on compute-node disk. No judgments.
#SBATCH --job-name=longeval-engine-arm
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --tmp=160G
#SBATCH --time=12:00:00
#SBATCH --output=longeval-engine-arm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-engine-arm-01"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
OUTPUT="$ROOT/build/pace-longeval-engine-arm/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
SEED="${1:?seed}"
case "$SEED" in 11|29|47) ;; *) echo 'seed must be 11, 29 or 47' >&2; exit 2 ;; esac
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ]; then
        if [ -f "$STAGE/sample/sample.json" ]; then cp "$STAGE/sample/sample.json" "$OUTPUT/" || true; fi
        if [ -f "$STAGE/dev-packages.txt" ]; then cp "$STAGE/dev-packages.txt" "$OUTPUT/" || true; fi
    fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT
printf 'seed=%s arms=grown,repaired,rebuilt\n' "$SEED" | tee "$OUTPUT/arm.txt"

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
"$CPU_PYTHON" - "$STAGE" <<'PY'
import shutil, sys
if shutil.disk_usage(sys.argv[1]).free < 125 * 1024**3:
    raise SystemExit('Need at least 125 GiB free on compute-node temporary disk before dense downloads')
PY
{
    date -u
    hostname
    lscpu
    printf 'node_stage=%s\n' "$STAGE"
    df -h "$STAGE"
    gcc --version | head -n 1
} > "$OUTPUT/hardware.txt"

PHASE=engine
echo "phase=$PHASE"
# The development Epzero, built from the packaged source into a node-local environment.
"$UV" venv --python "$CPU_PYTHON" --seed "$STAGE/dev"
DEV_PYTHON="$STAGE/dev/bin/python"
# The versions that built this source locally (scikit-build-core, pybind11, CMake 3.31) and the study's numpy.
"$DEV_PYTHON" -m pip install --only-binary=:all: numpy==2.4.6 scikit-build-core==1.1.2 pybind11==3.1.0 cmake==3.31.6 ninja==1.11.1.4
export PATH="$STAGE/dev/bin:$PATH"
(cd "$PACKAGE/engine" && "$DEV_PYTHON" -m pip install --no-build-isolation --no-deps . )
"$DEV_PYTHON" -m pip freeze > "$STAGE/dev-packages.txt"
"$DEV_PYTHON" - <<'PY' | tee "$OUTPUT/engine-check.json"
import json, inspect, numpy as np, epzero
index = epzero.Index(dim=4, max_elements=10, extra_backlinks=4)
index.add(np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32))
index.refine_links(threads=1, ef=0, first=5)
print(json.dumps({'epzero': epzero.__file__, 'extra_backlinks': True, 'refine_first': True}))
PY

PHASE=tests
echo "phase=$PHASE"
(cd "$STAGE" && "$CPU_PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_engine_arm_analysis.py" "$PACKAGE/research/longeval/tests/test_post_freeze_analyses.py" \
    --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml")

PHASE=prepare
echo "phase=$PHASE"
(cd "$STAGE" && srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/preparation-resources.txt" \
    "$CPU_PYTHON" -P -m research.longeval prepare-dense "$STAGE/dense" --output "$STAGE/preparation")

PHASE=sample
echo "phase=$PHASE"
# Vectors in insertion-ready form, arrival groups, and the exact neighbours of the test and the
# confirmation queries, each checked against the study's own exact references.
(cd "$STAGE" && srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/sample-resources.txt" \
    "$CPU_PYTHON" -P -m research.longeval.insertion_pilot prepare-full "$STAGE/dense" "$PACKAGE/inputs" "$STAGE/sample" \
    --draw 0) > "$OUTPUT/sample-summary.json"
cat "$OUTPUT/sample-summary.json"

PHASE=run
echo "phase=$PHASE"
for arm in grown repaired rebuilt; do
    # Each arm is built in a fresh process; only the node-local sample is shared.
    (cd "$STAGE" && srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/$arm-resources.txt" \
        "$DEV_PYTHON" -P -m research.longeval.engine_arm "$STAGE/sample" "$OUTPUT/builds" --arm "$arm" --seed "$SEED") \
        > "$OUTPUT/$arm-summary.json"
    printf 'built %s\n' "$arm"
done
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: repaired-Epzero arm completed for seed %s: %s\n' "$SEED" "$OUTPUT"
