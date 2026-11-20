#!/usr/bin/env bash
# Build/correctness validation only; scientific timing jobs use exclusive allocations.
#SBATCH --job-name=longeval-bootstrap
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=longeval-bootstrap-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
STAGE="$ROOT/build/pace-longeval"
PYTHON="$STAGE/venv/bin/python"
OUTPUT="$STAGE/jobs/${SLURM_JOB_ID:?run inside a Slurm allocation}"
mkdir -p "$STAGE/jobs"
mkdir "$OUTPUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUTPUT/exit-code.txt"' EXIT
exec 9>"$STAGE/setup.lock"
flock -n 9 || { echo 'Another setup/build holds the research environment lock' >&2; exit 1; }
module purge
module load gcc/13.3.0
export PATH="$STAGE/venv/bin:$PATH"
export PIP_CACHE_DIR="$STAGE/pip-cache" PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INDEX=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PYTHON" scripts/pace/longeval-stage.py verify
cp longeval-source-manifest.json longeval-bootstrap-manifest.json "$STAGE/staged-files.json" "$STAGE/setup-receipt.txt" "$OUTPUT/"
{
    date -u
    hostname
    lscpu
    g++ --version
    "$PYTHON" --version
    git rev-parse HEAD
} > "$OUTPUT/hardware.txt"
"$PYTHON" -m pip install --no-index --no-deps "$STAGE"/wheelhouse/*.whl
"$PYTHON" -m pip install --no-index --no-build-isolation --no-deps --force-reinstall \
    --config-settings="build-dir=$OUTPUT/python-build" ./engine
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$OUTPUT/packages.txt"
# Python 3.12 safe extraction; all downloads were completed on the login node.
"$PYTHON" - "$STAGE/googletest-v1.15.2.tar.gz" "$OUTPUT/gtest" <<'PY'
import sys, tarfile
with tarfile.open(sys.argv[1]) as archive:
    archive.extractall(sys.argv[2], filter="data")
PY
cmake -S engine -B "$OUTPUT/cpp" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DEPZERO_BUILD_TESTS=ON -DEPZERO_BUILD_BENCH=OFF -DEPZERO_BUILD_NODE=OFF \
    -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST="$OUTPUT/gtest/googletest-1.15.2" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build "$OUTPUT/cpp" --parallel "$SLURM_CPUS_PER_TASK"
ctest --test-dir "$OUTPUT/cpp" --output-on-failure --output-junit "$OUTPUT/cpp-tests.xml"
"$PYTHON" -m pytest -q research/longeval/tests engine/python/tests/test_ir.py \
    engine/python/tests/test_epzero.py engine/python/tests/test_auto.py engine/python/tests/test_metadata.py \
    --basetemp="$OUTPUT/pytest-tmp" --junitxml="$OUTPUT/python-tests.xml"
"$PYTHON" -m research.longeval fixture "$OUTPUT/fixture"
"$PYTHON" -m research.longeval run --dataset "$OUTPUT/fixture/dataset.json" --output "$OUTPUT/pilot" --native-origin checkout-build
"$PYTHON" -m research.longeval verify "$OUTPUT/pilot" --dataset "$OUTPUT/fixture/dataset.json"
"$PYTHON" scripts/pace/longeval-stage.py source
printf 'PASS: build, native/Python tests and synthetic pilot; results: %s\n' "$OUTPUT"
