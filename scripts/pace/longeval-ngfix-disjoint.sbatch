#!/usr/bin/env bash
# Same published competitor; independent fitting and effort selection.
#SBATCH --job-name=longeval-ngsplit
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --tmp=48G
#SBATCH --time=02:00:00
#SBATCH --output=longeval-ngsplit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated Epzero checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-ngfix-disjoint-01"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
OUTPUT="$ROOT/build/pace-longeval-ngfix-disjoint/jobs/${SLURM_JOB_ID:?requires Slurm}"
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    if [ -n "${STAGE:-}" ]; then
        for name in fixture fold-0 fold-1 fold-2; do
            if [ -f "$STAGE/$name.log" ]; then cp "$STAGE/$name.log" "$OUTPUT/" || true; fi
        done
    fi
    exit "$code"
}
trap finish EXIT
exec 9>"$ROOT/build/pace-longeval/setup.lock"
flock -sn 9 || { echo 'Research environment is being rebuilt' >&2; exit 1; }
module purge
module load gcc/13.3.0
PHASE=environment
echo "phase=$PHASE"
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PACKAGE"
"$PYTHON" -P -m research.longeval.ngfix_screen package-verify "$PACKAGE"
for feature in avx512f avx512bw avx512dq avx512vl; do
    grep -qw "$feature" /proc/cpuinfo || { echo "Missing $feature" >&2; exit 1; }
done
[ "$(g++ -dumpfullversion)" = '13.3.0' ] || { echo 'GCC13.3.0 required' >&2; exit 1; }
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE"
cp "$PACKAGE/manifest.json" "$OUTPUT/package-manifest.json"
"$PYTHON" -P -m research.longeval.ngfix_disjoint protocol "$PACKAGE/inputs" "$OUTPUT/protocol.json"
cmp "$PACKAGE/protocol.json" "$OUTPUT/protocol.json"
{
    date -u
    hostname
    lscpu
    g++ --version
    "$PYTHON" --version
    "$PYTHON" -P -m pip freeze
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
} > "$OUTPUT/hardware-and-packages.txt"
PHASE=compile
echo "phase=$PHASE"
printf '%s\n' 'g++ -O3 -std=c++20 -fopenmp -march=skylake-avx512; identical source and include closure to screen03' > "$OUTPUT/compile-command.txt"
g++ -O3 -std=c++20 -fopenmp -march=skylake-avx512 \
    -I"$PACKAGE/vendor/ngfix" -I"$PACKAGE/vendor/ngfix/rabitqlib" -I"$PACKAGE/vendor/boost" \
    "$PACKAGE/research/longeval/ngfix_driver.cpp" -o "$STAGE/ngfix-driver" 2> "$OUTPUT/compile.log"
sha256sum "$STAGE/ngfix-driver" > "$OUTPUT/binary.sha256"
# /tmp exists before and after job-local mount setup. All following paths are absolute.
cd /tmp
PHASE=tests
echo "phase=$PHASE"
export LONGEVAL_TEST_PACKAGE="$PACKAGE"
srun --chdir=/tmp --cpu-bind=cores "$PYTHON" -P -m pytest -q \
    "$PACKAGE/research/longeval/tests/test_ngfix_screen.py" \
    "$PACKAGE/research/longeval/tests/test_ngfix_disjoint.py" \
    "$PACKAGE/scripts/pace/tests/test_ngfix_launch.py" \
    -o "cache_dir=$STAGE/pytest-cache" --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml"
PHASE=native_fixture
echo "phase=$PHASE"
"$PYTHON" -P -m research.longeval.ngfix_screen fixture "$STAGE/fixture-parent"
srun --chdir=/tmp --cpu-bind=cores "$PYTHON" -P -m research.longeval.ngfix_screen native-negative-checks \
    "$STAGE/ngfix-driver" "$STAGE/fixture-parent" "$STAGE/native-negative-tests"
cp "$STAGE/native-negative-tests/verification.json" "$OUTPUT/native-negative-tests.json"
"$PYTHON" -P -m research.longeval.ngfix_disjoint prepare "$STAGE/fixture-parent" 0 "$STAGE/fixture-inputs"
srun --chdir=/tmp --cpu-bind=cores "$PYTHON" -P -m research.longeval.ngfix_disjoint execute \
    "$STAGE/ngfix-driver" "$STAGE/fixture-parent" "$STAGE/fixture-inputs" "$STAGE/fixture"
retain() {
    name="$1"
    mkdir "$OUTPUT/$name"
    for file in verification.json disjoint-verification.json queries.tsv rankings.tsv stages.tsv; do
        cp "$STAGE/$name/$file" "$OUTPUT/$name/"
    done
    cp "$STAGE/$name-inputs/protocol.json" "$OUTPUT/$name/input-protocol.json"
    cp "$STAGE/$name-inputs/report.json" "$OUTPUT/$name/input-report.json"
}
retain fixture
# Large downloads, vectors and all graph states stay on compute-node local temp disk.
PHASE=download
echo "phase=$PHASE"
"$PYTHON" -P -m research.longeval.ngfix_screen download "$STAGE/dense"
srun --chdir=/tmp --cpu-bind=cores "$PYTHON" -P -m research.longeval.ngfix_screen sample \
    "$STAGE/dense/snapshot-1" "$PACKAGE/inputs" "$STAGE/parent-inputs"
cp "$STAGE/dense/snapshot-1/ingestion.json" "$OUTPUT/snapshot-1-ingestion.json"
"$PYTHON" -P -m research.longeval.ngfix_disjoint protocol "$STAGE/parent-inputs" "$STAGE/protocol.json"
cmp "$PACKAGE/protocol.json" "$STAGE/protocol.json"
for fold in 0 1 2; do
    PHASE="fold-$fold"
    echo "phase=$PHASE"
    "$PYTHON" -P -m research.longeval.ngfix_disjoint prepare "$STAGE/parent-inputs" "$fold" "$STAGE/fold-$fold-inputs"
    srun --chdir=/tmp --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/fold-$fold-resources.txt" \
        "$PYTHON" -P -m research.longeval.ngfix_disjoint execute \
        "$STAGE/ngfix-driver" "$STAGE/parent-inputs" "$STAGE/fold-$fold-inputs" "$STAGE/fold-$fold"
    retain "fold-$fold"
done
"$PYTHON" -P -m research.longeval.ngfix_screen package-verify "$PACKAGE"
printf 'PASS: all three disjoint fit/tune/development screens verified; no novel-method or full-data claim: %s\n' "$OUTPUT"
