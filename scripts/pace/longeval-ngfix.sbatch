#!/usr/bin/env bash
# Isolated, pinned NGFix correctness and small real-vector integration screen.
#SBATCH --job-name=longeval-ngfix
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
#SBATCH --output=longeval-ngfix-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated Epzero checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-ngfix-screen-02"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
OUTPUT="$ROOT/build/pace-longeval-ngfix/jobs/${SLURM_JOB_ID:?requires Slurm}"
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    if [ -n "${STAGE:-}" ]; then
        for name in fixture sample; do
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
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_NO_INDEX=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PACKAGE"
"$PYTHON" -m research.longeval.ngfix_screen package-verify "$PACKAGE"
for feature in avx512f avx512bw avx512dq avx512vl; do
    grep -qw "$feature" /proc/cpuinfo || { echo "Missing $feature; no corpus download attempted" >&2; exit 1; }
done
[ "$(g++ -dumpfullversion)" = '13.3.0' ] || { echo 'GCC13.3.0 required by pinned header closure' >&2; exit 1; }
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
cd "$PACKAGE"
export PYTHONPATH="$PACKAGE"
cp manifest.json "$OUTPUT/package-manifest.json"
{
    date -u
    hostname
    lscpu
    g++ --version
    "$PYTHON" --version
    "$PYTHON" -m pip freeze
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
} > "$OUTPUT/hardware-and-packages.txt"
printf '%s\n' 'g++ -O3 -std=c++20 -fopenmp -march=skylake-avx512 -Ivendor/ngfix -Ivendor/ngfix/rabitqlib -Ivendor/boost research/longeval/ngfix_driver.cpp -o NODE_STAGE/ngfix-driver' > "$OUTPUT/compile-command.txt"
g++ -O3 -std=c++20 -fopenmp -march=skylake-avx512 \
    -Ivendor/ngfix -Ivendor/ngfix/rabitqlib -Ivendor/boost \
    research/longeval/ngfix_driver.cpp -o "$STAGE/ngfix-driver" 2> "$OUTPUT/compile.log"
sha256sum "$STAGE/ngfix-driver" > "$OUTPUT/binary.sha256"
srun --cpu-bind=cores "$PYTHON" -m pytest -q research/longeval/tests/test_ngfix_screen.py \
    -o "cache_dir=$STAGE/pytest-cache" --basetemp="$STAGE/tests" --junitxml="$OUTPUT/tests.xml"
"$PYTHON" -m research.longeval.ngfix_screen fixture "$STAGE/fixture-inputs"
srun --cpu-bind=cores "$PYTHON" -m research.longeval.ngfix_screen native-negative-checks \
    "$STAGE/ngfix-driver" "$STAGE/fixture-inputs" "$STAGE/native-negative-tests"
cp "$STAGE/native-negative-tests/verification.json" "$OUTPUT/native-negative-tests.json"
srun --cpu-bind=cores "$PYTHON" -m research.longeval.ngfix_screen execute \
    "$STAGE/ngfix-driver" "$STAGE/fixture-inputs" "$STAGE/fixture"
mkdir "$OUTPUT/fixture"
cp "$STAGE/fixture/verification.json" "$STAGE/fixture/queries.tsv" "$STAGE/fixture/rankings.tsv" "$STAGE/fixture/stages.tsv" "$OUTPUT/fixture/"
# Large archive, extraction, sample matrices, binary and every graph stay node-local.
"$PYTHON" -m research.longeval.ngfix_screen download "$STAGE/dense"
srun --cpu-bind=cores "$PYTHON" -m research.longeval.ngfix_screen sample \
    "$STAGE/dense/snapshot-1" "$PACKAGE/inputs" "$STAGE/sample-inputs"
srun --cpu-bind=cores /usr/bin/time -v -o "$OUTPUT/sample-resources.txt" \
    "$PYTHON" -m research.longeval.ngfix_screen execute \
    "$STAGE/ngfix-driver" "$STAGE/sample-inputs" "$STAGE/sample"
mkdir "$OUTPUT/sample"
cp "$STAGE/sample/verification.json" "$STAGE/sample/queries.tsv" "$STAGE/sample/rankings.tsv" "$STAGE/sample/stages.tsv" "$OUTPUT/sample/"
cp "$STAGE/dense/snapshot-1/ingestion.json" "$OUTPUT/snapshot-1-ingestion.json"
"$PYTHON" -m research.longeval.ngfix_screen package-verify "$PACKAGE"
printf 'PASS: NGFix synthetic and sample controls verified; full-data and novel-method claims remain untested: %s\n' "$OUTPUT"
