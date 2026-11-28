#!/usr/bin/env bash
# Screen A2b: NGFix against the arrival bias (docs/screen-a2b-ngfix.md, NGFIX_ARRIVAL.md).
#   sbatch ngfix-arrival-01/scripts/pace/ngfix-arrival.sbatch       (from the PACE LongEval checkout)
# Compiles the unchanged NGFix adapter from the pinned source of package longeval-ngfix-screen-02 and runs it on the
# GloVe and LongEval samples, as arrived and shuffled, one after another on one exclusive Gold 6226 node (one thread).
# Vectors, graphs and rankings stay on node disk; only aggregates and timings are copied back.
#SBATCH --job-name=ngfix-arrival
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
#SBATCH --output=ngfix-arrival-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/ngfix-arrival-01"
PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
OUTPUT="$ROOT/build/pace-ngfix-arrival/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT

PHASE=package
echo "phase=$PHASE"
(cd "$PACKAGE" && sha256sum --check --quiet --strict SHA256SUMS)
cp "$PACKAGE/SHA256SUMS" "$PACKAGE/manifest.json" "$OUTPUT/"

PHASE=environment
echo "phase=$PHASE"
module purge
module load gcc/13.3.0
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_NO_INDEX=1
for feature in avx512f avx512bw avx512dq avx512vl; do
    grep -qw "$feature" /proc/cpuinfo || { echo "missing $feature" >&2; exit 1; }
done
[ "$(g++ -dumpfullversion)" = '13.3.0' ] || { echo 'GCC 13.3.0 required by the pinned header closure' >&2; exit 1; }
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE"
{ date -u; hostname; lscpu; g++ --version; "$PYTHON" --version; } > "$OUTPUT/hardware.txt"

PHASE=compile
echo "phase=$PHASE"
(cd "$PACKAGE" && g++ -O3 -std=c++20 -fopenmp -march=skylake-avx512 \
    -Ivendor/ngfix -Ivendor/ngfix/rabitqlib -Ivendor/boost \
    research/longeval/ngfix_driver.cpp -o "$STAGE/ngfix-driver" 2> "$OUTPUT/compile.log")
sha256sum "$STAGE/ngfix-driver" | cut -d' ' -f1 > "$OUTPUT/binary.sha256"

PHASE=tests
echo "phase=$PHASE"
(cd /tmp && "$PYTHON" -P -m pytest -q -p no:cacheprovider "$PACKAGE/research/longeval/tests/test_ngfix_arrival.py" \
    --junitxml="$OUTPUT/tests.xml")

for sample in glove longeval; do
    for order in appended shuffled; do
        RUN="$sample-$order"
        PHASE="prepare-$RUN"
        echo "phase=$PHASE"
        (cd /tmp && "$PYTHON" -P -m research.longeval.ngfix_arrival prepare "$PACKAGE/inputs/$sample" "$STAGE/prep-$RUN" --order "$order")
        PHASE="ngfix-$RUN"
        echo "phase=$PHASE"
        /usr/bin/time -v -o "$OUTPUT/$RUN-resources.txt" "$STAGE/ngfix-driver" "$STAGE/prep-$RUN/base.fbin" \
            "$STAGE/prep-$RUN/queries.fbin" "$STAGE/prep-$RUN/calibration_top200.ibin" "$STAGE/run-$RUN"
        PHASE="score-$RUN"
        echo "phase=$PHASE"
        mkdir "$OUTPUT/$RUN"
        (cd /tmp && "$PYTHON" -P -m research.longeval.ngfix_arrival score "$STAGE/prep-$RUN" "$STAGE/run-$RUN" \
            "$PACKAGE/inputs/$sample" "$OUTPUT/$RUN/scored.json")
        cp "$STAGE/prep-$RUN/prepared.json" "$STAGE/run-$RUN/stages.tsv" "$OUTPUT/$RUN/"
        rm -rf "$STAGE/prep-$RUN" "$STAGE/run-$RUN"
    done
done
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: NGFix arrival screen, four runs scored: %s\n' "$OUTPUT"
