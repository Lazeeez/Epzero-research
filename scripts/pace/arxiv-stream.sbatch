#!/usr/bin/env bash
# Screen A1: the arXiv stream's sample and its builds (docs/screen-a1-arxiv-stream.md, ARXIV_STREAM.md).
#   sbatch arxiv-stream-01/scripts/pace/arxiv-stream.sbatch sample VECTORS_JOB_ID
#   sbatch arxiv-stream-01/scripts/pace/arxiv-stream.sbatch build appended|reversed|shuffled epzero|faiss_hnsw 11|29|47
#   sbatch arxiv-stream-01/scripts/pace/arxiv-stream.sbatch repaired 11|29|47
# From the PACE LongEval checkout. Builds run on one exclusive Gold 6226 node, single-threaded (the repair's
# refine uses 8 threads); keep every job of the screen on that CPU type and never overlap other CPU types.
#SBATCH --job-name=arxiv-stream
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-cpu,ice-cpu
#SBATCH --constraint=gold6226
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=96G
#SBATCH --tmp=60G
#SBATCH --time=16:00:00
#SBATCH --output=arxiv-stream-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/arxiv-stream-01"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
SAMPLE="$ROOT/build/pace-arxiv-stream/sample"
MODE="${1:?mode: sample, build or repaired}"
case "$MODE" in
    sample) VECTORS="$ROOT/build/pace-arxiv-bge/jobs/${2:?vectors job id}/vectors"; ARM="sample" ;;
    build) ORDER="${2:?order}"; BACKEND="${3:?backend}"; SEED="${4:?seed}"
        case "$ORDER" in appended|reversed|shuffled) ;; *) echo 'order: appended, reversed or shuffled' >&2; exit 2 ;; esac
        case "$BACKEND" in epzero|faiss_hnsw) ;; *) echo 'backend: epzero or faiss_hnsw' >&2; exit 2 ;; esac
        ARM="$ORDER-$BACKEND-$SEED" ;;
    repaired) SEED="${2:?seed}"; ARM="repaired-epzero-$SEED" ;;
    *) echo 'mode: sample, build or repaired' >&2; exit 2 ;;
esac
case "${SEED:-11}" in 11|29|47) ;; *) echo 'seed must be 11, 29 or 47' >&2; exit 2 ;; esac
OUTPUT="$ROOT/build/pace-arxiv-stream/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
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
printf 'arm=%s\n' "$ARM" | tee "$OUTPUT/arm.txt"

PHASE=package
echo "phase=$PHASE"
(cd "$PACKAGE" && sha256sum --check --quiet --strict SHA256SUMS)
cp "$PACKAGE/SHA256SUMS" "$PACKAGE/manifest.json" "$OUTPUT/"

PHASE=environment
echo "phase=$PHASE"
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$UV" venv --python "$CPU_PYTHON" --seed "$STAGE/venv"
PYTHON="$STAGE/venv/bin/python"
"$PYTHON" -m pip install --only-binary=:all: epzero==0.6.0 faiss-cpu==1.16.0 numpy==2.4.6 pytest==8.3.5
"$PYTHON" -m pip freeze > "$OUTPUT/packages.txt"
{ date -u; hostname; lscpu; } > "$OUTPUT/hardware.txt"
grep -q 'Gold 6226' "$OUTPUT/hardware.txt" || { echo 'not a Gold 6226 node' >&2; exit 1; }

PHASE=tests
echo "phase=$PHASE"
(cd /tmp && "$PYTHON" -P -m pytest -q -p no:cacheprovider "$PACKAGE/research/arxiv_stream/tests" --junitxml="$OUTPUT/tests.xml")

if [ "$MODE" = sample ]; then
    PHASE=sample
    echo "phase=$PHASE"
    [ -f "$VECTORS/report.json" ] || { echo "no encoding at $VECTORS" >&2; exit 1; }
    [ ! -e "$SAMPLE" ] || { echo "the sample already exists: $SAMPLE" >&2; exit 1; }
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
    (cd /tmp && /usr/bin/time -v -o "$OUTPUT/resources.txt" "$PYTHON" -P -m research.arxiv_stream.stream sample \
        "$VECTORS" "$PACKAGE/inputs" "$SAMPLE")
    cp "$SAMPLE/sample.json" "$OUTPUT/"
else
    PHASE=copy-sample
    echo "phase=$PHASE"
    [ -f "$SAMPLE/sample.json" ] || { echo "run the sample job first" >&2; exit 1; }
    cp -a "$SAMPLE" "$STAGE/sample"                           # 8.6 GB onto node disk: the build reads it at random
    PHASE=build
    echo "phase=$PHASE"
    if [ "$MODE" = build ]; then
        (cd /tmp && /usr/bin/time -v -o "$OUTPUT/resources.txt" "$PYTHON" -P -m research.longeval.insertion_pilot run \
            "$STAGE/sample" "$OUTPUT/build" --order "$ORDER" --backend "$BACKEND" --seed "$SEED" --efs 100 200 400 800)
    else
        (cd /tmp && /usr/bin/time -v -o "$OUTPUT/resources.txt" "$PYTHON" -P -m research.arxiv_stream.stream repaired \
            "$STAGE/sample" "$OUTPUT/build" --seed "$SEED" --threads 8)
    fi
fi
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: arXiv stream %s: %s\n' "$ARM" "$OUTPUT"
