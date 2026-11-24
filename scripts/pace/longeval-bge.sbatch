#!/usr/bin/env bash
# E4 (second encoder): encode the LongEval-Sci collection and queries with BGE-base-en-v1.5.
# Design, pinned revision and predictions: research/longeval/docs/protocol/E4_SECOND_ENCODER.md.
#   sbatch longeval-bge-04/scripts/pace/longeval-bge.sbatch
# Downloads (document archives, model) and the environment stay on compute-node disk; the vectors are
# copied to the scratch build folder. They are derived from LongEval data and are never committed.
#SBATCH --job-name=longeval-bge
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --tmp=60G
#SBATCH --time=06:00:00
#SBATCH --output=longeval-bge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-bge-04"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
OUTPUT="$ROOT/build/pace-longeval-bge/jobs/${SLURM_JOB_ID:?requires a Slurm job}"
MIN_GPU_MIB=15000
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ]; then
        for name in packages.txt torch-install.json encoder-install.json; do
            if [ -f "$STAGE/$name" ]; then cp "$STAGE/$name" "$OUTPUT/" || true; fi
        done
        if [ -f "$STAGE/bge/report.json" ]; then cp "$STAGE/bge/report.json" "$OUTPUT/encoding-report.json" || true; fi
    fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    exit "$code"
}
trap finish EXIT

PHASE=package
echo "phase=$PHASE"
(cd "$PACKAGE" && sha256sum --check --quiet --strict SHA256SUMS)
cp "$PACKAGE/SHA256SUMS" "$PACKAGE/manifest.json" "$OUTPUT/"

PHASE=gpu
echo "phase=$PHASE"
# Job 7254797 (package -01, any GPU) stopped here with exit 127: the node had no nvidia-smi.
command -v nvidia-smi > /dev/null || { echo "nvidia-smi not found: this node has no NVIDIA driver tools" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUTPUT/gpu.txt"
MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[ "$MIB" -ge "$MIN_GPU_MIB" ] || { echo "GPU has ${MIB} MiB; at least ${MIN_GPU_MIB} MiB required" >&2; exit 1; }

PHASE=environment
echo "phase=$PHASE"
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
/usr/bin/python3 - "$STAGE" <<'PY'
import shutil, sys
free = shutil.disk_usage(sys.argv[1]).free
if free < 25 * 1024**3:
    raise SystemExit(f'Need at least 25 GiB free on the compute-node temporary disk; found {free / 1024**3:.0f} GiB')
PY
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE" PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
{
    date -u
    hostname
    lscpu
    printf 'node_stage=%s\n' "$STAGE"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
} > "$OUTPUT/hardware.txt"

PHASE=tests
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_bge_encoder.py" --junitxml="$OUTPUT/tests.xml")

PHASE=gpu-environment
echo "phase=$PHASE"
"$UV" venv --python "$CPU_PYTHON" --seed "$STAGE/venv"
GPU_PYTHON="$STAGE/venv/bin/python"
"$GPU_PYTHON" -m pip install --only-binary=:all: --report "$STAGE/torch-install.json" \
    torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
"$GPU_PYTHON" -m pip install --only-binary=:all: --report "$STAGE/encoder-install.json" \
    -r "$PACKAGE/research/longeval/encoder-requirements.txt"
"$GPU_PYTHON" -m pip check
"$GPU_PYTHON" -m pip freeze > "$STAGE/packages.txt"
"$GPU_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print(torch.cuda.get_device_name(0))'

PHASE=documents
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval fetch-documents \
    "$PACKAGE/release/files/downloads.json" "$STAGE/documents")

PHASE=model
echo "phase=$PHASE"
(cd /tmp && HF_HUB_OFFLINE=0 "$GPU_PYTHON" -P -m research.longeval.bge_encoder fetch "$STAGE/model") \
    > "$OUTPUT/model-identity.json"

PHASE=encode
echo "phase=$PHASE"
date -u +%FT%TZ
# Job 7254809 (package -02) spent its 4 hours here under `srun` with no progress line. The encoder now
# runs in the batch step, where the GPU check above ran, reports each stage and its rate, and stops
# early if the GPU is far too slow. GPU use is logged every minute while it runs.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 60 > "$OUTPUT/gpu-usage.csv" &
SMI_PID=$!
(cd /tmp && /usr/bin/time -v -o "$OUTPUT/encode-resources.txt" "$GPU_PYTHON" -P -m research.longeval.bge_encoder encode \
    "$STAGE/model" "$PACKAGE/inputs/bge-ids-snapshot-3.json.gz" "$STAGE/documents" "$PACKAGE/release" \
    "$STAGE/bge" --device cuda --batch-size 256)
kill "$SMI_PID" 2>/dev/null || true
date -u +%FT%TZ

PHASE=copy
echo "phase=$PHASE"
cp -a "$STAGE/bge" "$OUTPUT/vectors"
(cd "$OUTPUT" && find . -type f ! -name EVIDENCE-SHA256SUMS ! -name exit-code.txt -print0 | sort -z \
    | xargs -0 sha256sum > EVIDENCE-SHA256SUMS)
PHASE=done
printf 'PASS: BGE encoding completed: %s\n' "$OUTPUT"
