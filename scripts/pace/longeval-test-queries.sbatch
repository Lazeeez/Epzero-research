#!/usr/bin/env bash
# Encode the 381 held-out test queries (FINAL_PROTOCOL_V1_CANDIDATE.md step 1). FP32, TF32 off, batch 1.
#SBATCH --job-name=longeval-testq
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=40G
#SBATCH --tmp=80G
#SBATCH --time=02:00:00
#SBATCH --output=longeval-testq-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated Epzero checkout}"
ROOT="$PWD"
PACKAGE="$ROOT/longeval-test-queries-03"
CPU_PYTHON="$ROOT/build/pace-longeval/venv/bin/python"
UV="$ROOT/build/pace-longeval/tools/bin/uv"
OUTPUT="$ROOT/build/pace-longeval-test-queries/jobs/${SLURM_JOB_ID:?requires Slurm}"
PHASE=initialization
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    printf 'exit_code=%s phase=%s\n' "$code" "$PHASE"
    if [ -n "${STAGE:-}" ]; then
        for name in packages.txt torch-install.json encoder-install.json; do
            if [ -f "$STAGE/$name" ]; then cp "$STAGE/$name" "$OUTPUT/" || true; fi
        done
        if [ -f "$STAGE/model/model-identity.json" ]; then cp "$STAGE/model/model-identity.json" "$OUTPUT/" || true; fi
    fi
    exit "$code"
}
trap finish EXIT

PHASE=package
echo "phase=$PHASE"
(cd "$PACKAGE" && sha256sum --check --quiet --strict SHA256SUMS)
cp "$PACKAGE/SHA256SUMS" "$PACKAGE/manifest.json" "$OUTPUT/"

PHASE=gpu-memory
echo "phase=$PHASE"
# Fail in seconds, before setup, on a GPU too small for the 16 GB FP32 model.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$OUTPUT/gpu.txt"
MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
[ "$MIB" -ge 20480 ] || { echo "GPU has ${MIB} MiB; at least 20480 MiB required" >&2; exit 1; }

PHASE=environment
echo "phase=$PHASE"
source "$PACKAGE/scripts/pace/longeval-node-env.sh"
STAGE="$LONGEVAL_NODE_STAGE"
cp -a "$PACKAGE" "$STAGE/package"
PACKAGE="$STAGE/package"
export PYTHONPATH="$PACKAGE" PIP_DISABLE_PIP_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
{
    printf 'node_stage=%s\n' "$STAGE"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
} > "$OUTPUT/node-storage.txt"

PHASE=tests
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m pytest -q -p no:cacheprovider \
    "$PACKAGE/research/longeval/tests/test_test_queries.py" --junitxml="$OUTPUT/tests.xml")
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.test_queries validate "$PACKAGE/inputs") | tee "$OUTPUT/inputs-validation.json"

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

PHASE=model
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval fetch-encoder "$STAGE/model")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
{
    date -u
    hostname
    nvidia-smi
} > "$OUTPUT/hardware.txt"
(cd /tmp && srun "$GPU_PYTHON" -P - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Requires a working CUDA GPU')
if torch.cuda.get_device_properties(0).total_memory < 20 * 1024**3:
    raise RuntimeError('Requires at least 20 GiB VRAM for the float32 model')
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
)

PHASE=encode
echo "phase=$PHASE"
(cd /tmp && srun "$GPU_PYTHON" -P -m research.longeval.test_queries encode \
    "$STAGE/model" "$PACKAGE/inputs" "$OUTPUT/encoding" --threads 4)

PHASE=verify
echo "phase=$PHASE"
(cd /tmp && "$CPU_PYTHON" -P -m research.longeval.test_queries verify "$PACKAGE/inputs" "$OUTPUT/encoding") \
    | tee "$OUTPUT/verification.json"
(cd "$OUTPUT/encoding" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
PHASE=done
printf 'PASS: test-query encodings verified and persisted: %s\n' "$OUTPUT"
