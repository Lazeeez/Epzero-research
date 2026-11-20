#!/usr/bin/env bash
#SBATCH --job-name=longeval-queries
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=coc-gpu,ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --tmp=80G
#SBATCH --time=04:00:00
#SBATCH --output=longeval-queries-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
ROOT="$PWD"
INPUTS="$ROOT/build/longeval-gpu-inputs/training-queries"
OUTPUT="$ROOT/build/pace-longeval-queries/jobs/${SLURM_JOB_ID:?run inside a Slurm allocation}"
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
finish() {
    code=$?
    if [ -n "${STAGE:-}" ]; then
        for name in packages.txt torch-install.json encoder-install.json; do
            if [ -f "$STAGE/$name" ]; then cp "$STAGE/$name" "$OUTPUT/" || true; fi
        done
        if [ -f "$STAGE/model/model-identity.json" ]; then
            cp "$STAGE/model/model-identity.json" "$OUTPUT/" || true
        fi
    fi
    printf '%s\n' "$code" > "$OUTPUT/exit-code.txt"
    exit "$code"
}
trap finish EXIT
source scripts/pace/longeval-node-env.sh
STAGE="$LONGEVAL_NODE_STAGE"
PYTHON="$STAGE/venv/bin/python"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
"$ROOT/build/pace-longeval/venv/bin/python" scripts/pace/longeval-gpu-verify.py
cp longeval-gpu-manifest.json "$OUTPUT/"
cp -r "$INPUTS" "$OUTPUT/inputs"
{
    printf 'node_stage=%s\n' "$STAGE"
    findmnt -T "$STAGE" -o SOURCE,FSTYPE,TARGET
    df -h "$STAGE"
} > "$OUTPUT/node-storage.txt"
bash scripts/pace/longeval-gpu-setup.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
{
    date -u
    hostname
    nvidia-smi
    "$PYTHON" -m pip freeze
} > "$OUTPUT/hardware-and-packages.txt"
srun "$PYTHON" - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise RuntimeError('Requires a working CUDA BF16 GPU')
if torch.cuda.get_device_properties(0).total_memory < 32 * 1024**3:
    raise RuntimeError('Requires at least 32 GiB VRAM for float32')
print('GPU:', torch.cuda.get_device_name(0), 'Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
for precision in bfloat16 float32; do
    srun "$PYTHON" -m research.longeval.query_diagnostic encode "$STAGE/model" "$OUTPUT/inputs" \
        "$OUTPUT/$precision" --dtype "$precision" --device cuda --threads 4
done
"$PYTHON" -m research.longeval.query_diagnostic compare "$OUTPUT/inputs" \
    "$OUTPUT/bfloat16" "$OUTPUT/float32" "$OUTPUT/query-precision-comparison.json"
"$PYTHON" scripts/pace/longeval-gpu-verify.py
printf 'PASS: training-query encodings verified; exact ranking sensitivity still required: %s\n' "$OUTPUT"
