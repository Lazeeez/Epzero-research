#!/usr/bin/env bash
# Select a GPU partition/account with a GPU >=16 GiB after the cluster preflight.
# Model/tokenizer files and the dedicated CUDA environment must already be staged.
#SBATCH --job-name=longeval-encoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=longeval-encoder-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the isolated research checkout}"
: "${LONGEVAL_ENCODER_PYTHON:?set the Python path in the prepared CUDA encoder environment}"
: "${LONGEVAL_MODEL:?set the checksum-verified model directory}"
: "${LONGEVAL_DOCUMENT_SAMPLE:?set the document diagnostic JSONL file}"
: "${LONGEVAL_OUTPUT:?set a new output directory}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
hostname
nvidia-smi
git rev-parse HEAD
"$LONGEVAL_ENCODER_PYTHON" -m pip freeze
srun "$LONGEVAL_ENCODER_PYTHON" -m research.longeval encode-documents \
    "$LONGEVAL_MODEL" "$LONGEVAL_DOCUMENT_SAMPLE" --output "$LONGEVAL_OUTPUT" \
    --dtype bfloat16 --device cuda --threads 4 --max-length 8192
