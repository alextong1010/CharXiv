#!/bin/bash
#SBATCH --job-name=generate
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 00-12:00:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log
#SBATCH --gpus=1
#SBATCH -p gpu_test

#SBATCH --account=ydu_lab
#SBATCH --mem=32GB


# Load necessary modules
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate lmvf

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found. Please create one with your API keys."
fi

# # Add requeue detection
# echo "=== Job Start/Restart: $(date) ==="
# echo "SLURM_RESTART_COUNT: ${SLURM_RESTART_COUNT:-0}"
# if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
#     echo "This job has been requeued ${SLURM_RESTART_COUNT} times"
#     echo "Auto-resume will be enabled in Python script"
# fi

# Export SLURM_RESTART_COUNT to ensure it's available to Python script
# export SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}

# echo "Starting Python script with auto-resume detection..."
# python src/generate.py \
#     --model_name gemma3-4b-it \
#     --split val \
#     --mode descriptive \
#     --model_path models/gemma-3-4b-it \
#     --model_api $GEMINI_KEY \
#     --parse_mode qcond_parse

# python src/generate.py \
#     --model_name gemma3-27b-it \
#     --split val \
#     --mode descriptive \
#     --model_path models/gemma-3-27b-it \
#     --model_api $GEMINI_KEY2 \
#     --parse_mode parse

# Multiple keys
# python src/generate.py \
#     --model_name gemma3-27b-it \
#     --split val \
#     --mode descriptive \
#     --model_path models/gemma-3-27b-it \
#     --model_api $GEMINI_KEY $GEMINI_KEY2 \
#     --parse_mode program_synthesis

python src/generate.py \
    --model_name gpt-4o-mini-2024-07-18 \
    --split val \
    --mode descriptive \
    --model_path gpt-4o-mini-2024-07-18 \
    --model_api $OPENAI_KEY \
    --parse_mode qcond_parse

# module load gcc/14.2.0-fasrc01
# module load cuda/12.4.1-fasrc01
# module load cudnn/9.5.1.17_cuda12-fasrc01 
# export HF_HOME=/n/netscratch/ydu_lab/Lab/alex/.cache/huggingface
# export CUDA_VISIBLE_DEVICES=0

# python src/generate.py \
#     --model_name deepseek-vl-7b-chat \
#     --split val \
#     --mode descriptive \
#     --model_path deepseek-ai/deepseek-vl-7b-chat \
#     --parse_mode parse
    