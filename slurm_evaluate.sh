#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH -c 4                               # 4 cores per task
#SBATCH -t 00-03:00:00
#SBATCH -o logs/output_%j.log
#SBATCH -e logs/error_%j.log
#SBATCH -p sapphire

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

# Don't use this for now, just stick with gpt-4o-mini
# python src/evaluate.py \
#     --model_name gemma3-4b-it \
#     --split val \
#     --mode descriptive \
#     --api_key $GEMINI_KEY \
#     --eval_model gemma-3-27b-it

python src/evaluate.py \
    --model_name gemma3-4b-it \
    --split val \
    --mode descriptive \
    --api_key $OPENAI_KEY \
    --eval_model gpt-4o-mini \
    --parse_mode program_synthesis



