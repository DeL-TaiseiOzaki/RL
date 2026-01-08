#!/bin/bash
#SBATCH --job-name=sft-stackmath-9b
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --gpus=8
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err
#SBATCH --nodelist=megatpa-a3meganodeset-1
set -e

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Change to project directory
cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL

# Activate virtual environment
source .venv/bin/activate

# Use existing Python environment instead of building new venvs
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# Check GPU
nvidia-smi

# Run SFT training
echo "Starting SFT training..."
python examples/run_sft.py \
    --config examples/configs/sft_stackmath_9b.yaml

echo "=========================================="
echo "End time: $(date)"
echo "Training complete!"
echo "=========================================="
