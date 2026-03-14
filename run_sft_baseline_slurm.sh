#!/bin/bash
#SBATCH --job-name=sft-baseline
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --gpus=8
#SBATCH --output=logs/sft-baseline-%j.out
#SBATCH --error=logs/sft-baseline-%j.err

# SFT training for baseline-model-instruct with math CoT data
#
# Usage:
#   sbatch run_sft_baseline_slurm.sh 8k
#   sbatch run_sft_baseline_slurm.sh 16k
#   sbatch run_sft_baseline_slurm.sh 32k

set -e

VARIANT="${1:?Usage: sbatch run_sft_baseline_slurm.sh <8k|16k|32k>}"

case "$VARIANT" in
  8k)  MODEL="ft-llm-team-mkj/baseline-model-instruct-8k-yarn" ;;
  16k) MODEL="ft-llm-team-mkj/baseline-model-instruct-16k-yarn" ;;
  32k) MODEL="ft-llm-team-mkj/baseline-model-instruct-32k-yarn" ;;
  *)   echo "Error: VARIANT must be 8k, 16k, or 32k"; exit 1 ;;
esac

CKPT_DIR="results/sft_baseline_${VARIANT}"
WANDB_NAME="baseline-${VARIANT}-sft-math"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL"
echo "Variant: $VARIANT"
echo "Checkpoint Dir: $CKPT_DIR"
echo "Start time: $(date)"
echo "=========================================="

cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL
source .venv/bin/activate
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

nvidia-smi

echo "Starting SFT training (baseline-${VARIANT} math CoT)..."
python examples/run_sft.py \
    --config examples/configs/sft_baseline_math.yaml \
    ++policy.model_name="$MODEL" \
    ++checkpointing.checkpoint_dir="$CKPT_DIR" \
    ++logger.wandb.name="$WANDB_NAME" \
    ++logger.wandb.project="baseline-sft-math"

echo "=========================================="
echo "End time: $(date)"
echo "Training complete! Checkpoint: $CKPT_DIR"
echo "=========================================="
