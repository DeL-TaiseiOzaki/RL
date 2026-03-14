#!/bin/bash
#SBATCH --job-name=grpo-16k-short
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --gpus=4
#SBATCH --output=logs/grpo-16k-short-%j.out
#SBATCH --error=logs/grpo-16k-short-%j.err
#SBATCH --exclude=megatpa-a3meganodeset-0

# GRPO training for baseline-model-instruct-16k-yarn with shorter context lengths
# Uses the 16k YaRN-extended model but trains with reduced context windows
#
# Usage:
#   sbatch run_grpo_16k_short_slurm.sh 4k
#   sbatch run_grpo_16k_short_slurm.sh 2k
#   sbatch run_grpo_16k_short_slurm.sh 1k

set -e

VARIANT="${1:?Usage: sbatch run_grpo_16k_short_slurm.sh <4k|2k|1k>}"

# Always use the 16k YaRN model
SFT_MODEL="ft-llm-team-mkj/baseline-model-instruct-16k-yarn"
TOKENIZER="ft-llm-team-mkj/baseline-model-instruct-16k-yarn"

# Max sequence length per variant
case "$VARIANT" in
  4k)  MAX_SEQ_LEN=4096 ;;
  2k)  MAX_SEQ_LEN=2048 ;;
  1k)  MAX_SEQ_LEN=1024 ;;
  *)   echo "Error: VARIANT must be 4k, 2k, or 1k"; exit 1 ;;
esac

# Generation settings
GEN_BATCH_SIZE=16
GPU_MEM_UTIL=0.6
TP_SIZE=1

CKPT_DIR="results/grpo_16k_yarn_ctx${VARIANT}"
WANDB_NAME="16k-yarn-ctx${VARIANT}-grpo-math"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Base Model: $SFT_MODEL (16k YaRN)"
echo "Context Length: $VARIANT ($MAX_SEQ_LEN tokens)"
echo "Gen Batch Size: $GEN_BATCH_SIZE"
echo "GPU Mem Util: $GPU_MEM_UTIL"
echo "TP Size: $TP_SIZE"
echo "Checkpoint Dir: $CKPT_DIR"
echo "Start time: $(date)"
echo "=========================================="

cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL
source .venv/bin/activate
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1
# NOTE: expandable_segments:True is incompatible with vLLM CuMemAllocator
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load environment variables from .env (for OPENAI_API_KEY etc.)
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

nvidia-smi

echo "Starting GRPO training (16k-yarn ctx${VARIANT} math)..."
python examples/run_grpo_math.py \
    --config examples/configs/grpo_baseline_math.yaml \
    ++cluster.gpus_per_node=4 \
    ++policy.model_name="$SFT_MODEL" \
    ++policy.tokenizer.name="$TOKENIZER" \
    ++policy.max_total_sequence_length="$MAX_SEQ_LEN" \
    ++policy.generation_batch_size="$GEN_BATCH_SIZE" \
    ++policy.generation.vllm_cfg.gpu_memory_utilization="$GPU_MEM_UTIL" \
    ++policy.generation.vllm_cfg.enforce_eager=True \
    ++policy.generation.vllm_cfg.tensor_parallel_size="$TP_SIZE" \
    ++checkpointing.checkpoint_dir="$CKPT_DIR" \
    ++logger.wandb.name="$WANDB_NAME" \
    ++logger.wandb.project="baseline-grpo-math"

echo "=========================================="
echo "End time: $(date)"
echo "Training complete! Checkpoint: $CKPT_DIR"
echo "=========================================="
