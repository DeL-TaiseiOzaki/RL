#!/bin/bash
#SBATCH --job-name=grpo-baseline
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --gpus=8
#SBATCH --output=logs/grpo-baseline-%j.out
#SBATCH --error=logs/grpo-baseline-%j.err
#SBATCH --exclude=megatpa-a3meganodeset-0

# GRPO training for baseline-model-instruct with math data
# Requires SFT checkpoint from run_sft_baseline_slurm.sh
#
# Usage:
#   sbatch run_grpo_baseline_slurm.sh 8k
#   sbatch run_grpo_baseline_slurm.sh 16k
#   sbatch run_grpo_baseline_slurm.sh 32k
#
# Or with custom SFT model path:
#   sbatch run_grpo_baseline_slurm.sh 8k /path/to/sft/model

set -e

VARIANT="${1:?Usage: sbatch run_grpo_baseline_slurm.sh <8k|16k|32k> [sft_model_path]}"
SFT_MODEL="${2:-}"

# If no SFT model path provided, use the base model directly
if [ -z "$SFT_MODEL" ]; then
    case "$VARIANT" in
      8k)  SFT_MODEL="ft-llm-team-mkj/baseline-model-instruct-8k-yarn" ;;
      16k) SFT_MODEL="ft-llm-team-mkj/baseline-model-instruct-16k-yarn" ;;
      32k) SFT_MODEL="ft-llm-team-mkj/baseline-model-instruct-32k-yarn" ;;
      *)   echo "Error: VARIANT must be 8k, 16k, or 32k"; exit 1 ;;
    esac
    echo "WARNING: No SFT model path provided. Using base model: $SFT_MODEL"
    echo "         For best results, provide SFT checkpoint path as second argument."
fi

# Tokenizer and max sequence length per variant
# Each variant uses its native context window for GRPO training.
case "$VARIANT" in
  8k)
    TOKENIZER="ft-llm-team-mkj/baseline-model-instruct-8k-yarn"
    MAX_SEQ_LEN=8192
    ;;
  16k)
    TOKENIZER="ft-llm-team-mkj/baseline-model-instruct-16k-yarn"
    MAX_SEQ_LEN=16384
    ;;
  32k)
    TOKENIZER="ft-llm-team-mkj/baseline-model-instruct-32k-yarn"
    MAX_SEQ_LEN=32768
    ;;
esac

# Generation settings: unified across all variants for fair comparison
# TP=1: each GPU runs an independent vLLM engine (DP=8), avoiding Ray placement group
# race conditions that cause KV cache OOM with TP>1. Matches official grpo_math_8B.yaml.
GEN_BATCH_SIZE=16
GPU_MEM_UTIL=0.6
TP_SIZE=1

CKPT_DIR="results/grpo_baseline_${VARIANT}"
WANDB_NAME="baseline-${VARIANT}-grpo-math"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "SFT Model: $SFT_MODEL"
echo "Tokenizer: $TOKENIZER"
echo "Variant: $VARIANT"
echo "Max Seq Length: $MAX_SEQ_LEN"
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

echo "Starting GRPO training (baseline-${VARIANT} math)..."
python examples/run_grpo_math.py \
    --config examples/configs/grpo_baseline_math.yaml \
    ++cluster.gpus_per_node=8 \
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
