#!/bin/bash
#SBATCH --job-name=consolidate-grpo
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=200
#SBATCH --gpus=8
#SBATCH --output=logs/consolidate-grpo-%j.out
#SBATCH --error=logs/consolidate-grpo-%j.err
#SBATCH --exclude=megatpa-a3meganodeset-0

# Consolidate SFT checkpoint -> Run GRPO training
#
# Usage:
#   sbatch consolidate_and_grpo.sh 8k 1100
#   sbatch consolidate_and_grpo.sh 16k 1100
#   sbatch consolidate_and_grpo.sh 32k 1100

set -e

VARIANT="${1:?Usage: sbatch consolidate_and_grpo.sh <8k|16k|32k> <sft_step>}"
SFT_STEP="${2:?Provide SFT checkpoint step number}"

case "$VARIANT" in
  8k)
    BASE_MODEL="ft-llm-team-mkj/baseline-model-instruct-8k-yarn"
    MAX_SEQ_LEN=8192
    ;;
  16k)
    BASE_MODEL="ft-llm-team-mkj/baseline-model-instruct-16k-yarn"
    MAX_SEQ_LEN=16384
    ;;
  32k)
    BASE_MODEL="ft-llm-team-mkj/baseline-model-instruct-32k-yarn"
    MAX_SEQ_LEN=32768
    ;;
  *)
    echo "Error: VARIANT must be 8k, 16k, or 32k"; exit 1
    ;;
esac

SFT_CKPT_DIR="results/sft_baseline_${VARIANT}/step_${SFT_STEP}/policy/weights/model"
CONSOLIDATED_DIR="${SFT_CKPT_DIR}/consolidated"
GRPO_CKPT_DIR="results/grpo_baseline_${VARIANT}"
WANDB_NAME="baseline-${VARIANT}-grpo-math"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Variant: $VARIANT"
echo "SFT Step: $SFT_STEP"
echo "SFT Checkpoint: $SFT_CKPT_DIR"
echo "Max Seq Length: $MAX_SEQ_LEN"
echo "Start time: $(date)"
echo "=========================================="

cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL
source .venv/bin/activate
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1

# ==========================================
# Phase 1: Consolidate SFT checkpoint
# ==========================================
if [ -f "$CONSOLIDATED_DIR/model-00001-of-00001.safetensors" ]; then
    echo "Phase 1: Consolidation already done, skipping."
else
    echo "Phase 1: Consolidating sharded safetensors..."
    python 3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py \
        --model-name "$BASE_MODEL" \
        --input-dir "$SFT_CKPT_DIR" \
        --output-dir "$CONSOLIDATED_DIR" \
        --backend gloo

    echo "Copying tokenizer..."
    python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL', trust_remote_code=True)
tokenizer.save_pretrained('$CONSOLIDATED_DIR')
print('Tokenizer saved.')
"
fi

echo "Phase 1 complete. Consolidated model: $CONSOLIDATED_DIR"
nvidia-smi

# ==========================================
# Phase 2: Run GRPO training
# ==========================================
echo "Phase 2: Starting GRPO training (baseline-${VARIANT} math)..."
python examples/run_grpo_math.py \
    --config examples/configs/grpo_baseline_math.yaml \
    ++policy.model_name="$CONSOLIDATED_DIR" \
    ++policy.tokenizer.name="$BASE_MODEL" \
    ++policy.max_total_sequence_length="$MAX_SEQ_LEN" \
    ++checkpointing.checkpoint_dir="$GRPO_CKPT_DIR" \
    ++logger.wandb.name="$WANDB_NAME" \
    ++logger.wandb.project="baseline-grpo-math"

echo "=========================================="
echo "End time: $(date)"
echo "GRPO complete! Checkpoint: $GRPO_CKPT_DIR"
echo "=========================================="
