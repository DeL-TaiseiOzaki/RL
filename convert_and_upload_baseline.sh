#!/bin/bash
#SBATCH --job-name=convert-baseline
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/convert-baseline-%j.out
#SBATCH --error=logs/convert-baseline-%j.err
#SBATCH --exclude=megatpa-a3meganodeset-0

# Convert sharded checkpoint to HuggingFace format and upload
#
# Usage:
#   sbatch convert_and_upload_baseline.sh grpo 8k 200 best
#   sbatch convert_and_upload_baseline.sh grpo 16k 600 final

set -e

STAGE="${1:?Usage: sbatch convert_and_upload_baseline.sh <sft|grpo> <8k|16k|32k> <step> <best|final>}"
VARIANT="${2:?Provide variant: 8k, 16k, or 32k}"
STEP="${3:?Provide checkpoint step number}"
TAG="${4:?Provide tag: best or final}"

case "$VARIANT" in
  8k)  BASE_MODEL="ft-llm-team-mkj/baseline-model-instruct-8k-yarn" ;;
  16k) BASE_MODEL="ft-llm-team-mkj/baseline-model-instruct-16k-yarn" ;;
  32k) BASE_MODEL="ft-llm-team-mkj/baseline-model-instruct-32k-yarn" ;;
  *)   echo "Error: VARIANT must be 8k, 16k, or 32k"; exit 1 ;;
esac

CKPT_DIR="results/${STAGE}_baseline_${VARIANT}/step_${STEP}/policy/weights/model"
CONSOLIDATED_DIR="${CKPT_DIR}/consolidated"
HF_REPO="ft-llm-team-mkj/baseline-model-instruct-${VARIANT}-yarn-${STAGE}-math-${TAG}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Stage: $STAGE"
echo "Variant: $VARIANT"
echo "Step: $STEP"
echo "Tag: $TAG"
echo "Checkpoint: $CKPT_DIR"
echo "HF Repo: $HF_REPO"
echo "Start time: $(date)"
echo "=========================================="

cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL
source .venv/bin/activate

# ==========================================
# Step 1: Consolidate sharded safetensors
# ==========================================
if [ -f "$CONSOLIDATED_DIR/model-00001-of-00001.safetensors" ]; then
    echo "Step 1: Consolidation already done, skipping."
else
    echo "Step 1: Consolidating sharded safetensors..."
    python 3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py \
        --model-name "$BASE_MODEL" \
        --input-dir "$CKPT_DIR" \
        --output-dir "$CONSOLIDATED_DIR" \
        --backend gloo
fi

# ==========================================
# Step 2: Copy tokenizer
# ==========================================
echo "Step 2: Copying tokenizer..."

python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$BASE_MODEL', trust_remote_code=True)
tokenizer.save_pretrained('$CONSOLIDATED_DIR')
print('Tokenizer saved.')
"

# ==========================================
# Step 3: Upload to HuggingFace Hub
# ==========================================
echo "Step 3: Uploading to HuggingFace Hub..."

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_REPO', exist_ok=True, private=False)
api.upload_folder(
    folder_path='$CONSOLIDATED_DIR',
    repo_id='$HF_REPO',
    commit_message='Upload ${STAGE} fine-tuned baseline-${VARIANT} for math (step ${STEP}, ${TAG})',
)
print('Uploaded to https://huggingface.co/$HF_REPO')
"

echo "=========================================="
echo "Done! Model available at: $HF_REPO"
echo "End time: $(date)"
echo "=========================================="
