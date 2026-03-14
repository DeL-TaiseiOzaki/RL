#!/bin/bash
# Convert sharded SFT checkpoint to HuggingFace format and upload to HF Hub
set -e

cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL
source .venv/bin/activate

SFT_CKPT_DIR="results/sft_qwen3_4b_structured/step_843/policy/weights/model"
OUTPUT_DIR="results/sft_qwen3_4b_structured/step_843/policy/weights/model/consolidated"
MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
HF_REPO="DeL-TaiseiOzaki/Qwen3-4B-Instruct-2507-sft-structured"

echo "=========================================="
echo "Step 1: Consolidating sharded safetensors"
echo "=========================================="

python 3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py \
    --model-name "$MODEL_NAME" \
    --input-dir "$SFT_CKPT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backend gloo

echo "=========================================="
echo "Step 2: Copying tokenizer files"
echo "=========================================="

# Copy tokenizer from the base model cache
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
tokenizer.save_pretrained('$OUTPUT_DIR')
print('Tokenizer saved to $OUTPUT_DIR')
"

echo "=========================================="
echo "Step 3: Uploading to HuggingFace Hub"
echo "=========================================="

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_REPO', exist_ok=True, private=True)
api.upload_folder(
    folder_path='$OUTPUT_DIR',
    repo_id='$HF_REPO',
    commit_message='Upload SFT fine-tuned Qwen3-4B for structured data generation',
)
print('Uploaded to https://huggingface.co/$HF_REPO')
"

echo "=========================================="
echo "Done! Model available at: $HF_REPO"
echo "=========================================="
