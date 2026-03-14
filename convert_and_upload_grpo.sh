#!/bin/bash
#SBATCH --job-name=convert-grpo
#SBATCH --partition=a3megatpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --output=logs/convert-grpo-%j.out
#SBATCH --error=logs/convert-grpo-%j.err
#SBATCH --exclude=megatpa-a3meganodeset-0

# Convert sharded GRPO checkpoint to HuggingFace format (fp16) and upload to HF Hub
set -e

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURM_NODELIST:-$(hostname)}"
echo "Start time: $(date)"
echo "=========================================="

cd /home/usr_ext_taisei_ozaki_ccoe_toyota/RL
source .venv/bin/activate

GRPO_CKPT_DIR="results/grpo_qwen3_4b_structured/step_457/policy/weights/model"
CONSOLIDATED_DIR="results/grpo_qwen3_4b_structured/step_457/policy/weights/model/consolidated"
FP16_DIR="results/grpo_qwen3_4b_structured/step_457/policy/weights/model/fp16"
MODEL_NAME="DeL-TaiseiOzaki/Qwen3-4B-Instruct-2507-sft-structured"
HF_REPO="DeL-TaiseiOzaki/Qwen3-4B-Instruct-2507-grpo-structured"

# ==========================================
# Step 1: Consolidate sharded safetensors (skip if already done)
# ==========================================
if [ -f "$CONSOLIDATED_DIR/model-00001-of-00001.safetensors" ]; then
    echo "Step 1: Consolidation already done, skipping."
else
    echo "Step 1: Consolidating sharded safetensors..."
    python 3rdparty/Automodel-workspace/Automodel/tools/offline_hf_consolidation.py \
        --model-name "$MODEL_NAME" \
        --input-dir "$GRPO_CKPT_DIR" \
        --output-dir "$CONSOLIDATED_DIR" \
        --backend gloo
fi

# ==========================================
# Step 2: Convert to fp16
# ==========================================
echo "Step 2: Converting to fp16..."

python -c "
import json
import os
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
import torch

consolidated_dir = '$CONSOLIDATED_DIR'
fp16_dir = '$FP16_DIR'
os.makedirs(fp16_dir, exist_ok=True)

input_file = os.path.join(consolidated_dir, 'model-00001-of-00001.safetensors')

print('Converting tensors to fp16...')
tensors = {}
with safe_open(input_file, framework='pt', device='cpu') as f:
    keys = list(f.keys())
    total = len(keys)
    for i, key in enumerate(keys):
        t = f.get_tensor(key)
        if t.is_floating_point() and t.dtype != torch.float16:
            tensors[key] = t.to(torch.float16)
        else:
            tensors[key] = t
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f'  {i + 1}/{total} tensors converted')

print('Saving fp16 safetensors...')
save_file(tensors, os.path.join(fp16_dir, 'model.safetensors'))
del tensors
print('Saved.')

# Copy and update config
with open(os.path.join(consolidated_dir, 'config.json'), 'r') as f:
    config = json.load(f)
config['torch_dtype'] = 'float16'
if 'dtype' in config:
    config['dtype'] = 'float16'
with open(os.path.join(fp16_dir, 'config.json'), 'w') as f:
    json.dump(config, f, indent=2)

# Copy generation_config
gen_cfg = os.path.join(consolidated_dir, 'generation_config.json')
if os.path.exists(gen_cfg):
    shutil.copy2(gen_cfg, fp16_dir)

print('Config files updated.')
"

# ==========================================
# Step 3: Copy tokenizer
# ==========================================
echo "Step 3: Copying tokenizer..."

python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
tokenizer.save_pretrained('$FP16_DIR')
print('Tokenizer saved.')
"

# ==========================================
# Step 4: Upload to HuggingFace Hub
# ==========================================
echo "Step 4: Uploading to HuggingFace Hub..."

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_REPO', exist_ok=True, private=True)
api.upload_folder(
    folder_path='$FP16_DIR',
    repo_id='$HF_REPO',
    commit_message='Upload GRPO fine-tuned Qwen3-4B (fp16) for structured data generation',
)
print('Uploaded to https://huggingface.co/$HF_REPO')
"

echo "=========================================="
echo "Done! Model available at: $HF_REPO"
echo "End time: $(date)"
echo "=========================================="
